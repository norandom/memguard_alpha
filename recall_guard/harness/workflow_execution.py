"""Execution-local controls and evidence; no mutable state is installed on clients."""

from dataclasses import replace
from datetime import UTC, datetime
from threading import Lock
from time import monotonic

from recall_guard.core.readiness_types import CancellationToken, ReadinessReason
from recall_guard.core.request_control import RequestError


class _ExecutionCancellation(CancellationToken):
    def __init__(self, caller):
        super().__init__()
        self.caller = caller

    @property
    def cancelled(self):
        return super().cancelled or self.caller is not None and self.caller.cancelled

    def wait(self, timeout_s):
        deadline = monotonic() + timeout_s
        while not self.cancelled:
            remaining = deadline - monotonic()
            if remaining <= 0:
                break
            super().wait(min(remaining, 0.05))
        return self.cancelled


class WorkflowExecution:
    """One run's shared cancellation latch and actual operation observations.

    A stopping role prevents later physical admission in every role. In-flight
    synchronous exchanges remain cooperative and are bounded by their transport.
    Preparation evidence never enters the execution's preflight cache.
    """

    def __init__(self, cancellation=None, on_observation=None):
        self.cancellation = _ExecutionCancellation(cancellation)
        self.on_observation = on_observation
        self.observations = []
        self.lock = Lock()
        self.root_failure = None
        self.root_observation = None

    def _observation(self, observation):
        root = self.root_observation
        caller = self.cancellation.caller
        if (observation.reason == "caller_cancelled" and root is not None
                and not (caller is not None and caller.cancelled)):
            return replace(observation, state=root.state, reason=root.reason)
        return observation

    def observe(self, observation):
        observation = self._observation(observation)
        with self.lock:
            self.observations.append(observation)
        if self.on_observation is not None:
            self.on_observation(observation)

    def retain(self, trace):
        """Retain terminal callback-failure evidence without notifying again."""
        with self.lock:
            for observation in trace:
                observation = self._observation(observation)
                if observation not in self.observations:
                    self.observations.append(observation)

    def sanitize_failure(self, client, error, *, response=None):
        """Use the owning core sanitizer before a response becomes failed evidence."""
        failure = client.sanitize_execution_failure(error, response=response)
        trace = getattr(failure, "readiness_trace", ()) or (
            () if failure.response_evidence is None else failure.response_evidence.readiness_trace or ())
        failure.readiness_trace = tuple(self._observation(item) for item in trace)
        caller = self.cancellation.caller
        if (failure.code == "caller_cancelled" and self.root_failure is not None
                and not (caller is not None and caller.cancelled)):
            failure.code = self.root_failure.code
            failure.args = ("Execution stopped after a required workflow failure.",)
        return failure

    def latch_failure(self, failure):
        """Publish the original failure before waking any waiting admission."""
        with self.lock:
            if self.root_failure is None:
                trace = getattr(failure, "readiness_trace", ())
                if trace:
                    terminal = trace[-1]
                    if terminal.state == "ready":
                        try:
                            reason = ReadinessReason(failure.code)
                        except ValueError:
                            reason = ReadinessReason.MALFORMED_RESPONSE
                        terminal = replace(terminal, state="rejected", reason=reason,
                                           timestamp=datetime.now(UTC).isoformat())
                        failure.readiness_trace = (*trace, terminal)
                    self.root_observation = terminal
                    if terminal not in self.observations:
                        self.observations.append(terminal)
                self.root_failure = failure
            return self.root_failure

    def generate(self, client, prompt, *, stage, parameters):
        try:
            return client.generate_for_execution(prompt, stage=stage, cancellation=self.cancellation,
                                                 on_observation=self.observe, **parameters)
        except Exception as error:
            failure = self.sanitize_failure(client, error)
            self.retain(failure.readiness_trace)
            self.latch_failure(failure)
            self.cancellation.cancel()
        raise failure

    def probe(self, client, prompt, *, stage):
        result = client.probe_with_evidence(prompt, stage=stage, cancellation=self.cancellation,
                                            on_observation=self.observe)
        self.retain(result.observations)
        if not result.ready:
            terminal = result.observations[-1]
            error = RequestError("Selected workflow preflight did not complete.",
                                 code=(terminal.reason or terminal.state).value,
                                 response_evidence=result.response)
            error.readiness_trace = result.observations
            error = self.sanitize_failure(client, error)
            self.latch_failure(error)
            self.cancellation.cancel()
            raise error
        return result

    def proxy(self, client):
        return _ExecutionClient(client, self)


class _ExecutionClient:
    """Runtime adapter used only by the existing rewrite/fidelity pipeline."""

    def __init__(self, client, execution):
        self.client, self.execution = client, execution

    @property
    def config(self):
        return self.client.config

    @property
    def request_accounting(self):
        return getattr(self.client, "request_accounting", None)

    def generate(self, prompt, **parameters):
        return self.generate_for_stage(prompt, stage="target", **parameters)

    def generate_for_stage(self, prompt, *, stage, **parameters):
        return self.execution.generate(self.client, prompt, stage=stage, parameters=parameters)

    def probe(self):
        return self.client.probe()
