"""The front in one run: list the catalogue, select an entry, run a point-in-time prompt, read the result.

PowerShell: uv run --no-sync python examples/pit_front.py --fixture

``--fixture`` needs no key and sends nothing: the model's reply is an invented
fixed reply, so the run shows the wiring and how to read a result, not a
model's quality. Without ``--fixture`` the front reaches the model through the
entry's provider profile, so that route must be up and its environment set; if
it is not, the script prints the step that failed with what to change.
``--entry ID`` selects another catalogue entry, ``--mode logprobs|logits``
names the mode instead of letting the front choose, and ``--json`` prints the
result as one JSON object for programs. The exit code is 0 for an answer and 1
for a failure.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rich.console import Console
from rich.table import Table

from recall_guard import WORKED_EXAMPLE_REPLY, FixedReplies, Mode, Status, ask, load_catalogue, worked_example

DEFAULT_ENTRY = "llamacpp-gemma3-4b"  # The entry that offers both modes.


def words(status):
    """A status as a person reads it: 'not tested' for Status.NOT_TESTED."""
    return status.value.replace("_", " ")


def catalogue_table(catalogue):
    """One row per entry: the model, its route, and each mode's status or 'not offered'."""
    table = Table(box=None, pad_edge=False, header_style="bold")
    for heading in ("entry", "model", "route", *(mode.value for mode in Mode)):
        table.add_column(heading, no_wrap=heading not in ("model", "route"), overflow="fold")
    for entry in catalogue.entries:
        statuses = [words(catalogue.qualification(entry.entry_id, mode).status) if entry.modes[mode].available
                    else "not offered" for mode in Mode]
        table.add_row(entry.entry_id, entry.model, entry.route, *statuses)
    return table


def as_json(result, *, fixed_reply):
    """The result as plain data: None where the front returned nothing."""
    failure = result.failure
    return {
        "entry": result.entry_id,
        "mode": None if result.mode is None else result.mode.value,
        "mode_choice": result.mode_choice,
        "status": None if result.status is None else result.status.value,
        "reason": result.reason,
        "answer": None if result.answer is None else dict(result.answer),
        "p_memorized": result.p_memorized,
        "corrected": None if result.corrected is None else dict(result.corrected),
        "limits": list(result.limits),
        "failure": None if failure is None else {"step": failure.step, "detail": failure.detail,
                                                 "remedy": failure.remedy},
        "fixed_reply": fixed_reply,
    }


def loadings(decision):
    """One line per component of a decision, signed, or 'none' where the component has no value."""
    return [f"  {name}: {'none' if value is None else format(value, '+.2f')}" for name, value in decision.items()]


def result_lines(result):
    """The result as a person reads it, top to bottom."""
    lines = [f"entry: {result.entry_id}"]
    if result.mode is not None:
        lines.append(f"mode: {result.mode.value} ({result.mode_choice})")
    if result.status is not None:
        lines += [f"status: {words(result.status)}", f"reason: {result.reason}"]
    if result.failure is not None:
        return [*lines, "no answer", f"failed at step: {result.failure.step}", f"detail: {result.failure.detail}",
                f"what to change: {result.failure.remedy}"]
    lines += ["answer (loading on each axis, from -1 to +1):", *loadings(result.answer)]
    lines.append("recall estimate: " + ("none" if result.p_memorized is None else f"{result.p_memorized:.3f}"))
    if result.corrected is not None:
        lines += ["corrected answer:", *loadings(result.corrected)]
    elif result.status is Status.QUALIFIED:
        lines.append("corrected answer: none, because the run did not meet what a correction needs; see the "
                     "reason and the limits")
    else:
        lines.append(f"corrected answer: none, because the status is {words(result.status)} and only a qualified "
                     "entry and mode is corrected")
    if result.limits:
        lines += ["limits:", *(f"  - {limit}" for limit in result.limits)]
    else:
        lines.append("limits: none")
    return lines


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fixture", action="store_true",
                        help="answer from an invented fixed reply; needs no key and sends nothing")
    parser.add_argument("--entry", default=DEFAULT_ENTRY, help=f"catalogue entry id (default: {DEFAULT_ENTRY})")
    parser.add_argument("--mode", choices=[mode.value for mode in Mode],
                        help="evidence mode; left out, the front chooses and says why")
    parser.add_argument("--json", action="store_true", help="print the result as one JSON object and nothing else")
    args = parser.parse_args(argv)

    catalogue = load_catalogue()
    prompt, definition = worked_example()
    # The whole call: a prompt, its decision definition and a catalogue entry. With client=None the front
    # builds the client from the entry's provider profile; the fixed client replays one reply for every draw.
    client = FixedReplies([WORKED_EXAMPLE_REPLY], cycle=True) if args.fixture else None
    mode = None if args.mode is None else Mode(args.mode)

    if args.json:
        result = ask(prompt, definition, entry=args.entry, mode=mode, client=client, catalogue=catalogue)
        print(json.dumps(as_json(result, fixed_reply=args.fixture), indent=2))
        return 0 if result.failure is None else 1

    # Piped output is not wrapped, so each line stays whole for whatever reads it.
    console = Console(highlight=False, width=None if sys.stdout.isatty() else 160)

    def say(*lines):
        for line in lines:
            console.print(line, markup=False, soft_wrap=True)

    def heading(text):
        say("")
        console.print(text, style="bold", markup=False)

    heading("1. The catalogue")
    qualified = len(catalogue.filter(status=Status.QUALIFIED))
    say(f"{len(catalogue.entries)} entries, each a model on a serving route; {qualified} with a qualified mode.", "")
    console.print(catalogue_table(catalogue))
    say("",
        "qualified: the recorded calibration for the entry and mode passed every check; its answers can be "
        "corrected.",
        "not qualified: a recorded check failed. not tested: a check has no recorded value.",
        "not offered: the route cannot serve that mode. An entry answers in any mode it offers, whatever the "
        "status.")

    heading("2. The selected entry")
    selected = next((entry for entry in catalogue.entries if entry.entry_id == args.entry), None)
    if selected is None:
        say(f"{args.entry} is not in the catalogue; the front says so below.")
    else:
        say(f"{selected.entry_id}: {selected.model} on {selected.route}")
        for item in Mode:
            record = selected.modes[item]
            qualification = catalogue.qualification(selected.entry_id, item)
            say(f"  {item.value}: " + (f"{words(qualification.status)} ({qualification.reason})" if record.available
                                       else f"not offered ({record.unavailable_reason})"))
    say("mode: " + ("the front chooses" if mode is None else f"{mode.value}, named with --mode"))

    heading("3. The prompt")
    say("The worked example: a fenced point-in-time prompt. It shows the macro state as z-scores and the assets "
        "by letter and category, with no date, no ticker and no raw level, and asks for the regime, not for a "
        "forecast.", "", prompt)

    heading("4. The run")
    if args.fixture:
        say("Fixed reply: nothing is sent. Every draw gets the same invented fixed reply, written by hand; it is "
            "not a model output. This run shows the wiring and how to read a result, not a model's quality.",
            f"The reply: {WORKED_EXAMPLE_REPLY}")
    else:
        say("The front reaches the model through the entry's provider profile and draws 32 replies.")
    result = ask(prompt, definition, entry=args.entry, mode=mode, client=client, catalogue=catalogue)

    heading("5. The result")
    say(*result_lines(result))
    if result.failure is None:
        say("",
            "Each limit is something this answer lacks. Read the answer as the model's reply to a fenced "
            "prompt; without a recall estimate nothing says how much of it is recall.",
            "No calibrator is fitted in this demo. A calibrator, fitted on a calibration set for the entry's "
            "model and mode and passed to ask, adds the recall estimate, and for a qualified entry and mode "
            "the corrected answer. See docs/pit-architecture.md.")
    return 0 if result.failure is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
