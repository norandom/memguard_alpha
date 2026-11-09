# Run the checkout in one locked environment

Docker runs the existing Python library and examples using `pyproject.toml` and
`uv.lock`. Compose and Dagger build the same root Dockerfile. The application
still consists of API clients and calculations. On Windows, Docker is the only
supported route: run Recall Guard, tests, probes, measurements and notebooks in
Linux containers, not with native Windows Python. On Linux and macOS, Docker is
optional for users installing the wheel or using an existing Python/Jupyter
environment.

## Start with an offline result

Use a current development checkout, an accessible Docker daemon and the Compose
plugin. Docker Desktop with Linux containers or Docker in the existing Debian
WSL environment can supply the daemon. Run commands from the repository root
in a shell where `docker info` and `docker compose version` succeed.

Create the output directory yourself so your host account owns it. In PowerShell:

```powershell
New-Item -ItemType Directory -Force runs/container
```

In a Linux shell:

```sh
mkdir -p runs/container
```

Then the same build/run commands apply to either shell:

```text
docker compose build dev
docker compose run --rm dev
```

The default callback example requires no keys or provider connection. It reports
an accepted rewritten ensemble, a fidelity-rejected rewrite with no target draws,
unavailable MCS and `actual_http_requests: 0`. This demonstrates mechanics, not
mitigation effectiveness. Configure model/settings and permitted features through
the existing [completion workflow](completion-api.md).

## Run focused checks, then the integration checks

Override the service command without changing its environment:

```text
docker compose run --rm dev uv run --no-sync pytest tests/test_packaging.py tests/test_architecture.py -q
docker compose run --rm dev uv run --no-sync python examples/completion_workflow.py
docker compose run --rm dev uv run --no-sync pytest -q
docker compose run --rm dev uv run --no-sync ruff check .
docker compose run --rm dev uv run --no-sync python scripts/check_architecture.py
docker compose run --rm dev uv build --out-dir /outputs/dist
docker compose run --rm dev uv run --no-sync mkdocs build --strict --site-dir /outputs/site
```

Artifacts appear under `runs/container/dist` and `runs/container/site` on the
host. Run focused checks while changing behavior and one full regression at the
completed integration boundary. Existing release compatibility jobs retain their
Python matrix; there is no need to replay that matrix for each helper.

Dependencies install with `uv sync --frozen` during the image build. Runtime
commands use `--no-sync`; edit the lock deliberately and rebuild when dependencies
change. Source is baked into the image, so rebuild after source changes too.
The cached dependency layer is reused when the lock is unchanged.

## Use Jupyter without replacing your existing kernel

The optional profile uses the development image and keeps token authentication:

```text
docker compose --profile notebook up notebook
```

Open the localhost URL printed by Jupyter and its token. Host publication is
`127.0.0.1:8888`; the server listens on all interfaces only inside the container.
The profile mounts `notebooks/` for persistent editing and `runs/container` at
`/outputs` for generated artifacts. It does not mount the whole checkout or your
host virtual environment. Existing notebooks use the same configured model and
feature semantics as ordinary Python.

Stop with Ctrl+C, then clean up only this project's service/network:

```text
docker compose --profile notebook down
```

For background operation use `up --detach notebook` with the same profile.
The long flag also avoids PowerShell functions interpreting `-d` as Debug.
No provider requests or credential injection occur at notebook startup.

## Install only the runtime when development tools are unnecessary

The runtime target contains the installed wheel and locked runtime dependencies:

```text
docker build --target runtime -t recall-guard:runtime .
docker run --rm --network none --workdir /tmp recall-guard:runtime
```

The default command prints the installed package path under `/opt/venv`. Import
works outside the source tree. The runtime excludes the checkout, test tools,
Jupyter, plotting extras and Dagger SDK. It performs no dependency synchronization
on startup. It is a library environment, not an HTTP server or a preconfigured
live-study runner; the developer scripts are in the `dev` target.

## Supply provider credentials only for an explicit run

Builds, tests and the default notebook service receive no provider keys. Docker
excludes `.env` files from the build context, and `compose.yaml` passes no
variables into services. Keep credentials in the single repository-root `.env`.

For an explicitly configured live study, name each variable the command needs
with `-e NAME` and no value. Compose reads the value from your shell environment
or, if unset there, from the root `.env`. Only the named variables enter the
container, and no value appears on the command line. This example only opens CLI
help:

```text
docker compose run --rm -e ANTHROPIC_API_KEY -e NVIDIA_API_KEY dev uv run --no-sync python scripts/run_intervention.py --help
```

This was checked with Compose v5.6.1 in WSL Debian using dummy values. Avoid
`docker run --env-file .env`, which passes every provider credential in the file
to the container. `docker compose --env-file` only feeds Compose interpolation and
does not reach the container. Never print credential values to check them.

Use the CLI's explicit `--run`, protocol, item, checkpoint and new output paths
only after freezing the study and allowance. Mount study inputs read-only and
write results to `/outputs`; input files under ignored host `runs/` are not baked
into the image. Follow [bounded calls and paired identity](completion-api.md#bound-calls-and-retain-paired-identity).
Docker does not change provider eligibility, response times, model settings,
request limits or calibration, and startup never runs a probe for you.

The [Jane Street article](https://blog.janestreet.com/mitigating-memorization-in-llms/)
explores both logit steering and rewritten inputs. Recall Guard's divergence
utility needs token-selection access; completion-only rewriting is a separate
experimental path. Container checks do not measure either method's effect. See
the [dated pilot](results/2026-10-31-completion-api-pilot.md) for existing evidence
and its limitations. Missing token evidence still means unavailable MCS.

## Reuse the same recipe in Dagger

In the shell with the existing Dagger CLI, the public commands remain:

```text
dagger --mod ci call test --source=.
dagger --mod ci call lint --source=.
dagger --mod ci call build --source=. export --path=dist
dagger --mod ci call docs --source=. export --path=site
```

`--paths="tests/core tests/test_packaging.py"` narrows `test` to space-separated
pytest targets; without it the whole suite runs.

The Python selector remains `--python=3.12` or `--python=3.14`; the Dockerfile owns
the base-image version and default. Dagger consumes its `dev` target rather than
maintaining another dependency-install recipe. Build and docs functions return
the same wheel/sdist/site artifacts from writable `/outputs` subdirectories.

Dagger source arguments exclude secrets and bulky local state before upload.
The complete build-context exclusions are then read from `.dockerignore`. Keep
the pre-upload secret/large-directory filters aligned when adding such paths;
the current build filter deliberately rejects negated ignore rules. Docker and
Dagger ignore syntax are not assumed interchangeable for arbitrary future rules.

## Resolve the concrete failure

| Observation | Next action |
| --- | --- |
| Docker daemon or WSL connection unavailable | Restore access to the configured daemon/shell. Keep the error receipt; do not treat it as an application or Python defect. |
| Output bind directory missing or not writable | Create `runs/container` as the host user and check its permissions; services never run privileged to work around it. |
| Code/dependency changes are absent | Rebuild `dev`; host source and environments are not overlaid by default. |
| Notebook port already in use | Stop the conflicting service or deliberately change only the host port in Compose. Keep loopback binding and authentication. |
| A configured API is slow or rejects access | Inspect its actual response and selected policy. A successful container build is not evidence of model availability. |
