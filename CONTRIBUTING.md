# Contributing

This repository holds the inference package and the benchmarks. Training
code and the training data are not part of it. Start with
[README.md](README.md) for usage, [bench/README.md](bench/README.md) for the
benchmarks and [AGENTS.md](AGENTS.md) for the conventions.

## Development setup

From a checkout, with Python 3.10 or later:

```bash
make install            # requires uv; creates .venv on Python 3.11 (or reuses one) with dev and eval
make test               # CPU tests, including a summary of skipped tests
make lint
make format-check
```

`make` lists the available commands. They use `.venv/bin/python` by default;
set `PYTHON` to use another interpreter. The equivalent manual setup is:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev,eval]"
python -m pytest -q -rs
```

The tests use a tiny model and need no weights or audio.

### What a fresh checkout shows

After `make install`, `make test` passes with nothing failing and these
skips, which `pytest -rs` names: the golden parity test, without the released
checkpoint under `models/sava.safetensors` (or `SAVA_GOLDEN_MODEL`), and the
CUDA graph test of `Streams`, without a CUDA GPU. The checkpoint's expected
checksum is in `src/inworld/sava/registry.py`. Include
the skip count when you report a run.

CI runs the same development checks on pushes and pull requests, on Ubuntu:
the tests on Python 3.10 and 3.12, and the style checks on Python 3.10 with
the pinned tools. Lint on Python 3.11 or earlier: from 3.12, pylint reads
quotes inside f-strings differently and reports `inconsistent-quotes`
warnings that CI does not.

## Reporting a problem

Include the command or a minimal Python example, what you expected, what
happened and the traceback. For inference issues paste the output of
`sava-doctor --json`, and add the package version or git revision, the
checkpoint name or checksum, the Python and PyTorch versions, and the input's
sample rate, channels and duration; a synthetic or shareable clip helps. For benchmark issues add the set, the script's flags
and the revision of the data you downloaded.

## Preparing a change

Keep changes focused and describe the behavior they fix. Follow
[AGENTS.md](AGENTS.md): use isort, yapf and Pylint, keep the license headers, put
tests beside the mirrored module. A change to the model's numerics is a
change to the released model and comes with a new checkpoint, checksum and
golden fixture. Weights, third-party code and evaluation audio do not belong
in patches.
Run the relevant tests from the repository root; run the full suite before
proposing a release.

`make format` applies the pinned isort and yapf versions to Python in `src/`,
`bench/scripts/`, `docs/` and `tests/`.
`make format-check` checks without writing. `make lint` also checks tracked
files for machine-specific paths. Keep touched tests in arrange / act / assert
sections separated by blank lines, and exercise public behavior rather than
private helpers.

Tests use `test_<module>.py` names under mirrored source directories. CI
builds the source archive and wheel, checks their contents, and runs the
installed CLI outside the checkout. Public benchmark results and manifests
remain available for reproduction.

Source archives contain the inference package, model documentation, integration
guide and README figures; wheels contain the package and license notices. Use a
repository checkout for the benchmark, tests and development commands.

Rebuild the README figures with `python docs/build_figures.py`; it needs no
plotting library. The figures are hand-written SVG with live text, in one font
stack and at one width, and the architecture carries a full accessible
description. Both light and dark variants are generated from the same layout.
