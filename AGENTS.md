# SAVA, guide for coding agents

This file serves two tasks:

- **Setting up SAVA for someone to use** (install it, check the machine, run
  it on a first recording): follow [Setting up SAVA for a user](#setting-up-sava-for-a-user),
  the last section, and skip the rest. It installs with `pip`; `make install` is
  the contributor setup, with development and evaluation extras a user does not
  need.
- **Changing this repository:** read every section before that one first.
  [CONTRIBUTING.md](CONTRIBUTING.md) covers setup and validation;
  [README.md](README.md) covers usage.

`CLAUDE.md` only imports this file.

## What this repository is

The public release of SAVA (semantic acoustic voice activity): an
installable package (`inworld-sava`, code under `src/inworld/sava/`), its
public benchmarks with their published results (`bench/`) and the tests.
Project code is published under Apache-2.0; write accordingly and keep links
within the published tree.

| path | what |
|---|---|
| `src/inworld/sava/__init__.py` | the public API (`inworld` itself is a namespace package: no `src/inworld/__init__.py`) |
| `src/inworld/sava/*.py` | `config` (the topology), `mel` (the causal log-mel front end), `ssd` (the Mamba-2 scan), `modules` (blocks), `model` (the network and its streaming step), `checkpoint` (the release file), `registry` (released weights by name), `streaming` (sessions and whole-recording curves), `streams` (many streams stepped together), `segmenter` (the reference turn policy), `audio`, `cli`, `doctor` (the `sava-doctor` readiness check, not part of the public API) |
| `tests/` | mirrors the source tree, `test_<module>.py` per module, `conftest.py` for the tiny model and the paths; `tests/fixtures/` holds the golden-parity clip and its expected logits |
| `bench/scripts/` | one script per benchmark, `build_vadsuite.py` builds the sets `vadsuite.py` scores from their downloads, `common.py` for what they share, `tables.py` renders the published tables from `bench/results/*.json` |
| `bench/data/` | the manifests the benchmarks need beyond the public downloads (bench/DATA.md) |
| `bench/results/` | the published score files; every number quoted in the READMEs comes from them through `tables.py` |
| `docs/` | generated light/dark figures and the integration guide |
| `MODEL_CARD.md` | the model card, shown as the Hugging Face page: purpose, quick start, benchmarks, architecture and training data; its TL;DR and quick start repeat the README's, so change them together |
| `models/`, `data/`, `staging/` | gitignored, local only: weights, benchmark audio and built manifests, scratch |

## Rules

1. **Python follows the Google style guide**, formatted with yapf
   (`.style.yapf`, 80 columns), with Google's isort profile and Pylint
   configuration (`.pylintrc`). Run `make format`, `make format-check` and
   `make lint`. Imports name modules. Docstrings
   open with a one-line summary and describe the caller's contract; public
   signatures carry type hints. Loading-only dependencies may be imported
   inside their loader, and optional extras stay behind helpful ImportErrors.
   Modeling code uses names and docstrings rather than inline comments;
   elsewhere, comments explain a why, contract or caveat. No public file carries
   internal names, paths, hostnames or checkpoint codenames. Python files
   carry the Apache-2.0 header. Module and script names use lowercase words
   joined by underscores; test modules are named `test_<module>.py`.
   Existing benchmark filenames such as `eotbench.py`, `smartturn.py` and
   `vadsuite.py` remain stable for published commands and adapter imports.
   `make format` applies the pinned isort and yapf versions; `make format-check`
   checks without writing. `make lint` runs pylint and checks tracked files for
   machine-specific paths, with the same configuration as CI.
2. **Tests** are pytest functions under `tests/`, one module per source
   module, each test arrange / act / assert with a blank line between the
   parts. Nothing private is reached into or mocked; dependencies are injected
   or patched where they are looked up, with `mock.patch.object(...,
   autospec=True)` and `mock.create_autospec(..., instance=True,
   spec_set=True)`. The golden test needs the released weights
   (`models/sava.safetensors` or `SAVA_GOLDEN_MODEL`) and skips without
   them; say so when reporting a run. `make test` runs the suite with the skip
   summary. Run focused tests for a change and the full suite before a release.
   [CONTRIBUTING.md](CONTRIBUTING.md#what-a-fresh-checkout-shows) lists what a
   fresh checkout shows: the skips to expect, and the platform and Python
   versions CI uses. Check a skip, a warning or a failure against it before
   investigating. If a check already fails on a clean checkout of `main`,
   before your change, report it in one line and carry on with the task you
   were given; do not investigate or fix it unasked.
3. **The model's two outputs are contracts.** The speech output emitted at
   frame t judges frame t − `lookahead_frames`; the turn output is on time.
   A full forward equals the stream of its frames; `test_model.py` and the
   golden test pin both. A change to the numerics is a change to the
   released model and needs a new checkpoint, a new checksum in `registry.py`
   and a new golden fixture computed by the reference implementation, never
   by hand.
4. **Benchmarks are black boxes.** A script scores what the model returns at
   the step it returns it, through `streaming.run` or a `Session`, and
   Silero or smart-turn through their own public paths. Numbers in the
   READMEs are generated by `bench/scripts/tables.py`, never typed. A result
   file in `bench/results/` is replaced only by a full rerun of its script on
   the set it names.
5. **Nothing third-party is vendored** and no evaluation audio is committed.
   Baselines are installed from their publishers and data is downloaded,
   each under its own license, as bench/DATA.md describes, and only the
   manifests under `bench/data/` are tracked.
6. **Publishing is gated.** No push, no Hugging Face or PyPI upload, no
   public URL without an explicit go from the maintainer.

## Environment

Python 3.10 or later, PyTorch 2.1 or later, CPU is enough for everything but
the benchmarks' wall-clock time. `make install` uses uv to create or reuse
`.venv` on Python 3.11 with the dev and eval extras; `make` lists the commands, and
`PYTHON` selects another interpreter. `PYTHONPATH=src` works without
installing. The microphone demo needs the `mic` extra.

Benchmark JSONs, tables and data manifests are available for reproduction.
Source archives contain the inference package, model documentation, integration
guide and README figures; wheels contain the package and license notices.
Use a checkout for development and benchmark work.

## Setting up SAVA for a user

A person asked you to set up Inworld SAVA: a model that listens to a live audio
stream and says, every 40 ms, whether someone is speaking and whether they have
finished their turn. You install the `inworld-sava` Python package, check the
machine with `sava-doctor`, and run it on a first recording. The rest of this
file is about changing the repository; you do not need it for this.

Complete every step yourself with your own tools. A step marked
**Ask the person** needs them: ask in one short line, wait until they answer,
then go on. Tell them what to do and what they will get; do not explain how the
model works inside.

What the model does and how well is in the [model card](MODEL_CARD.md). To
build it into another project or service,
[the integration guide](docs/USAGE.md) covers its API.

### 0. Before you start

- **Already installed?** If `sava-doctor` runs, skip to [step 3](#3-verify).
- **Use a virtual environment**, never the system Python. Use Python 3.10 or
  later.
  - With `uv`: `uv venv`, then prefix every `pip` below with `uv`
    (`uv pip install ...`), because a `uv` environment has no `pip` of its own.
  - Without it: `python -m venv .venv`, then activate it.

### 1. Install

Try these in order. When one fails, go to the next.

1. **Inside a clone** (a `pyproject.toml` in this folder names
   `inworld-sava`). Do not use `make install` here: that is the contributor
   setup.

   ```
   pip install -e .
   ```

2. **From PyPI:**

   ```
   pip install inworld-sava
   ```

3. **From GitHub**, pinned to the newest tag (`git ls-remote --tags
   https://github.com/inworld-ai/sava.git` lists them; use `main` if there is
   none):

   ```
   pip install "inworld-sava @ git+https://github.com/inworld-ai/sava.git@<tag>"
   ```

Add the `[mic]` extra (for example `pip install -e ".[mic]"`) only when the
person wants to run SAVA on their microphone.

If every route fails, stop and tell the person: "I couldn't install Inworld
SAVA from PyPI or GitHub. Check the network or proxy settings and ask me
again."

### 2. Weights

Nothing to do. The model downloads from Hugging Face (`inworld/sava`, about
51 MB) the first time it is used, and its checksum is verified.

### 3. Verify

```
sava-doctor --full
```

It prints one line per check, each starting with `OK`, `INFO`, `WARN` or
`FAIL`, and ends with one summary line. The first run downloads the weights.

- **`All set: …`** means it worked. Go to step 4.
- **Anything else:** each `FAIL` line names one next step. Do the steps you
  can, give the person the rest, and run the check again.

`sava-doctor` without `--full` uses no network and loads no model; use it to
report the machine's state, and `--json` for a machine-readable copy.

### 4. First result

Use the example recording: 23 s of speech on a street. In a clone it is
`tests/fixtures/speech.wav`; otherwise download it:

```
curl -LO https://raw.githubusercontent.com/inworld-ai/sava/main/tests/fixtures/speech.wav
sava run speech.wav --turns_dir speech_turns
```

This prints one line per turn (where it starts and ends, how much speech it
held and how it closed) and writes each kept turn to `speech_turns/` as its
own wav. Tell the person what it found and where the files are.

**Ask the person:** "Want to try your own recording, or your microphone? Give
me a file path, or say microphone."

For a recording, run the same command with their file and a folder beside it.
For the microphone, install the `[mic]` extra if it is missing and run
`sava mic --turns_dir mic_turns`: it listens for a minute (`--seconds`
changes that) and prints each turn as it closes. Options are in the
[README](README.md#quick-start).

### 5. Finish

Tell the person, in one or two lines, for example:

> SAVA is ready on the CPU. It found 2 turns in speech.wav and saved them in
> speech_turns/. Give me a recording to segment, or say "microphone" to try it
> live.

Name the device the doctor reported.

### Rules

- Keep the person's audio on this machine: do not upload it anywhere.
- Never ask the person to paste a Hugging Face token or any other secret into
  the chat.

### If something goes wrong

| What you see | What to do |
|---|---|
| Every install route in step 1 fails | Check the network or proxy, then try again. Give the person the message in step 1. |
| `FAIL … inworld/sava is not reachable` | Hugging Face refused the download. Check that `huggingface.co` opens from this machine, or pass a local model file with `--model`. |
| `FAIL … cannot reach huggingface.co` | No network, a proxy, or `HF_HUB_OFFLINE` is set. Fix the connection, or pass a local model file. |
| `FAIL … checksum mismatch` | Delete the file the line names and run `sava-doctor --full` again. |
| `FAIL device: … is not available` or `does not exist` | This machine does not have that device. Run with `--device cpu`. |
| `FAIL … failed on cuda` | Run with `--device cpu`. Report the error line to the person. |
| `sava mic` says it needs the `mic` extra | Install it as in step 1, for example `pip install "inworld-sava[mic]"`. |
| `sava mic` hears nothing | The terminal needs microphone permission: on macOS, System Settings, Privacy & Security, Microphone. |

### Remove SAVA

- `pip uninstall inworld-sava` (with `uv`: `uv pip uninstall`).
- Delete the weight file at the path `sava-doctor` prints on its `cached at`
  line.
- Delete the virtual environment you created.
