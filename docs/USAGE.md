# Using SAVA

## Use

```python
from inworld import sava

model = sava.load("inworld/sava")   # downloads the weights on first use; a local path also works
curves = sava.run(model, sava.load_audio("recording.wav"))
curves.speech   # P(speech) per 40 ms frame, lined up with the frames
curves.turn     # P(the turn has ended) per frame
```

The same model streams. A session takes 16 kHz mono samples in pieces of any
size and returns the frames it could step, each with the two probabilities at
the step they come out:

```python
session = sava.Session(model)
for piece in pieces_of_16k_audio:                      # NumPy arrays or tensors
    frames = session.push(piece)
    frames.speech   # value i judges frame frames.first + i - model.lookahead_frames
    frames.turn     # P(the turn has ended by frame frames.first + i)
frames = session.flush()                               # the last frames' verdicts
```

A stream's first `lookahead_frames` speech values are the no-speech floor,
since they would judge frames before the stream; `flush()` steps that many
frames of silence so the last real frames get theirs. `sava.run` does both
for a whole recording.

A server with many live streams steps them together with `sava.Streams`,
one batched step per frame instead of one step per stream; streams open, push
and flush at any time, and each stream gets what its own session would:

```python
streams = sava.Streams(model, capacity=64)             # up to 64 open streams
caller, other = streams.open(), streams.open()
frames = streams.push({caller: piece, other: piece})   # Frames by stream id
frames = streams.flush([caller])                        # ends it, frees its slot
```

On a GPU, `sava.Streams(model, capacity=64, cuda_graphs=True)` replays each
step of every open stream as a CUDA graph, one launch instead of several
hundred; a step that leaves out an open stream runs as usual.

`sava.segment(curves, sava.Policy(), model.lookahead_frames)` and
`sava.TurnSegmenter` turn the probabilities into turns with the reference
policy: a turn opens on the first frame judged speech and starts 800 ms
earlier; it closes when the end-of-turn probability crosses 0.9 after the
speaker goes quiet, or after 1.6 s of silence whatever the model says; a turn
shorter than 400 ms of speech (200 ms of confident speech) is reported as
dropped. The benchmarks score the raw probabilities, not this policy.

## Limits to know before integrating

- The speech verdict lags its frame by 120 ms. A policy that counts silence
  from it sees silence 120 ms late, which is why the reference policy checks
  the end-of-turn probability as soon as the speech verdict goes quiet rather
  than after a fixed wait.
- The turn head is trained for conversational turns and reads sentence
  completeness; it is not a dialogue policy and does not model interruptions
  or backchannels as such.
- The model hears quiet and distant speech by design. Very quiet, speech-like
  noise (a far-away television, a ventilation drone at −80 dBFS) can still
  trigger it where an on-time detector would stay silent; see the
  benchmarks' ESC-50 row.
- Audio is 16 kHz mono. Other rates and channel counts are converted on the
  way in; nothing restores a band-limited recording's missing frequencies.

## How it works

See the model card for the [architecture](../MODEL_CARD.md#architecture).

## Requirements and tested configurations

| | |
|---|---|
| Python | ≥ 3.10 |
| PyTorch | ≥ 2.1 (developed on 2.14) |
| Weights | 51 MB (bf16 safetensors, loaded in fp32 on CPU) |

| Hardware | Status | Speed |
|---|---|---|
| Apple M3 Max, CPU, fp32 | primary; the test suite and the golden parity run here | one stream 3.8 ms per 40 ms frame; a 12 s recording in 0.04 s |
| NVIDIA A100, fp32 | the benchmarks' published results | whole recordings, hundreds of times real time |
