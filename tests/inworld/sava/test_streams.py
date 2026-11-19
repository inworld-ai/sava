# Copyright 2026 Inworld AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Streams stepped together against each stream's own session."""

import numpy as np
import pytest
import torch

import conftest
from inworld.sava import config
from inworld.sava import streaming
from inworld.sava import streams

FRAME = config.FRAME_SAMPLES


def _session_frames(model, audio, pieces):
    session = streaming.Session(model)
    out = [session.push(audio[a:b]) for a, b in pieces]
    return out + [session.flush()]


def _assert_frames_equal(got, want):
    assert got.first == want.first
    np.testing.assert_allclose(got.speech, want.speech, atol=1e-6)
    np.testing.assert_allclose(got.turn, want.turn, atol=1e-6)
    np.testing.assert_allclose(got.turns, want.turns, atol=1e-6)


def _pieces(length, size):
    return [(a, min(a + size, length)) for a in range(0, length, size)]


def test_streams_at_different_frames_each_equal_their_own_session(model):
    # (tick it opens at, frames, piece size): openings past every window
    plan = [(0, 40, FRAME), (3, 9, 1000), (11, 30, 3 * FRAME + 17),
            (12, 5, 7 * FRAME), (20, 18, FRAME // 2)]
    audio = [
        conftest.pcm(batch=1, frames=n, seed=i)[0].numpy()
        for i, (_, n, _) in enumerate(plan)
    ]
    pieces = [_pieces(a.size, size) for a, (_, _, size) in zip(audio, plan)]
    engine = streams.Streams(model, capacity=4)
    ids, got = {}, {i: [] for i in range(len(plan))}

    for tick in range(80):
        for i, (start, _, _) in enumerate(plan):
            if tick == start:
                ids[i] = engine.open()
        due = {
            i: pieces[i][tick - plan[i][0]]
            for i in ids
            if tick - plan[i][0] < len(pieces[i])
        }
        pushed = engine.push({
            ids[i]: audio[i][a:b] for i, (a, b) in due.items()
        })
        for i in due:
            got[i].append(pushed[ids[i]])
        done = [i for i in ids if tick - plan[i][0] == len(pieces[i])]
        flushed = engine.flush([ids[i] for i in done])
        for i in done:
            got[i].append(flushed[ids.pop(i)])

    for i, a in enumerate(audio):
        want = _session_frames(model, a, pieces[i])
        assert len(got[i]) == len(want)
        for g, w in zip(got[i], want):
            _assert_frames_equal(g, w)


def test_a_slot_freed_by_a_flush_starts_its_next_stream_fresh(model):
    first = conftest.pcm(batch=1, frames=20, seed=1)[0].numpy()
    second = conftest.pcm(batch=1, frames=12, seed=2)[0].numpy()
    engine = streams.Streams(model, capacity=1)
    stream = engine.open()
    engine.push({stream: first})
    engine.flush([stream])
    stream = engine.open()

    got = [
        engine.push({stream: second})[stream],
        engine.flush([stream])[stream]
    ]

    want = _session_frames(model, second, [(0, second.size)])
    for g, w in zip(got, want):
        _assert_frames_equal(g, w)


def test_a_model_without_lookahead_streams_on_time():
    model = conftest.tiny_model(lookahead_frames=0)
    audio = conftest.pcm(batch=1, frames=10)[0].numpy()
    engine = streams.Streams(model, capacity=2)
    stream = engine.open()

    got = [engine.push({stream: audio})[stream], engine.flush([stream])[stream]]

    want = _session_frames(model, audio, [(0, audio.size)])
    for g, w in zip(got, want):
        _assert_frames_equal(g, w)


def test_a_short_push_steps_nothing_until_a_frame_is_complete(model):
    engine = streams.Streams(model, capacity=1)
    stream = engine.open()

    frames = engine.push({stream: np.zeros(FRAME - 1, dtype=np.float32)})

    assert len(frames[stream]) == 0
    assert frames[stream].turns.shape == (0, len(model.cfg.turn_horizons))


def test_every_slot_taken_refuses_a_new_stream(model):
    engine = streams.Streams(model, capacity=2)
    engine.open()
    engine.open()

    with pytest.raises(RuntimeError, match="slots"):
        engine.open()


def test_open_streams_lists_what_is_not_flushed(model):
    engine = streams.Streams(model, capacity=3)
    a, b = engine.open(), engine.open()

    engine.flush([a])

    assert engine.open_streams == [b]


def test_an_unknown_stream_is_refused(model):
    engine = streams.Streams(model, capacity=1)

    with pytest.raises(KeyError):
        engine.push({7: np.zeros(FRAME, dtype=np.float32)})


@pytest.mark.parametrize("samples", [
    np.zeros((2, FRAME), dtype=np.float32),
    np.zeros(FRAME, dtype=np.int16),
])
def test_non_mono_or_integer_samples_are_refused(model, samples):
    engine = streams.Streams(model, capacity=1)
    stream = engine.open()

    with pytest.raises(ValueError):
        engine.push({stream: samples})


def test_a_capacity_below_one_is_refused(model):
    with pytest.raises(ValueError):
        streams.Streams(model, capacity=0)


def test_flushing_several_streams_keeps_the_others_streaming(model):
    audio = [
        conftest.pcm(batch=1, frames=20, seed=i)[0].numpy() for i in range(3)
    ]
    engine = streams.Streams(model, capacity=3)
    ids = [engine.open() for _ in audio]
    first = engine.push({s: a[:10 * FRAME] for s, a in zip(ids, audio)})

    engine.flush(ids[:2])
    rest = [
        engine.push({ids[2]: audio[2][10 * FRAME:]})[ids[2]],
        engine.flush([ids[2]])[ids[2]]
    ]

    want = _session_frames(model, audio[2], [(0, 10 * FRAME),
                                             (10 * FRAME, 20 * FRAME)])
    for g, w in zip([first[ids[2]]] + rest, want):
        _assert_frames_equal(g, w)


def test_a_refused_push_or_flush_consumes_no_stream_audio(model):
    audio = conftest.pcm(batch=1, frames=6)[0].numpy()
    engine = streams.Streams(model, capacity=2)
    stream = engine.open()
    with pytest.raises(ValueError):
        engine.push({
            stream: audio[:FRAME + 5],
            engine.open(): np.zeros((2, FRAME), dtype=np.float32)
        })
    with pytest.raises(KeyError):
        engine.flush([stream, 99])

    got = [
        engine.push({stream: audio[:FRAME + 5]})[stream],
        engine.push({stream: audio[FRAME + 5:]})[stream],
        engine.flush([stream])[stream]
    ]

    want = _session_frames(model, audio, [(0, FRAME + 5),
                                          (FRAME + 5, audio.size)])
    for g, w in zip(got, want):
        _assert_frames_equal(g, w)


def test_a_half_precision_model_streams_as_its_session():
    model = conftest.tiny_model().to(torch.bfloat16)
    audio = conftest.pcm(batch=1, frames=30)[0].numpy()
    engine = streams.Streams(model, capacity=1)
    stream = engine.open()

    got = [engine.push({stream: audio})[stream], engine.flush([stream])[stream]]

    want = _session_frames(model, audio, [(0, audio.size)])
    for g, w in zip(got, want):
        _assert_frames_equal(g, w)


def test_cuda_graphs_off_cuda_are_refused(model):
    with pytest.raises(ValueError, match="CUDA"):
        streams.Streams(model, capacity=2, cuda_graphs=True)


def _run_plan(engine, audio, plan, skip):
    """Opens per ``plan`` and pushes each tick the frames due so far; a
    stream whose (index, tick) is in ``skip`` pushes nothing that tick, so
    its next push carries several frames. Flushes once all is pushed."""
    ids, got, sent = {}, {i: [] for i in range(len(plan))}, {}
    for tick in range(max(start + n for start, n in plan) + 4):
        for i, (start, _) in enumerate(plan):
            if tick == start:
                ids[i], sent[i] = engine.open(), 0
        pieces = {}
        for i in ids:
            due = min(tick - plan[i][0] + 1, plan[i][1])
            if (i, tick) not in skip and sent[i] < due:
                pieces[i] = audio[i][sent[i] * FRAME:due * FRAME]
                sent[i] = due
        pushed = engine.push({ids[i]: a for i, a in pieces.items()})
        for i in pieces:
            got[i].append(pushed[ids[i]])
        done = [i for i in ids if sent[i] == plan[i][1]]
        flushed = engine.flush([ids[i] for i in done])
        for i in done:
            got[i].append(flushed[ids.pop(i)])
    return got


def test_graphed_streams_equal_eager_streams_and_sessions():
    if not torch.cuda.is_available():
        pytest.skip("CUDA graphs run on CUDA")
    model = conftest.tiny_model().cuda()
    plan = [(0, 30), (2, 12), (5, 25), (9, 4)]
    audio = [
        conftest.pcm(batch=1, frames=n, seed=i)[0].numpy()
        for i, (_, n) in enumerate(plan)
    ]
    # stream 2 skips ticks, so those ticks leave an open stream out
    skip = {(2, 7), (2, 8), (2, 15)}

    graphed = _run_plan(streams.Streams(model, capacity=4, cuda_graphs=True),
                        audio, plan, skip)

    eager = _run_plan(streams.Streams(model, capacity=4), audio, plan, skip)
    for i, a in enumerate(audio):
        session = streaming.Session(model)
        want = np.concatenate([session.push(a).speech, session.flush().speech])
        got = np.concatenate([f.speech for f in graphed[i]])
        np.testing.assert_array_equal(
            got, np.concatenate([f.speech for f in eager[i]]))
        np.testing.assert_allclose(got, want, atol=1e-5)
        assert all(
            np.array_equal(g.turns, e.turns)
            for g, e in zip(graphed[i], eager[i]))


def test_graphed_streams_record_on_the_model_device_not_the_current_one():
    if torch.cuda.device_count() < 2:
        pytest.skip("needs two CUDA devices")
    model = conftest.tiny_model().to("cuda:1")
    audio = conftest.pcm(batch=1, frames=8)[0].numpy()
    engine = streams.Streams(model, capacity=1, cuda_graphs=True)
    stream = engine.open()

    with torch.cuda.device(0):
        got = [
            engine.push({stream: audio})[stream],
            engine.flush([stream])[stream]
        ]

    want = _session_frames(model, audio, [(0, audio.size)])
    for g, w in zip(got, want):
        _assert_frames_equal(g, w)
