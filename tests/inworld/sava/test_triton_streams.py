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
"""The CUDA kernels of graphed Streams against the eager step, on CUDA."""

import numpy as np
import pytest
import torch

import conftest
from inworld.sava import config
from inworld.sava import ssd
from inworld.sava import streaming
from inworld.sava import streams


def _kernels_or_skip():
    if not torch.cuda.is_available():
        pytest.skip("the Streams kernels run on CUDA")
    pytest.importorskip("triton.language.extra.libdevice")
    # pylint: disable-next=import-outside-toplevel
    from inworld.sava import triton_streams
    return triton_streams


def _randn(*shape, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=g).cuda()


def test_advance_writes_the_eager_state_into_its_slots():
    kernels = _kernels_or_skip()
    state = _randn(6, 4, 24, 16)
    slots = torch.tensor([4, 0, 2], device="cuda")
    x, log_decay = _randn(3, 4, 24, seed=1), -_randn(3, 4, seed=2).abs()
    b, c = _randn(3, 4, 16, seed=3), _randn(3, 4, 16, seed=4)
    want_y, want = ssd.recurrent_step(x, log_decay, b, c, state[slots])
    untouched = state[[1, 3, 5]].clone()

    y = kernels.advance(state, slots, x, log_decay, b, c)

    assert torch.equal(state[slots], want)
    assert torch.equal(state[[1, 3, 5]], untouched)
    torch.testing.assert_close(y, want_y, atol=1e-5, rtol=1e-5)


def _eager_ring(keys, values, slots, positions, q, k, v, scale):
    window = keys.shape[1]
    newest = positions % window
    keys[slots, newest] = k[:, 0]
    values[slots, newest] = v[:, 0]
    age = (positions[:, None] -
           torch.arange(window, device=q.device)[None]) % window
    held = age <= positions[:, None]
    batch, _, heads, head_dim = q.shape
    kv_heads = keys.shape[2]
    qh = q.permute(0, 2, 1, 3).reshape(batch, kv_heads, heads // kv_heads,
                                       head_dim)
    kh, vh = (t[slots].permute(0, 2, 1, 3) for t in (keys, values))
    scores = (qh @ kh.transpose(-1, -2) * scale).masked_fill(
        ~held[:, None, None], float("-inf"))
    out = torch.softmax(scores, dim=-1) @ vh
    return out.reshape(batch, heads, 1, head_dim).permute(0, 2, 1, 3)


@pytest.mark.parametrize("window,rows", [(64, 1), (512, 1), (512, 3),
                                         (512, 40)])
def test_attend_equals_the_eager_ring(window, rows):
    kernels = _kernels_or_skip()
    keys, values = _randn(41, window, 2, 32), _randn(41, window, 2, 32, seed=1)
    slots = torch.randperm(41, device="cuda")[:rows]
    # fresh streams, streams inside their first window and wrapped ones
    positions = torch.tensor([0, 5, window - 1, window, 3 * window + 7] * 8,
                             device="cuda")[:rows]
    q, k, v = (
        _randn(rows, 1, h, 32, seed=s) for s, h in ((2, 8), (3, 2), (4, 2)))
    eager_keys, eager_values = keys.clone(), values.clone()

    out = kernels.attend(keys, values, slots, positions, q, k, v, 32**-0.5)

    want = _eager_ring(eager_keys, eager_values, slots, positions, q, k, v,
                       32**-0.5)
    assert torch.equal(keys, eager_keys) and torch.equal(values, eager_values)
    torch.testing.assert_close(out, want, atol=1e-5, rtol=1e-5)


def _cuda_model():
    attention = config.AttentionConfig(num_heads=4,
                                       num_kv_heads=2,
                                       head_dim=16,
                                       sliding_window=64)
    branch = config.BranchConfig(hidden_size=32,
                                 ffn_size=64,
                                 attention=attention)
    return conftest.tiny_model(attention=attention,
                               vad_branch=branch,
                               turn_branch=branch).cuda()


def test_graphed_streams_with_the_kernels_each_equal_their_own_session():
    _kernels_or_skip()
    model = _cuda_model()
    plan = [(0, 150), (7, 40), (30, 120), (31, 70)]
    audio = [
        conftest.pcm(batch=1, frames=n, seed=i)[0].numpy()
        for i, (_, n) in enumerate(plan)
    ]
    engine = streams.Streams(model, capacity=4, cuda_graphs=True)
    ids, got = {}, {i: [] for i in range(len(plan))}

    for tick in range(160):
        for i, (start, n) in enumerate(plan):
            if tick == start:
                ids[i] = engine.open()
        live = {i: tick - plan[i][0] for i in ids}
        pushed = engine.push({
            ids[i]:
                audio[i]
                [t * config.FRAME_SAMPLES:(t + 1) * config.FRAME_SAMPLES]
            for i, t in live.items()
            if t < plan[i][1]
        })
        for i in live:
            if live[i] < plan[i][1]:
                got[i].append(pushed[ids[i]])
        done = [i for i in ids if live[i] == plan[i][1]]
        flushed = engine.flush([ids[i] for i in done])
        for i in done:
            got[i].append(flushed[ids.pop(i)])

    for i, a in enumerate(audio):
        session = streaming.Session(model)
        want = [session.push(a), session.flush()]
        speech = np.concatenate([f.speech for f in got[i]])
        turns = np.concatenate([f.turns for f in got[i]])
        np.testing.assert_allclose(speech,
                                   np.concatenate([w.speech for w in want]),
                                   atol=1e-5)
        np.testing.assert_allclose(turns,
                                   np.concatenate([w.turns for w in want]),
                                   atol=1e-5)
