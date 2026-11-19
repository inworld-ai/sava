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
"""The two state-heavy parts of a ``Streams`` step as Triton kernels, for CUDA.

``advance`` takes one frame of the Mamba-2 recurrence of each stream's state
where it lies in the slot buffer: the state is read and written once, where
the eager step reads and writes it several times and gathers and scatters
the slots. Its rounding is the eager one's (accurate ``exp``, no fused
multiply-add), so the state it writes equals ``ssd.recurrent_step``'s.

``attend`` writes each stream's new key and value into its ring and attends
its group of query heads over the window, in float32 with an online softmax.
Like vLLM's decode attention, it splits the window across programs when the
batch alone would leave the GPU idle, and merges the parts. A stream's new key
is taken from registers by the first part and its slot skipped by the
others, so no part reads a slot another is writing.

Imported only by a ``Streams`` with ``cuda_graphs``; needs Triton 3.
"""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

KEYS_PER_TILE = 64
MAX_SPLITS = 8
# Programs that keep the GPU busy; the window is split until a step reaches
# this many.
BUSY_PROGRAMS = 256


@triton.jit
def _advance_kernel(state, slots, x, log_decay, b, c, y, heads, head_dim, x_row,
                    x_head, x_dim, decay_row, decay_head, b_row, b_head,
                    b_state, c_row, c_head, c_state, block_dim: tl.constexpr,
                    states: tl.constexpr):
    program = tl.program_id(0)
    row, head = program // heads, program % heads
    slot = tl.load(slots + row)
    dim = tl.program_id(1) * block_dim + tl.arange(0, block_dim)
    inside = dim < head_dim
    state_index = tl.arange(0, states)
    decay = libdevice.exp(
        tl.load(log_decay + row * decay_row + head * decay_head).to(tl.float32))
    x_t = tl.load(x + row * x_row + head * x_head + dim * x_dim,
                  mask=inside,
                  other=0.0).to(tl.float32)
    b_t = tl.load(b + row * b_row + head * b_head + state_index * b_state).to(
        tl.float32)
    c_t = tl.load(c + row * c_row + head * c_head + state_index * c_state).to(
        tl.float32)
    at = ((slot * heads + head) * head_dim + dim[:, None]) * states + \
        state_index[None, :]
    h = tl.load(state + at, mask=inside[:, None], other=0.0)
    h = libdevice.add_rn(libdevice.mul_rn(h, decay),
                         libdevice.mul_rn(x_t[:, None], b_t[None, :]))
    tl.store(state + at, h, mask=inside[:, None])
    tl.store(y + program * head_dim + dim,
             tl.sum(h * c_t[None, :], 1),
             mask=inside)


@triton.jit
def _attend_kernel(q, k, v, keys, values, slots, positions, peaks, totals, sums,
                   q_row, q_head, kv_row, kv_head, cache_slot, cache_frame,
                   cache_head, group, scale: tl.constexpr, window: tl.constexpr,
                   head_dim: tl.constexpr, block_group: tl.constexpr,
                   block_keys: tl.constexpr, splits: tl.constexpr):
    row = tl.program_id(0)
    head = tl.program_id(1)
    part = tl.program_id(2)
    slot = tl.load(slots + row)
    now = tl.load(positions + row)
    member = tl.arange(0, block_group)
    in_group = member < group
    dim = tl.arange(0, head_dim)
    query = tl.load(q + row * q_row +
                    (head * group + member[:, None]) * q_head + dim[None, :],
                    mask=in_group[:, None],
                    other=0.0).to(tl.float32)
    newest = now % window
    base = slot * cache_slot + head * cache_head
    peak = tl.full([block_group], -1e30, dtype=tl.float32)
    total = tl.zeros([block_group], dtype=tl.float32)
    acc = tl.zeros([block_group, head_dim], dtype=tl.float32)
    span: tl.constexpr = window // splits
    for start in range(part * span, (part + 1) * span, block_keys):
        frame = start + tl.arange(0, block_keys)
        # held: a position of this stream (>= 0) still in the window; the
        # newest slot holds the one the window drops, replaced below
        held = ((now - frame + window) % window <= now) & (frame != newest)
        at = base + frame[:, None] * cache_frame + dim[None, :]
        key = tl.load(keys + at, mask=held[:, None], other=0.0).to(tl.float32)
        value = tl.load(values + at, mask=held[:, None],
                        other=0.0).to(tl.float32)
        score = tl.dot(query, tl.trans(key), input_precision="ieee") * scale
        score = tl.where(held[None, :], score, float("-inf"))
        top = tl.maximum(peak, tl.max(score, 1))
        shrink = tl.exp(peak - top)
        weight = tl.exp(score - top[:, None])
        total = total * shrink + tl.sum(weight, 1)
        acc = acc * shrink[:, None] + tl.dot(
            weight, value, input_precision="ieee")
        peak = top
    if part == 0:
        new_key = tl.load(k + row * kv_row + head * kv_head + dim).to(
            tl.float32)
        new_value = tl.load(v + row * kv_row + head * kv_head + dim).to(
            tl.float32)
        score = tl.sum(query * new_key[None, :], 1) * scale
        top = tl.maximum(peak, score)
        shrink = tl.exp(peak - top)
        weight = tl.exp(score - top)
        total = total * shrink + weight
        acc = acc * shrink[:, None] + weight[:, None] * new_value[None, :]
        peak = top
        at = base + newest * cache_frame + dim
        tl.store(keys + at, new_key.to(keys.dtype.element_ty))
        tl.store(values + at, new_value.to(values.dtype.element_ty))
    out = ((row * tl.num_programs(1) + head) * group + member) * splits + part
    tl.store(peaks + out, peak, mask=in_group)
    tl.store(totals + out, total, mask=in_group)
    tl.store(sums + out[:, None] * head_dim + dim[None, :],
             acc,
             mask=in_group[:, None])


def supports_attention(head_dim: int, window: int, group: int) -> bool:
    """Whether ``attend`` takes this head size, window and head group."""
    return (head_dim >= 16 and head_dim & (head_dim - 1) == 0 and
            window % KEYS_PER_TILE == 0 and group <= 16)


def supports_advance(states: int) -> bool:
    """Whether ``advance`` takes this state size."""
    return states & (states - 1) == 0


def advance(state: torch.Tensor, slots: torch.Tensor, x: torch.Tensor,
            log_decay: torch.Tensor, b: torch.Tensor,
            c: torch.Tensor) -> torch.Tensor:
    """One frame of the recurrence of the slots' state, in place.

    Args:
        state: The slot buffer ``[slots, H, P, N]`` in float32, contiguous.
        slots: The rows' slots ``[B]``.
        x: ``[B, H, P]`` inputs already scaled by ``dt``.
        log_decay: ``[B, H]``.
        b: ``[B, H, N]``.
        c: ``[B, H, N]``.

    Returns:
        ``y [B, H, P]`` in ``x.dtype``, as ``ssd.recurrent_step`` returns it.
    """
    batch, heads, head_dim = x.shape
    y = torch.empty(batch, heads, head_dim, device=x.device)
    block_dim = 16
    _advance_kernel[(batch * heads,
                     triton.cdiv(head_dim, block_dim))](state,
                                                        slots,
                                                        x,
                                                        log_decay,
                                                        b,
                                                        c,
                                                        y,
                                                        heads,
                                                        head_dim,
                                                        *x.stride(),
                                                        *log_decay.stride(),
                                                        *b.stride(),
                                                        *c.stride(),
                                                        block_dim=block_dim,
                                                        states=state.shape[-1])
    return y.to(x.dtype)


def attend(keys: torch.Tensor, values: torch.Tensor, slots: torch.Tensor,
           positions: torch.Tensor, q: torch.Tensor, k: torch.Tensor,
           v: torch.Tensor, scale: float) -> torch.Tensor:
    """Writes one frame into each slot's ring and attends over its window.

    Args:
        keys: The ring buffer ``[slots, window, Hkv, D]``, contiguous.
        values: Like ``keys``.
        slots: The rows' slots ``[B]``.
        positions: The rows' frame positions ``[B]``.
        q: ``[B, 1, Hq, D]`` rotated queries, contiguous.
        k: ``[B, 1, Hkv, D]`` rotated keys, contiguous.
        v: ``[B, 1, Hkv, D]``, contiguous.
        scale: The attention scale.

    Returns:
        The attention output ``[B, 1, Hq, D]`` in ``q.dtype``.
    """
    batch, _, heads, head_dim = q.shape
    window, kv_heads = keys.shape[1], keys.shape[2]
    group = heads // kv_heads
    splits = 1
    while (splits < MAX_SPLITS and batch * kv_heads * splits < BUSY_PROGRAMS and
           window % (2 * splits * KEYS_PER_TILE) == 0):
        splits *= 2
    peaks = torch.empty(batch, heads, splits, device=q.device)
    totals = torch.empty_like(peaks)
    sums = torch.empty(batch, heads, splits, head_dim, device=q.device)
    _attend_kernel[(batch, kv_heads,
                    splits)](q,
                             k,
                             v,
                             keys,
                             values,
                             slots,
                             positions,
                             peaks,
                             totals,
                             sums,
                             q.stride(0),
                             q.stride(2),
                             k.stride(0),
                             k.stride(2),
                             keys.stride(0),
                             keys.stride(1),
                             keys.stride(2),
                             group,
                             scale=scale,
                             window=window,
                             head_dim=head_dim,
                             block_group=max(16, triton.next_power_of_2(group)),
                             block_keys=KEYS_PER_TILE,
                             splits=splits,
                             num_warps=4,
                             num_stages=2)
    top = peaks.max(-1, keepdim=True).values
    weight = torch.exp(peaks - top)
    out = (sums * weight[..., None]).sum(-2) / (totals * weight).sum(
        -1, keepdim=True)
    return out.reshape(batch, 1, heads, head_dim).to(q.dtype)
