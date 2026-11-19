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
"""Many streams through one model, stepped together.

A ``Session`` steps one stream; a server with many live calls would step
each separately, one small step per call per 40 ms frame. ``Streams`` steps
every call with a whole frame ready as one batch: a stream may open, push or
flush at any time, so the streams in a step are at different frames.

Each stream owns a slot of fixed state buffers and its own frame count. Its
rotary positions, the reach of its attention and its first frames' no-speech
floor follow its own count, as in its own ``Session``; the attention cache is
a ring of the window, the newest key written over the one the window drops,
so a step writes one frame of state per stream rather than rebuilding it.
Open streams keep consecutive slots (a flushed stream's slot takes the last
stream's state), so a step of every open stream works on its state in place.
With ``cuda_graphs``, a step of every open stream replays as a CUDA graph of
the next recorded batch size, one launch instead of several hundred.
Each stream's outputs equal those of a ``Session`` given the same audio, to
float32 rounding (the attention sums its window in ring order).
"""

from collections.abc import Iterable, Mapping
import dataclasses
import itertools

import numpy as np
import torch

from inworld.sava import config
from inworld.sava import mel
from inworld.sava import model as model_lib
from inworld.sava import modules
from inworld.sava import ssd
from inworld.sava import streaming

# Batch sizes recorded as CUDA graphs, as an LLM server records its decode
# steps; a step of n streams replays the smallest size of at least n.
GRAPH_SIZES = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)


@dataclasses.dataclass
class _Stream:
    slot: int
    pending: np.ndarray
    frames: int = 0


class _Rows:
    """The slots of one step: a range, used in place, or any slots, gathered
    and scattered."""

    def __init__(self, slots: list[int], device: torch.device) -> None:
        self.index = torch.tensor(slots, device=device)
        first = slots[0]
        consecutive = slots == list(range(first, first + len(slots)))
        self.span = slice(first, first + len(slots)) if consecutive else None

    def take(self, buffer: torch.Tensor) -> torch.Tensor:
        if self.span is not None:
            return buffer[self.span]
        return buffer.index_select(0, self.index)

    def put(self, buffer: torch.Tensor, value: torch.Tensor) -> None:
        if self.span is not None:
            buffer[self.span].copy_(value)
        else:
            buffer.index_copy_(0, self.index, value)


def _advance_in_place(state: torch.Tensor, x: torch.Tensor,
                      log_decay: torch.Tensor, b: torch.Tensor,
                      c: torch.Tensor) -> torch.Tensor:
    """``ssd.recurrent_step`` with ``state`` advanced where it is: the same
    operations in the same order, so the same values."""
    state.mul_(torch.exp(log_decay.float())[..., None, None])
    state.add_(x.float().unsqueeze(-1) * b.float().unsqueeze(-2))
    return torch.einsum("bhpn,bhn->bhp", state, c.float()).to(x.dtype)


def _rotate(x: torch.Tensor, positions: torch.Tensor,
            freqs: torch.Tensor) -> torch.Tensor:
    """``modules.apply_rope`` of ``x [B, 1, H, D]`` at one position per row."""
    half = x.shape[-1] // 2
    angles = positions.float()[:, None] * freqs[None]
    cos, sin = angles.cos()[:, None, None], angles.sin()[:, None, None]
    x1, x2 = x.float()[..., :half], x.float()[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin],
                     dim=-1).to(x.dtype)


class Streams:
    """Up to ``capacity`` streams of one model, stepped one frame at a time.

    ``open`` starts a stream and returns its id; ``push`` takes samples for
    any streams and returns, per stream, the frames it stepped, as
    ``Session.push`` would; ``flush`` ends streams as ``Session.flush`` does
    and frees their slots. ``cuda_graphs`` (CUDA only) replays each step of
    every open stream as a CUDA graph, recorded the first time a batch size
    is reached. Not safe for concurrent calls.

    Raises:
        ValueError: For a capacity below one, or ``cuda_graphs`` off CUDA.
    """

    def __init__(self,
                 model: model_lib.SavaModel,
                 capacity: int,
                 cuda_graphs: bool = False) -> None:
        if capacity < 1:
            raise ValueError("capacity must be at least 1.")
        self.model = model
        self.capacity = capacity
        self.lag = model.lookahead_frames
        device = next(model.parameters()).device
        self.device = device
        if cuda_graphs and device.type != "cuda":
            raise ValueError("cuda_graphs needs the model on a CUDA device.")
        self._graphed = cuda_graphs
        self._graphs = {}
        slots = capacity
        # float32 state whatever the model's dtype, as a Session keeps it
        model.embedder.init_state(1, device)
        self._frontend = torch.zeros(slots, mel.CONTEXT, device=device)
        self._mixer = model.mixer.init_state(slots, device, torch.float32).tail
        self._mamba = {}
        self._attention = {}
        for name, block in self._blocks():
            state = block.init_state(slots, device, torch.float32)
            if isinstance(block, modules.MambaBlock):
                self._mamba[name] = state
            else:
                self._attention[name] = state
        self._previous = torch.zeros(slots, device=device)
        self._positions = torch.zeros(slots, dtype=torch.long, device=device)
        self._streams: dict[int, _Stream] = {}
        self._ids = itertools.count()

    def _blocks(self):
        model = self.model
        for index, block in enumerate(model.trunk):
            yield f"trunk.{index}", block
        yield "vad", model.vad_branch.block
        yield "turn", model.turn_branch.block

    @property
    def open_streams(self) -> list[int]:
        """The ids of the streams not yet flushed."""
        return list(self._streams)

    def open(self) -> int:
        """Starts a stream and returns its id.

        Raises:
            RuntimeError: When every slot holds a stream.
        """
        slot = len(self._streams)
        if slot == self.capacity:
            raise RuntimeError(
                f"All {self.capacity} slots hold a stream; flush one first.")
        for buffer in (self._frontend, self._mixer, self._previous,
                       self._positions):
            buffer[slot].zero_()
        for state in self._mamba.values():
            state.conv_tail[slot].zero_()
            state.ssm[slot].zero_()
        stream = next(self._ids)
        self._streams[stream] = _Stream(slot, np.zeros(0, dtype=np.float32))
        return stream

    def push(
        self, pieces: Mapping[int, np.ndarray | torch.Tensor]
    ) -> dict[int, streaming.Frames]:
        """Appends samples to streams and steps every whole frame now ready.

        Args:
            pieces: Mono floating-point 16 kHz samples by stream id.

        Returns:
            The stepped frames of each stream in ``pieces``, as its
            ``Session.push`` would return them.

        Raises:
            KeyError: For an id that is not open.
            ValueError: If samples are not a one-dimensional float array.
        """
        # every id and piece is checked before any stream takes its piece
        streams = {stream_id: self._streams[stream_id] for stream_id in pieces}
        checked = {
            stream_id:
                streaming._mono_float(samples
                                     )  # pylint: disable=protected-access
            for stream_id, samples in pieces.items()
        }
        chunks = {}
        for stream_id, samples in checked.items():
            stream = streams[stream_id]
            pending = np.concatenate([stream.pending, samples])
            whole = pending.size - pending.size % config.FRAME_SAMPLES
            chunks[stream_id], stream.pending = pending[:whole], pending[whole:]
        return self._run(chunks)

    def flush(self, stream_ids: Iterable[int]) -> dict[int, streaming.Frames]:
        """Ends streams as ``Session.flush`` does and frees their slots.

        Pads each stream's last partial frame and steps ``lag`` frames of
        silence, so its last frames get their verdicts.

        Raises:
            KeyError: For an id that is not open.
        """
        # every id is checked before any stream is ended
        streams = {
            stream_id: self._streams[stream_id]
            for stream_id in dict.fromkeys(stream_ids)
        }
        chunks = {}
        for stream_id, stream in streams.items():
            tail = (-stream.pending.size) % config.FRAME_SAMPLES
            chunks[stream_id] = np.concatenate([
                stream.pending,
                np.zeros(tail + self.lag * config.FRAME_SAMPLES,
                         dtype=np.float32)
            ])
            stream.pending = np.zeros(0, dtype=np.float32)
        frames = self._run(chunks)
        for stream_id in chunks:
            self._release(self._streams.pop(stream_id).slot)
        return frames

    def _buffers(self):
        yield from (self._frontend, self._mixer, self._previous,
                    self._positions)
        for state in self._mamba.values():
            yield from (state.conv_tail, state.ssm)
        for state in self._attention.values():
            yield from (state.keys, state.values)

    def _release(self, slot: int) -> None:
        """Frees a slot, moving the last open stream into it."""
        last = len(self._streams)
        if slot == last:
            return
        for buffer in self._buffers():
            buffer[slot].copy_(buffer[last])
        mover = next(s for s in self._streams.values() if s.slot == last)
        mover.slot = slot

    def _run(self, chunks: Mapping[int,
                                   np.ndarray]) -> dict[int, streaming.Frames]:
        """Steps every chunk's frames, a frame of every chunk per step."""
        counts = {s: c.size // config.FRAME_SAMPLES for s, c in chunks.items()}
        outputs = {s: [] for s in chunks}
        for frame in range(max(counts.values(), default=0)):
            ids = sorted((s for s in chunks if counts[s] > frame),
                         key=lambda s: self._streams[s].slot)
            pcm = torch.from_numpy(
                np.stack([
                    chunks[s][frame * config.FRAME_SAMPLES:(frame + 1) *
                              config.FRAME_SAMPLES] for s in ids
                ])).to(self.device)
            slots = [self._streams[s].slot for s in ids]
            with torch.inference_mode():
                speech, turns = self._step_slots(pcm, slots)
            speech, turns = speech.cpu().numpy(), turns.cpu().numpy()
            for row, stream_id in enumerate(ids):
                outputs[stream_id].append((speech[row], turns[row]))
        frames = {}
        for stream_id, rows in outputs.items():
            stream = self._streams[stream_id]
            turns = np.array([t for _, t in rows], dtype=np.float32).reshape(
                len(rows), len(self.model.cfg.turn_horizons))
            frames[stream_id] = streaming.Frames(
                stream.frames, np.array([s for s, _ in rows], dtype=np.float32),
                turns[:, 0], turns)
            stream.frames += len(rows)
        return frames

    def _step_slots(self, pcm: torch.Tensor,
                    slots: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        """One frame of the streams in ``slots``: their probabilities, speech
        as ``Session`` emits it ``[B]`` and turn ``[B, K]``."""
        if self._graphed and slots == list(range(len(self._streams))):
            # the model's device, which need not be the current one
            with torch.cuda.device(self.device):
                return self._replay(pcm)
        return self._step(pcm, _Rows(slots, self.device))

    def _replay(self, pcm: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """A step of every open stream, which hold slots 0 to n - 1, as a
        graph of the next recorded size: its extra rows step free slots, whose
        state ``open`` resets."""
        batch = pcm.shape[0]
        size = min(next((s for s in GRAPH_SIZES if s >= batch), self.capacity),
                   self.capacity)
        if size not in self._graphs:
            self._graphs[size] = self._record(size)
        graph, static, outputs, _ = self._graphs[size]
        static[:batch].copy_(pcm)
        static[batch:].zero_()
        graph.replay()
        return tuple(output[:batch] for output in outputs)

    def _record(self, size: int):
        """Records the step of slots 0 to ``size`` - 1. Warming up runs the
        step, so their state is restored after it. The rows are kept with the
        graph: it reads their slot indices where they were when recorded."""
        rows = _Rows(list(range(size)), self.device)
        saved = [buffer[:size].clone() for buffer in self._buffers()]
        static = torch.zeros(size, config.FRAME_SAMPLES, device=self.device)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):
                self._step(static, rows)
        torch.cuda.current_stream().wait_stream(side)
        for buffer, value in zip(self._buffers(), saved):
            buffer[:size].copy_(value)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outputs = self._step(static, rows)
        return graph, static, outputs, rows

    def _step(self, pcm: torch.Tensor,
              slots: _Rows) -> tuple[torch.Tensor, torch.Tensor]:
        model = self.model
        positions = slots.take(self._positions).clone()
        joined = torch.cat([slots.take(self._frontend), pcm.float()], dim=1)
        features = model.embedder.features(joined)
        slots.put(self._frontend, joined[:, -mel.CONTEXT:])
        rows = model.mixer.rows(features, pcm[:, None])
        tail = slots.take(self._mixer)
        hidden = model.mixer._mix(rows, tail)  # pylint: disable=protected-access
        slots.put(self._mixer, modules.next_tail(rows, tail))
        for name, block in zip((n for n, _ in self._blocks()), model.trunk):
            if isinstance(block, modules.MambaBlock):
                hidden = self._mamba_block(name, block, hidden, slots)
            else:
                hidden = self._attention_block(name, block, hidden, slots,
                                               positions)
        hidden = model.final_norm(hidden)
        vad = model.vad_branch
        features = vad.out_norm(
            self._attention_block("vad", vad.block,
                                  vad.down(model.vad_norm(hidden)), slots,
                                  positions))
        speech = vad.readout(features).squeeze(-1)
        signal = torch.sigmoid(speech.detach().float())
        emitted = speech
        if model.vad_la_readout is not None:
            lookahead = model.vad_la_readout(features).squeeze(-1)
            emitted = torch.where((positions < model.lookahead_frames)[:, None],
                                  torch.full_like(lookahead,
                                                  model_lib.NO_SPEECH_LOGIT),
                                  lookahead)
        turn_input = model._turn_input(  # pylint: disable=protected-access
            hidden,
            slots.take(self._previous).unsqueeze(1))
        branch = model.turn_branch
        turn = branch.readout(
            branch.out_norm(
                self._attention_block("turn", branch.block,
                                      branch.down(turn_input), slots,
                                      positions)))
        slots.put(self._previous, signal[:, -1])
        slots.put(self._positions, positions + 1)
        return (torch.sigmoid(emitted[:, 0].float()),
                torch.sigmoid(turn[:, 0].float()))

    def _mamba_block(self, name: str, block: modules.MambaBlock,
                     x: torch.Tensor, slots: _Rows) -> torch.Tensor:
        mixer = block.mixer
        state = self._mamba[name]
        tail = slots.take(state.conv_tail)
        z, xs, log_decay, b, c, dt, raw = mixer._split(  # pylint: disable=protected-access
            block.mixer_norm(x), tail)
        slots.put(state.conv_tail, modules.next_tail(raw, tail))
        y = self._advance(state.ssm, slots, (xs * dt[..., None])[:, 0],
                          log_decay[:, 0], b[:, 0], c[:, 0])
        x = x + mixer._output(y.to(xs.dtype).unsqueeze(1), xs, z)  # pylint: disable=protected-access
        return x + block.mlp(block.mlp_norm(x))

    def _advance(self, ssm: torch.Tensor, slots: _Rows, x: torch.Tensor,
                 log_decay: torch.Tensor, b: torch.Tensor,
                 c: torch.Tensor) -> torch.Tensor:
        """One frame of the recurrence of the slots' state, written back."""
        if slots.span is not None:
            return _advance_in_place(ssm[slots.span], x, log_decay, b, c)
        y, advanced = ssd.recurrent_step(x, log_decay, b, c, slots.take(ssm))
        slots.put(ssm, advanced)
        return y

    def _attention_block(self, name: str, block: modules.AttentionBlock,
                         x: torch.Tensor, slots: _Rows,
                         positions: torch.Tensor) -> torch.Tensor:
        attention = block.attention
        cfg = attention.cfg
        batch = x.shape[0]
        q, k, v = attention.qkvr(block.attn_norm(x)).split(
            (attention.q_width, attention.kv_width, attention.kv_width), dim=-1)
        q = q.reshape(batch, 1, cfg.num_heads, cfg.head_dim)
        k = k.reshape(batch, 1, cfg.num_kv_heads, cfg.head_dim)
        v = v.reshape(batch, 1, cfg.num_kv_heads, cfg.head_dim)
        freqs = attention._freqs_on(x.device)  # pylint: disable=protected-access
        q = _rotate(modules.rms_norm(q, attention.q_norm, attention.eps),
                    positions, freqs)
        k = _rotate(modules.rms_norm(k, attention.k_norm, attention.eps),
                    positions, freqs)
        out = self._attend(self._attention[name], slots, positions, q, k, v,
                           cfg.sliding_window, attention.scale)
        if cfg.xsa:
            out = modules.reject_own_value(out, v, cfg.xsa_alpha)
        x = x + attention.wo(out.reshape(batch, 1, -1))
        return x + block.mlp(block.mlp_norm(x))

    def _attend(self, state: modules.AttentionState, slots: _Rows,
                positions: torch.Tensor, q: torch.Tensor, k: torch.Tensor,
                v: torch.Tensor, window: int, scale: float) -> torch.Tensor:
        """Writes the frame's key and value into each slot's ring and attends
        over the ring: the ``window`` newest positions, none before 0."""
        newest = positions % window
        state.keys[slots.index, newest] = k[:, 0].to(state.keys.dtype)
        state.values[slots.index, newest] = v[:, 0].to(state.values.dtype)
        keys, values = slots.take(state.keys), slots.take(state.values)
        ring = torch.arange(window, device=q.device)
        age = (positions[:, None] - ring[None, :]) % window
        held = age <= positions[:, None]
        batch, _, heads, head_dim = q.shape
        kv_heads = keys.shape[2]
        group = heads // kv_heads
        qh = q.permute(0, 2, 1, 3).float().reshape(batch, kv_heads, group,
                                                   head_dim)
        kh = keys.permute(0, 2, 1, 3).float()
        vh = values.permute(0, 2, 1, 3).float()
        scores = (qh @ kh.transpose(-1, -2) * scale).masked_fill(
            ~held[:, None, None], float("-inf"))
        out = torch.softmax(scores, dim=-1) @ vh
        return out.reshape(batch, heads, 1, head_dim).permute(0, 2, 1,
                                                              3).to(q.dtype)
