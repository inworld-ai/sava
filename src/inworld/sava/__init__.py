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
"""SAVA: streaming voice activity and end-of-turn detection in one model.

    from inworld import sava

    model = sava.load("inworld/sava")
    curves = sava.run(model, sava.load_audio("recording.wav"))
    turns = sava.segment(curves, sava.Policy(), model.lookahead_frames)

    session = sava.Session(model)
    for piece in pieces_of_16k_audio:
        frames = session.push(piece)
    frames = session.flush()
"""

from inworld.sava import audio
from inworld.sava import checkpoint
from inworld.sava import config
from inworld.sava import segmenter
from inworld.sava import streaming
from inworld.sava import streams

__version__ = "1.0.0"

SAMPLE_RATE = config.SAMPLE_RATE
FRAME_SECONDS = config.FRAME_SECONDS

load = checkpoint.load
load_audio = audio.load
save_audio = audio.save
run = streaming.run
Session = streaming.Session
Streams = streams.Streams
Curves = streaming.Curves
Frames = streaming.Frames
Policy = segmenter.Policy
Turn = segmenter.Turn
TurnSegmenter = segmenter.TurnSegmenter
segment = segmenter.segment

__all__ = [
    "SAMPLE_RATE",
    "FRAME_SECONDS",
    "Curves",
    "Frames",
    "Policy",
    "Session",
    "Streams",
    "Turn",
    "TurnSegmenter",
    "load",
    "load_audio",
    "run",
    "save_audio",
    "segment",
]
