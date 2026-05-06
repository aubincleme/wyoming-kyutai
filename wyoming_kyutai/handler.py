"""Wyoming event handler for Kyutai STT transcription."""

import asyncio
import logging
import os
import tempfile
import wave
from collections import deque
from typing import Optional

import sentencepiece
import sphn
import torch
from moshi.models import loaders
from moshi.models.lm import LMGen
from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioChunkConverter, AudioStop
from wyoming.event import Event
from wyoming.info import Describe, Info
from wyoming.server import AsyncEventHandler

_LOGGER = logging.getLogger(__name__)


class KyutaiEventHandler(AsyncEventHandler):
    def __init__(
        self,
        wyoming_info: Info,
        checkpoint_info: loaders.CheckpointInfo,
        mimi,
        text_tokenizer: sentencepiece.SentencePieceProcessor,
        lm,
        device: str,
        lock: asyncio.Lock,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.wyoming_info_event = wyoming_info.event()
        self.checkpoint_info = checkpoint_info
        self.mimi = mimi
        self.text_tokenizer = text_tokenizer
        self.lm = lm
        self.device = device
        self._lock = lock

        self._language: Optional[str] = None
        self._wav_dir = tempfile.TemporaryDirectory()
        self._wav_path = os.path.join(self._wav_dir.name, "speech.wav")
        self._wav_file: Optional[wave.Wave_write] = None
        # Wyoming audio is normalised to 16 kHz / 16-bit / mono before writing.
        # sphn.read will resample from 16 kHz to the model's 24 kHz on load.
        self._audio_converter = AudioChunkConverter(rate=16000, width=2, channels=1)

    async def handle_event(self, event: Event) -> bool:
        if Describe.is_type(event.type):
            await self.write_event(self.wyoming_info_event)
            _LOGGER.debug("Sent info")
            return True

        if Transcribe.is_type(event.type):
            transcribe = Transcribe.from_event(event)
            self._language = transcribe.language
            _LOGGER.debug("Language set to %s", self._language)
            return True

        if AudioChunk.is_type(event.type):
            chunk = self._audio_converter.convert(AudioChunk.from_event(event))
            if self._wav_file is None:
                self._wav_file = wave.open(self._wav_path, "wb")
                self._wav_file.setframerate(chunk.rate)
                self._wav_file.setsampwidth(chunk.width)
                self._wav_file.setnchannels(chunk.channels)
            self._wav_file.writeframes(chunk.audio)
            return True

        if AudioStop.is_type(event.type):
            _LOGGER.debug("Audio stopped")
            if self._wav_file is not None:
                self._wav_file.close()
                self._wav_file = None

            # Serialise transcriptions: the mimi streaming context is stateful
            # and cannot be used by two requests at the same time.
            async with self._lock:
                text = await asyncio.to_thread(self._transcribe, self._wav_path)

            _LOGGER.info("Transcript: %s", text)
            await self.write_event(Transcript(text=text).event())
            _LOGGER.debug("Completed request")

            # Reset per-request state
            self._language = None
            return False

        return True

    def _transcribe(self, wav_path: str) -> str:
        """Synchronous transcription – runs in a thread pool worker."""
        # Read audio and resample to Mimi's native sample rate (24 kHz).
        in_pcms, _ = sphn.read(wav_path, sample_rate=int(self.mimi.sample_rate))
        in_pcms = torch.from_numpy(in_pcms).to(device=self.device)
        if in_pcms.dim() == 1:
            in_pcms = in_pcms.unsqueeze(0)  # [channels, samples]
        in_pcms = in_pcms[None, 0:1]  # [batch=1, ch=1, samples]

        # Pad to account for the model's audio prefix and lookahead delay.
        stt_cfg = self.checkpoint_info.stt_config
        sr = int(self.mimi.sample_rate)
        pad_left = int(stt_cfg.get("audio_silence_prefix_seconds", 0.0) * sr)
        pad_right = int((stt_cfg.get("audio_delay_seconds", 0.0) + 1.0) * sr)
        in_pcms = torch.nn.functional.pad(in_pcms, (pad_left, pad_right))

        frame_size = int(self.mimi.sample_rate / self.mimi.frame_rate)
        # Drop the last incomplete frame to avoid shape mismatches.
        chunks = deque(
            chunk
            for chunk in in_pcms.split(frame_size, dim=2)
            if chunk.shape[-1] == frame_size
        )

        if not chunks:
            return ""

        lm_gen = LMGen(self.lm, **self.checkpoint_info.lm_gen_config)
        text_pieces: list[str] = []

        with torch.no_grad(), self.mimi.streaming(1), lm_gen.streaming(1):
            first_frame = True
            while chunks:
                chunk = chunks.popleft()
                codes = self.mimi.encode(chunk)
                # The first step primes the model's delay buffer; the return value
                # is expected to be None (no output yet due to delays).
                if first_frame:
                    lm_gen.step(codes)
                    first_frame = False
                tokens = lm_gen.step(codes)
                if tokens is None:
                    continue
                token_id = int(tokens[0, 0].cpu().item())
                if token_id not in (0, 3):  # skip padding / EOS
                    piece = self.text_tokenizer.id_to_piece(token_id)
                    text_pieces.append(piece.replace("▁", " "))

        return "".join(text_pieces).strip()
