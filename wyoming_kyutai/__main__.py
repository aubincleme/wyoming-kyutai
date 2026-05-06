#!/usr/bin/env python3
"""Wyoming protocol server for Kyutai STT models."""

import argparse
import asyncio
import logging
from functools import partial

import torch
from moshi.models import loaders
from wyoming.info import AsrModel, AsrProgram, Attribution, Info
from wyoming.server import AsyncServer

from . import __version__
from .handler import KyutaiEventHandler

_LOGGER = logging.getLogger(__name__)

DEFAULT_HF_REPO = "kyutai/stt-1b-en_fr"
SUPPORTED_LANGUAGES = ["en", "fr"]


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Wyoming protocol server for Kyutai STT models"
    )
    parser.add_argument(
        "--uri",
        required=True,
        help="Server URI (e.g. tcp://0.0.0.0:10300 or unix:///tmp/kyutai.sock)",
    )
    parser.add_argument(
        "--hf-repo",
        default=DEFAULT_HF_REPO,
        help=f"HuggingFace repository for the model (default: {DEFAULT_HF_REPO})",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Inference device: cpu, cuda, cuda:0, … (default: cpu)",
    )
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "float32", "float16", "bfloat16"],
        help=(
            "Model weight dtype. 'auto' picks bfloat16 for CUDA and float32 for CPU "
            "(default: auto)"
        ),
    )
    parser.add_argument("--debug", action="store_true", help="Enable DEBUG logging")
    parser.add_argument(
        "--log-format",
        default=logging.BASIC_FORMAT,
        help="Python logging format string",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=__version__,
        help="Print version and exit",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format=args.log_format,
    )
    _LOGGER.debug(args)

    # Resolve dtype
    if args.dtype == "auto":
        dtype = torch.float32 if args.device == "cpu" else torch.bfloat16
    else:
        dtype = getattr(torch, args.dtype)

    _LOGGER.info("Loading checkpoint from %s", args.hf_repo)
    checkpoint_info = loaders.CheckpointInfo.from_hf_repo(args.hf_repo)

    _LOGGER.info("Loading Mimi audio codec")
    mimi = checkpoint_info.get_mimi(device=args.device)

    _LOGGER.info("Loading text tokenizer")
    text_tokenizer = checkpoint_info.get_text_tokenizer()

    _LOGGER.info("Loading language model (dtype=%s, device=%s)", dtype, args.device)
    lm = checkpoint_info.get_moshi(device=args.device, dtype=dtype)

    wyoming_info = Info(
        asr=[
            AsrProgram(
                name="kyutai-stt",
                description="Kyutai STT speech recognition",
                attribution=Attribution(
                    name="Kyutai",
                    url="https://kyutai.org",
                ),
                installed=True,
                version=__version__,
                models=[
                    AsrModel(
                        name=args.hf_repo,
                        description=f"Kyutai STT model ({args.hf_repo})",
                        attribution=Attribution(
                            name="Kyutai",
                            url="https://huggingface.co/kyutai",
                        ),
                        installed=True,
                        languages=SUPPORTED_LANGUAGES,
                        version=__version__,
                    )
                ],
            )
        ],
    )

    # One lock shared across all handler instances to serialise access to
    # the mimi streaming context (which is stateful and not concurrency-safe).
    transcription_lock = asyncio.Lock()

    server = AsyncServer.from_uri(args.uri)
    _LOGGER.info("Ready")

    await server.run(
        partial(
            KyutaiEventHandler,
            wyoming_info,
            checkpoint_info,
            mimi,
            text_tokenizer,
            lm,
            args.device,
            transcription_lock,
        )
    )


def run() -> None:
    asyncio.run(main())


if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        pass
