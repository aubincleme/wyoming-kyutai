# wyoming-kyutai

A [Wyoming protocol](https://github.com/OHF-Voice/wyoming) server for speech-to-text using [Kyutai STT](https://huggingface.co/kyutai/stt-1b-en_fr) models. Designed to integrate with [Home Assistant](https://www.home-assistant.io/) voice pipelines and any other Wyoming-compatible client.

## Supported models

| HuggingFace repo | Languages | Parameters |
|---|---|---|
| `kyutai/stt-1b-en_fr` (default) | English, French | 1 B |
| `kyutai/stt-2.6b-en` | English | 2.6 B |

The model is downloaded automatically from HuggingFace Hub on first run and cached in `~/.cache/huggingface/hub/`.

## Requirements

- Python 3.10+
- [PyTorch](https://pytorch.org/get-started/locally/) (CPU or CUDA)
- The `moshi` package pulls in `sphn` and `sentencepiece` as transitive dependencies

## Installation

```bash
pip install -r requirements.txt
```

For CUDA inference, install the appropriate PyTorch build first:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

## Usage

### TCP socket (recommended for Home Assistant)

```bash
wyoming-kyutai --uri tcp://0.0.0.0:10300
```

### Unix socket

```bash
wyoming-kyutai --uri unix:///tmp/kyutai.sock
```

### CUDA inference

```bash
wyoming-kyutai --uri tcp://0.0.0.0:10300 --device cuda --dtype bfloat16
```

### Different model

```bash
wyoming-kyutai --uri tcp://0.0.0.0:10300 --hf-repo kyutai/stt-2.6b-en
```

## CLI reference

| Option | Default | Description |
|---|---|---|
| `--uri` | *(required)* | Server URI: `tcp://host:port` or `unix:///path` |
| `--hf-repo` | `kyutai/stt-1b-en_fr` | HuggingFace repository to load the model from |
| `--device` | `cpu` | Inference device (`cpu`, `cuda`, `cuda:0`, …) |
| `--dtype` | `auto` | Weight dtype: `auto`, `float32`, `float16`, `bfloat16`. `auto` picks `float32` on CPU and `bfloat16` on CUDA |
| `--debug` | off | Enable DEBUG-level logging |
| `--version` | | Print version and exit |

## Home Assistant integration

Add the Wyoming integration in Home Assistant and point it at the host and port where this server is running. The server advertises English and French as supported languages; language detection is automatic.

## License

MIT — see [LICENSE](LICENSE).
