<p align="center">
  <img src="https://raw.githubusercontent.com/hherb/ollamazip/main/assets/ollamazip_logo_small.png" alt="ollamazip logo" width="160" height="160">
</p>

<h1 align="center">ollamazip</h1>

<p align="center">
  Bundle Ollama models into single portable archives for transfer between machines.
</p>

<p align="center">
  <a href="https://pypi.org/project/ollamazip/"><img alt="PyPI" src="https://img.shields.io/pypi/v/ollamazip?style=flat-square"></a>
  <a href="https://pypi.org/project/ollamazip/"><img alt="Python" src="https://img.shields.io/pypi/pyversions/ollamazip?style=flat-square"></a>
  <a href="https://www.gnu.org/licenses/agpl-3.0"><img alt="License: AGPL-3.0-or-later" src="https://img.shields.io/badge/license-AGPL--3.0--or--later-blue?style=flat-square"></a>
</p>

## Installation

```bash
pip install ollamazip

# With zstd compression support (recommended):
pip install ollamazip[zstd]

# With the graphical interface:
pip install ollamazip[gui]

# GUI in a native desktop window (via pywebview):
pip install ollamazip[gui-native]
```

## Usage

```bash
# Pack a model into a portable .ollamazip file
ollamazip pack llama3:8b
ollamazip pack mymodel:latest -o mymodel.ollamazip
ollamazip pack qwen3:27b --compress gzip

# Unpack on another machine
ollamazip unpack llama3-8b.ollamazip
ollamazip unpack mymodel.ollamazip --name newname:v2
ollamazip unpack mymodel.ollamazip --verify

# Inspect an archive without unpacking
ollamazip list mymodel.ollamazip

# List installed Ollama models
ollamazip models

# Launch the graphical interface (opens in browser)
ollamazip gui
ollamazip gui --native    # open in a native window (requires gui-native extra)
```

## Graphical interface

`ollamazip gui` starts a small local web app (NiceGUI) that lets you:

- Browse installed Ollama models with sizes
- Pack any local model into a `.ollamazip` archive
- Rename or delete local models (with orphaned-blob pruning)
- Browse a folder of `.ollamazip` archives, inspect metadata, unpack, move, or delete them

By default it opens `http://127.0.0.1:8734` in your browser. Pass `--native` to
run in a standalone desktop window instead.

| Local models | Archives |
|:---:|:---:|
| ![Local models tab](https://raw.githubusercontent.com/hherb/ollamazip/main/assets/ollamazip_main.png) | ![Archives tab](https://raw.githubusercontent.com/hherb/ollamazip/main/assets/ollamazip_archives.png) |
| Browse, pack, rename, delete | Inspect, unpack, move, delete |

## Features

- **Single-file bundles**: Packs the manifest and all content-addressed blobs into one archive
- **Smart compression**: Auto-selects zstd (if available) or gzip; override with `--compress`
- **Blob deduplication**: Skips blobs already present on the target machine during unpack
- **SHA256 verification**: `--verify` validates every blob against manifest digests
- **Model renaming**: `--name newmodel:newtag` imports under a different name
- **Cross-platform**: Works on macOS, Linux, and Windows; respects `OLLAMA_MODELS` env var
- **No dependencies**: Core functionality uses only the Python standard library; zstd is optional

## How it works

Ollama stores models as a JSON manifest referencing content-addressed blobs (SHA256).
`ollamazip pack` reads the manifest, collects all referenced blobs, and bundles them
into a tar archive. `ollamazip unpack` extracts them into the target machine's Ollama
model store (`~/.ollama/models/`).

## License

Copyright © Horst Herb.

Licensed under the [GNU Affero General Public License v3.0 or later](LICENSE).
AGPL-3.0 means you are free to use, modify, and redistribute this software, but any
modified network-facing deployment must make the corresponding source available to
its users.
