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

<p align="center">
  <a href="docs/essay.md"><b>Why would I need ollamazip?</b></a>
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
model store.

## Where ollamazip looks for models

ollamazip discovers Ollama's model directory automatically. The candidates,
probed in order, are:

1. `$OLLAMA_MODELS` if set (hard override; `~` and `$VAR` are expanded).
2. The per-user store: `~/.ollama/models` on macOS and Linux,
   `%USERPROFILE%\.ollama\models` on Windows.
3. **Linux only:** `/usr/share/ollama/.ollama/models` — the default for the
   official Ollama systemd-service install, where Ollama runs as a dedicated
   `ollama` system user.

If candidate (2) is empty/missing on Linux but (3) contains models,
ollamazip uses (3). Run `ollamazip models` to see which path was picked;
if the list is empty, it prints every path that was searched.

### Linux systemd-service note

If you installed Ollama via the official Linux script, the model store at
`/usr/share/ollama/.ollama/models` is owned by the `ollama` system user. As a
regular user you can **read** that directory (so `ollamazip pack` and
`ollamazip models` work), but you **cannot write** to it. To install models
with `ollamazip unpack`, pick one:

- Re-run with `sudo`: `sudo ollamazip unpack <archive>`
- Add yourself to the `ollama` group and grant group write:
  ```bash
  sudo usermod -aG ollama "$USER"   # log out and back in
  sudo chmod -R g+w /usr/share/ollama/.ollama/models
  ```
- Move Ollama's storage to a directory you own and tell the service:
  ```bash
  sudo systemctl edit ollama.service
  # add under [Service]:
  #     Environment="OLLAMA_MODELS=/path/you/own"
  sudo systemctl restart ollama
  export OLLAMA_MODELS=/path/you/own
  ```

ollamazip prints similar guidance whenever a write operation (`unpack`,
`delete`, `rename`) hits a permission error on the system path.

> **Fresh-install note:** if you run `ollamazip unpack` *before* installing
> Ollama as a systemd service (so neither candidate dir contains models
> yet), ollamazip writes to `~/.ollama/models`. The systemd-service Ollama
> won't see those models — point the service at your user dir with the
> `systemctl edit` recipe above, or move the unpacked files into
> `/usr/share/ollama/.ollama/models` after installing the service.

## License

Copyright © Horst Herb.

Licensed under the [GNU Affero General Public License v3.0 or later](LICENSE).
AGPL-3.0 means you are free to use, modify, and redistribute this software, but any
modified network-facing deployment must make the corresponding source available to
its users.
