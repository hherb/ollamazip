# ollamazip

Bundle Ollama models into single portable archives for transfer between machines.

## Installation

```bash
pip install ollamazip

# With zstd compression support (recommended):
pip install ollamazip[zstd]
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
```

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
