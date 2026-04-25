# Stop re-downloading Ollama models: meet ollamazip

If you've used Ollama for any length of time, you've probably done this: pulled `qwen3:32b` on your desktop, sat through 19 GB of download, then walked over to the laptop in the next room and… waited 19 GB all over again. I have three machines I run models on, a metered link at one of them, and an external SSD I keep meaning to use as cold storage. The friction adds up fast.

The annoying part is that the bytes already exist on disk. They're just locked inside a layout that doesn't move well.

## Why is this harder than it should be?

GGUF spoiled us. A GGUF file is one self-contained blob — copy it, scp it, drop it on a USB stick, and it works anywhere with llama.cpp.

Ollama doesn't store models like that. Under `~/.ollama/models/` you'll find:

```
manifests/registry.ollama.ai/library/llama3/8b
blobs/sha256-6a0746a1ec1aef3e7e...
blobs/sha256-4fa551d4f938f68b8c...
blobs/sha256-...
```

The manifest is a small JSON file referencing several content-addressed blobs — the weights, the tokenizer, a chat template, a parameter file, a license. A single model is four to eight files spread across two directory trees. There's no `ollama export`. The closest workflow is `ollama push`/`pull` against a registry, which means standing up a server or relying on a cloud registry — overkill if all you want is to copy one model from one box to another.

So people improvise: tar up the whole `~/.ollama` directory (lumps every model together, all-or-nothing), or copy individual blob hashes by hand (error-prone, and you have to recreate the manifest path on the other side). Neither survives ordinary use.

## ollamazip in 30 seconds

`ollamazip` is a small Python tool that turns one Ollama model into one file:

```bash
pip install ollamazip[zstd]

ollamazip pack llama3:8b              # → llama3-8b.ollamazip
ollamazip unpack llama3-8b.ollamazip  # restores it on another machine
ollamazip list  llama3-8b.ollamazip   # peek inside without unpacking
ollamazip models                      # list local Ollama models
```

That's the whole loop. The archive is a single `.ollamazip` file (zstd-compressed tar by default, gzip if zstandard isn't installed) containing the manifest, every blob it references, and a small `metadata.json` describing what's inside. Drop it on an SSD, attach it to an email, scp it across the LAN — it doesn't matter, it's just a file.

## A GUI for the click-inclined

Not everyone wants to live in a terminal — especially when you're walking a non-technical family member through "please move that 30 GB thing onto the external drive." `ollamazip gui` starts a small local web app (built on [NiceGUI](https://nicegui.io/)) that wraps the same core API:

```bash
pip install ollamazip[gui]
ollamazip gui              # opens http://127.0.0.1:8734 in your browser
ollamazip gui --native     # opens in a standalone desktop window (pywebview)
```

Two tabs, no surprises:

| Local models | Archives |
|:---:|:---:|
| ![Local models tab](https://raw.githubusercontent.com/hherb/ollamazip/main/assets/ollamazip_main.png) | ![Archives tab](https://raw.githubusercontent.com/hherb/ollamazip/main/assets/ollamazip_archives.png) |
| Browse, pack, rename, delete | Inspect, unpack, move, delete |

The **Local models** tab lists everything Ollama has pulled, with sizes, and lets you pack a model into an archive, rename it, or delete it (with optional pruning of orphaned blobs — handy after experimenting with lots of variants). The **Archives** tab points at any folder of `.ollamazip` files, lets you peek at the metadata, unpack, move, or delete them. It's the same set of operations as the CLI; just easier when you're triaging a dozen archives at once.

Because the GUI calls into the same `core.py` functions as the CLI, the two stay in lockstep — no second implementation drifting out of sync.

## How it actually works

The mechanism is short enough to summarize:

1. **Pack** reads the manifest at `manifests/<registry>/<namespace>/<model>/<tag>`, parses the digests it references (the config blob plus all layer blobs), deduplicates them, and streams everything into a tar. A `metadata.json` header records the original ref, blob count, uncompressed size, and source platform — useful for the inspect command.
2. **Unpack** reads the tar back, looks at each blob filename, and **skips any blob that already exists** in the local store with the same size. Because Ollama's blobs are content-addressed, this gives you free deduplication: if you've already got `llama3:8b-instruct` and you unpack a related quantization, the shared layers transfer once.
3. **Verify** (`--verify` on unpack) hashes every extracted blob and compares against the digest encoded in its filename. If anything is corrupt, you find out before Ollama does.
4. **Rename** (`--name newmodel:newtag` on unpack) imports the archive under a different ref. Since Ollama's manifest path *is* the model name, this is just writing the manifest bytes to a different directory. Blobs stay shared.

Compression is autoselected: zstandard if available, gzip otherwise. Streaming tar means a 30 GB model doesn't materialize in RAM — it flows blob-by-blob through the compressor and out to disk. The whole thing is built on Python's standard library; `zstandard` is the only optional dependency. There's no Ollama daemon API anywhere in the code — it works entirely against the filesystem layout, and respects `OLLAMA_MODELS` if you've relocated your store.

The full pack/unpack core is around 300 lines of straightforward code; small enough to read in one sitting.

## Three use cases this was built for

**Multiple machines at home.** This was the original itch. I have a desktop, a laptop, and a small home server. Pulling a 30 GB model three times over a residential link is a waste of an afternoon. Now I pull once, pack once, and rsync the archive over the LAN. Unpacks are roughly disk-write-bound.

**Reclaiming disk space.** When my SSD fills up, I pack the models I'm not actively using onto an external drive and `ollama rm` the originals. When I want one back, I unpack from the external drive — which, thanks to dedup-on-unpack, only rewrites the blobs not already present. Models become more like documents and less like permanent installations.

**Archiving.** Tags on the registry get superseded, occasionally retired, and you can't always count on a specific quantization being available next year. If you've fine-tuned something, or have a working build of a particular quant you depend on, a single immutable file with a SHA256-verifiable payload is a real upgrade over "hope the registry still has it."

## What it deliberately doesn't do

It doesn't talk to the Ollama daemon. It doesn't push to remote registries. It doesn't repack across quantizations or convert formats. It's not a sync tool — there's no incremental "what's new on A vs B." Each of those is a reasonable separate project. The goal here was the boring 80% case: take *this* model on *this* machine, give me a file, let me put it back somewhere later.

## Try it

```bash
pip install ollamazip[zstd]
ollamazip pack <your-favourite-model>

# or, if you'd rather click than type:
pip install ollamazip[gui]
ollamazip gui
```

Source: github.com/hherb/ollamazip — AGPL-3.0. Issues, PRs, and "I wish it did X" reports all welcome.
