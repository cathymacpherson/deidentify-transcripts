# This project

Everything here is specific to the dataset in `data/`. Nothing in this directory is imported by
`src/`, and nothing in the general documentation depends on it — the tool works without this
directory present.

- [DATASET_NOTES.md](DATASET_NOTES.md) — corpus measurements, label conventions found in this
  data, evaluation split design, and the decisions that follow from them.

The general, reusable side of the repository lives outside this directory:

| | |
|---|---|
| [README.md](../README.md) | Install, server setup, and the commands |
| [docs/FORMATS.md](../docs/FORMATS.md) | Accepted transcript formats |
| [docs/SPEAKER_LABELLING.md](../docs/SPEAKER_LABELLING.md) | Speaker-labelling design and evaluation methodology |
| [docs/LOCAL_OLLAMA.md](../docs/LOCAL_OLLAMA.md) | Local-only fallback |

## What belongs where

Something belongs in `project/` if it would be wrong, misleading, or meaningless for a different
dataset: measured statistics, the meaning of a label this corpus happens to use, the shape of its
participant graph, decisions justified by a rate observed here.

Something belongs in the general documentation if it holds regardless of the data: how the pipeline
works, what to measure and why, how to build leakage-free splits, what the commands do.

The institutional inference server is **general**, not project-specific — it is shared
infrastructure available to any project using this repository, and it is documented in the main
[README](../README.md).

Data files themselves — transcripts, identity mappings, manifests — are gitignored and never
committed, wherever they sit. See [CLAUDE.md](../CLAUDE.md).
