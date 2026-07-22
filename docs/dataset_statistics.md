# Reproducing A3-Synth dataset statistics

`scripts/analyze_a3_synth.py` computes the website-state coverage,
instruction-diversity, and frequent-functional-page statistics reported for
the A3-Synth training set. It requires Python 3.12 or newer and only uses the
standard library; installing the model-training dependencies is unnecessary.

## Input and command

Download `training/train.jsonl` from the public
[McGill-NLP/A3-Synth](https://huggingface.co/datasets/McGill-NLP/A3-Synth/tree/main/training)
dataset. The following command pins the dataset revision used for the reported
statistics:

```bash
mkdir -p data/A3-Synth/training
curl --fail --location \
  --output data/A3-Synth/training/train.jsonl \
  https://huggingface.co/datasets/McGill-NLP/A3-Synth/resolve/d30e81302a74df6ec18d6361aab22a48ba4f32e7/training/train.jsonl
echo 'dc1e105274a84b8336919aef0a680efe06852c5b00831b91933a705a06ceb487  data/A3-Synth/training/train.jsonl' \
  | sha256sum --check
```

The pinned file is 505,096,714 bytes. From the repository root, run:

```bash
python3.12 scripts/analyze_a3_synth.py data/A3-Synth/training/train.jsonl
```

For stable machine-readable output:

```bash
python3.12 scripts/analyze_a3_synth.py \
  data/A3-Synth/training/train.jsonl \
  --format json > a3_synth_statistics.json
```

The JSONL data itself is not copied into this repository. Consult the dataset
card for its terms and distribution details before redistributing it.

## Methodology

- **Analysis unit.** Each JSONL row is one SFT training step. The released
  flattened rows do not include an explicit trajectory identifier, so the
  text between `## Goal:` and `# Observation`, with surrounding whitespace
  removed, is used as the instruction/trajectory key. Repeated steps for the
  same instruction are counted once in instruction statistics.
- **Host normalization.** The WebArena instance suffix `-xl-N` is removed from
  the network location. For example, `wa-shopping-xl-2.example.org` and
  `wa-shopping-xl-3.example.org` are treated as the same logical host. Hosts
  and URL schemes are lowercased; paths and queries retain their original
  case and encoding. URL fragments are discarded.
- **Unique paths.** A path state is the normalized scheme, host, and path.
  Query strings are discarded, but site identity is retained to prevent paths
  shared by different websites from colliding.
- **Unique URLs.** A URL state is the normalized scheme, host, path, and query
  string. This captures query-bearing states such as searches and filters.
  Steps whose active tab is `about:blank` remain in the step and instruction
  totals but cannot contribute a webpage state; the report counts them
  separately.
- **Distinct-n.** Instructions are lowercased and tokenized with
  `[a-z0-9']+`. For each `n`, the numerator is the number of unique n-grams
  across distinct instructions and the denominator is the total number of
  within-instruction n-grams. N-grams never cross instruction boundaries.
  This is the Distinct-n family introduced by
  [Li et al. (2016)](https://aclanthology.org/N16-1014/).
- **Functional pages.** The ranking counts normalized path states once per
  training step and excludes root paths (`/`), paths containing `Landing`, and
  the Shopping Admin dashboard (`/admin/admin/dashboard`, with an optional
  trailing slash). These are the dataset's entry or landing pages. Ties are
  broken lexicographically by normalized URL for deterministic output.

The script validates every nonempty row and exits with the source line number
if JSON, message content, goal text, or an active-tab URL other than the
recognized `about:blank` state is malformed.

## Expected headline output

For the pinned training file, the report contains 16,353 steps and 2,322
distinct instructions. Of those steps, 16,351 contain webpage URLs and two
contain `about:blank`. It reports 1,636 unique paths and 2,876 unique URLs.
Instruction lengths have mean 25.0, median 23, minimum 7, and maximum 80.

| Site | Unique URLs | Unique paths |
|---|---:|---:|
| Shopping | 711 | 231 |
| Map | 572 | 145 |
| GitLab | 541 | 369 |
| Wikipedia | 494 | 465 |
| Reddit | 387 | 255 |
| Shopping Admin | 171 | 171 |

| Metric | Unique | Total | Distinct-n |
|---|---:|---:|---:|
| Distinct-1 | 6,257 | 57,975 | 0.1079 |
| Distinct-2 | 22,496 | 55,653 | 0.4042 |
| Distinct-3 | 33,206 | 53,331 | 0.6226 |
| Distinct-4 | 37,467 | 51,009 | 0.7345 |

## Tests

The focused tests use a small synthetic JSONL fixture and do not require the
released dataset:

```bash
python3.12 -m unittest tests/test_analyze_a3_synth.py
```
