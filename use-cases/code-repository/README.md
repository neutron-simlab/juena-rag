# Ingesting a code repository

How juena-rag's own corpus — eight scientific-software repositories — is indexed, with
`juena-rag ingest-repo`.

It clones a git repository, splits every file into chunks, embeds them and stores the whole
thing as **one source** — so a search during an ingest sees the old revision or the new one,
never a half-built index.

This started as a script in this folder. It moved into the package as `juena_rag.ingest.code`
when a second application, juena-lab, needed the same pipeline and was running this
folder's script from outside the package. What stays here is the config that belongs to
*this* corpus, [`repositories.yaml`](repositories.yaml), and `search.py`. Every caller keeps
its own config, and that config — not juena-rag — decides `chunk_size` and
`chunk_overlap`.

## Running it

```bash
docker compose up -d                    # from the repository root
uv sync --extra ingest
set -a; source .env; set +a             # export .env into this shell

uv run --extra ingest juena-rag ingest-repo jscatter \
    --config use-cases/code-repository/repositories.yaml --dry-run
uv run --extra ingest juena-rag ingest-repo jscatter \
    --config use-cases/code-repository/repositories.yaml
```

The command never reads a file for its settings, so a container passes the same values as
ordinary environment variables. `source` rather than `uv run --env-file .env`, because
`.env` builds its connection strings from references like `${POSTGRES_PORT:-5433}`: the
shell expands them, uv's loader passes them through as they are written.

`--dry-run` chunks and counts without touching the network or the database, which is the
right way to check a config change. The full run needs two things in `.env`:

- `BLABLADOR_API_KEY`
- `JUENA_RAG_DSN`, pointing at a database with the schema already applied at width 4096

**`JUENA_RAG_DSN` must not be the database the tests use.** The test fixture puts `public` on
its search path, so a corpus living there makes every `apply_ddl()` in the suite fail with
*"already exist(s)"*. See `.env.example`, which has the two-line setup.

| Flag | |
| --- | --- |
| `--dry-run` | chunk and count only |
| `--force` | re-ingest even when the stored revision already matches |
| `--limit N` | first N files only, for a cheap smoke test against the real model |
| `--no-sync` | use the clone as it stands, without fetching |
| `--cache-dir` | where clones live (default `data/repos`) |

Adding a repository is a block in [`repositories.yaml`](repositories.yaml), not new Python.

## Reading it back

```bash
uv run python use-cases/code-repository/search.py \
    "how do I fit a form factor to SANS data?"
```

```
  0.226  src/jscatter/examples/example_Sedimentation.py:37
  0.240  src/jscatter/doc/source/sas.rst:1  (doc)
  0.289  src/jscatter/smallanglescattering.py:1855  [transmissionCorrection]
```

The number is **cosine distance, so smaller is closer** — not a score, and not turned into
a percentage anywhere, because inverting it into a similarity would claim the vectors are
unit-length and only the embedding model knows that. Ask in any language the model knows;
the corpus being English and Python does not stop a German question from working.

| Flag | |
| --- | --- |
| `-n` | how many hits (default 5) |
| `-k` | how many candidates the index retrieves before filtering |
| `--source` | restrict to one source id; repeatable |
| `--code-only` | skip prose — documentation, README text |
| `--full` | print each chunk's text, not just where it is |

**Raise `-k` whenever you narrow the filter.** An HNSW index finds the nearest vectors
first and filters afterwards, so `--code-only` over a corpus that is a third prose can
return fewer than the `-n` you asked for, quietly.

In your own code it is three lines, and `search.py` is only a wrapper around them:

```python
with Store(dsn, embedder=BlabladorEmbedder.from_env()) as store:
    result = store.search("your question", n=5)
```

There is no permission argument, and no permission column. Every source in this corpus is
public open-source code, and a `visibility` field that never varies is not access control —
it is a name the next reader trusts by mistake. What it would take to bring back is recorded
in plan 02.

## What it produced

Eight repositories, at `chunk_size: 3000` — [measured, not
inherited](#how-big-should-a-chunk-be). `sampledb` is the ninth in the chatbot's config and
is deliberately not here.

| source | chunks | with a `symbol` | doc | prose | files |
| --- | ---: | ---: | ---: | ---: | ---: |
| datreat | 12,362 | 7,202 | 134 | 169 | 2,329 |
| vitess | 11,356 | 2,105 | 594 | 283 | 2,757 |
| silx | 4,624 | 2,094 | 523 | 356 | 1,006 |
| sasview | 3,111 | 1,170 | 103 | 385 | 752 |
| mdanse | 1,921 | 640 | 176 | 555 | 477 |
| sasdata | 1,872 | 201 | 24 | 427 | 213 |
| jscatter | 1,641 | 875 | 94 | 229 | 165 |
| sasmodels | 1,590 | 375 | 130 | 217 | 341 |
| **total** | **38,477** | **14,662** | **1,778** | **2,621** | **8,040** |

| | |
| --- | --- |
| Embedding, all eight | 41 minutes |
| Longest chunk | 3,000 tokens, against the model's 8,192 — nothing near it |
| On disk | 730 MB, of which the HNSW index is **27 MB** |
| A search | 4.3 ms, `Index Scan using chunks_embedding_bits_idx` |

The index being 27 MB against a 730 MB table is binary quantization working as designed:
one bit per dimension rather than 4,096 floats.

Retrieval, asked in German against an English-and-code corpus:

```
--- how do I simulate a neutron guide with a curved section?
  0.251  WWW/guide.html:75  (doc)
  0.252  FILES/TUTORIALS/RaytracingExample/Ex3h_Raytracing.gui:111  (doc)
  0.268  SRC/guide.c:1

--- read a NeXus HDF5 file into a data object
  0.239  sasdata/dataloader/readers/cansas_reader_HDF5.py:41  [Reader]
  0.274  doc/source/modules/io/nxdata.rst:2  (doc)

--- wie berechne ich den Strukturfaktor einer harten Kugel?
  0.218  src/jscatter/doc/source/structurefactor.rst:1  (doc)
  0.326  src/jscatter/structurefactor/fluid.py:1046  [PercusYevick]
```

The first two cross repositories on their own — a question about guides reaches vitess's C
source, its HTML documentation and a tutorial file; a question about file formats reaches
sasdata and silx. Nothing tells the search which repository to look in.

## How big should a chunk be?

`chunk_size` turned out to matter more than the chunker does, so it was measured rather
than guessed. Two corpora, 25 questions with the expected answer written down first, and
the rank of the first correct hit recorded:

| `chunk_size` | code, top-3 | code MRR | manual, right content | manual MRR |
| ---: | ---: | ---: | ---: | ---: |
| 1,500 — the chatbot's | 7/9 | 0.731 | 13/16 | 0.705 |
| **3,000 — this pipeline** | **8/9** | 0.759 | **15/16** | **0.865** |
| 6,000 | 7/9 | **0.828** | 15/16 | 0.812 |

**Bigger chunks retrieve better, up to a point, and cost less.** At 1,500 the median Fortran
subroutine does not fit in one chunk, so functions get cut mid-body and the docstring that
carries the meaning is separated from the code it describes. 3,000 halves the chunk count
*and* improves retrieval — the one place in this project where cheaper and better agreed.

**What it costs is citation precision, and that part is not fixable.** A chunk covering four
sections cannot honestly be labelled with one of them, so its `heading_path` collapses to
their shared parent — `Fitting > Model fitting` where a smaller chunk said `Fitting > Model
fitting > fit Fitting the model parameters to data`. Both are true; the second is more
useful. Past about 6,000 the retrieval gain stops while this keeps getting worse, which is
what put the setting at 3,000.

The 25 questions were written by hand, with their expected answers chosen by hand. That
makes this a sanity check, not a benchmark.

## How it is put together

| File | |
| --- | --- |
| `juena_rag.ingest.code.repository` | clone or refresh, resolve the revision, list the files worth indexing |
| `juena_rag.ingest.code.chunking` | one file → `list[CodeChunk]` |
| `juena_rag.parsers` | what each file declares, and on which lines |
| `juena_rag.ingest.code.pipeline` | wires those to `Store.replace_source`; `juena-rag ingest-repo` calls it |
| [`search.py`](search.py) | the other direction — `Store.search`, printed for a person |
| [`repositories.yaml`](repositories.yaml) | which repositories, and how to treat their files |

The splitting is carried over from the chatbot's indexer rather than invented: chonkie's
`CodeChunker` for Python and Markdown, langchain's `RecursiveCharacterTextSplitter` with
language-aware separators elsewhere. Both versions are **pinned exactly** in
`pyproject.toml`, and that is not caution — an unpinned chonkie resolved to a newer release
that split the same 170 files into 4,114 chunks instead of 3,293, silently.

Four things it does that the chatbot's version does not: a line range (derived from the
character offsets, because `formel.py:412-448` is what a citation shows a person), the
enclosing definition, `chunk_kind` split between code and prose, and a `docs_paths` that
actually matches the repository — the chatbot's names `README.md` and `docs/`, and jscatter
has `README.rst` and `src/jscatter/doc/`, so every chunk in the live index is marked
`is_doc=False`.

**Symbols come from `juena_rag.parsers`, not from here.** Python uses the standard library's
`ast`; Fortran, C, C++ and LaTeX use tree-sitter grammars that `chonkie[code]` already
carried, so supporting them cost a set entry rather than a dependency. A language with no
parser gets an empty list, never an error — which is what `.yaml`, `.sh`, `.ui` and vitess's
`.gui` files get, about 19% of the corpus.

**C is parsed without a preprocessor, on purpose.** An unexpanded macro can misparse as a
function definition, and the grammar then takes the next keyword as its name — which is how
`main.while` got into a citation before it was caught. Neither C nor C++ nests a function
inside a function, so the walk refuses to look inside one, and the artifacts go away without
a macro expander or a list of banned names.

**A chunk is named for what contains all of it.** A chunk running from the end of one
subroutine into the next is inside neither, so it takes their shared parent, or nothing.
Naming the one it happened to start in is a citation pointing at a line that does not
contain what was quoted — which the pipeline did until it was measured, on 392 of jscatter's
chunks.

## One source per repository

`source_id` is the repository id, `revision` is the git commit sha, and each file is a
chunk's `source_uri`. That is what `Store` already assumes — `delete_chunks(source_id,
uri=…)` means *"remove one file's chunks from a source"*.

**Any change re-ingests the whole repository — a choice, not a constraint.** `Store` supports
the other way: `delete_chunks(source_id, uri=…)` and `upsert_chunks` exist precisely for
*"one file of many changed"*, and search filters on source, category and kind but
**never on revision**, so chunks of mixed revisions would all stay findable.

What made it not worth it here is that `chunk_id` hashes the revision, so the commit sha is
inside every chunk's name. Re-index one file at a new sha and the other 164 files keep the
old one, which costs three things: `sources.active_revision` stops describing what is stored
and nothing can detect the drift; `chunk_index` stops meaning "position within the source";
and two runs at the same sha collide on the primary key, because both batches number from
zero. Answering "which file is at which revision" needs a manifest — the machinery this port
deleted.

The saving worth having is not the rewrite but the **re-embedding**: rewriting 3,293 rows
locally is milliseconds, embedding them is 115 seconds of API calls. `content_hash` already
identifies text the store has seen before, so a one-file commit could reuse ~3,250 vectors in
SQL and embed only the rest — one revision, still atomic, and only the new vectors ever
reaching Python. That is W1 in the execution plan, and it is the open question here.

## What it cost the package

Two defects, both found on first real use rather than by review:

- **`replace_source` could not run twice at the same revision.** It inserted before deleting
  and deleted only *other* revisions, so a re-run collided on the primary key. Fixed: the
  delete now runs first and takes every chunk of the source, inside the same transaction, so
  the atomicity guarantee is unchanged. Two tests in `tests/test_store.py` cover it.
- **`replace_source` holds every vector in memory at once.** It embeds before opening its
  transaction, which is right — a multi-minute network call has no business inside one — but
  4,096 floats cost ~131 kB as a Python list against 16 kB in the database. jscatter's 3,293
  chunks are ~430 MB. **Not fixed**, and it is the thing to fix before ingesting anything
  much larger.

## Tests

```bash
uv run --extra ingest pytest tests/test_chunking.py tests/test_repository.py
```

No network, no git, no database — they build directory trees in a temporary folder and chunk
strings. They skip cleanly if the `[ingest]` extra is not installed, the same way the
database tests skip without a database.
