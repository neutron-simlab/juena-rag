# Ingesting a code repository

A worked example of the thing the package's README claims is easy: *"a new document type
is a new script in someone's own repository plus a schema subclass."* This is that script,
for source code, using the `CodeChunk` class juena-rag already ships.

It clones a git repository, splits every file into chunks, embeds them and stores the whole
thing as **one source** — so a search during an ingest sees the old revision or the new one,
never a half-built index.

Nothing here is imported by `juena_rag`. It sits outside `src/` deliberately: the package is
pydantic and nothing else, and this needs git, a YAML parser and a code chunker.

## Running it

```bash
docker compose up -d                    # from the repository root
uv sync --group use-cases

uv run --group use-cases python use-cases/code-repository/ingest.py jscatter --dry-run
uv run --group use-cases python use-cases/code-repository/ingest.py jscatter
```

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
uv run --group use-cases python use-cases/code-repository/search.py \
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
    result = store.search("your question", visibility=["public"], n=5)
```

`visibility` is required and has no default — a store that guesses which levels a caller
may see guesses generously exactly once.

## What it produced

jscatter at `0654b756`, on a laptop:

| | |
| --- | --- |
| Files → chunks | 170 → 3,293, across 165 distinct files |
| Kinds | 2,846 `code_block`, 447 `text` |
| With a `symbol` | 2,467 |
| Documentation chunks | 181 |
| Longest chunk | 1,471 tokens, against the model's 8,192 |
| Embedding | 115 s |
| On disk | 123 MB, of which the HNSW index is 2.9 MB |
| A search | 6.9 ms, using `chunks_embedding_bits_idx` |

The index being 2.9 MB against a 123 MB table is the binary-quantization decision working as
designed: one bit per dimension instead of 4,096 floats.

Retrieval, unscored but real — note the last one, asked in German against an
English-and-code corpus:

```
--- how do I fit a form factor to small angle scattering data?
  0.226  src/jscatter/examples/example_Sedimentation.py:37
  0.240  src/jscatter/doc/source/sas.rst:1 (doc)
  0.289  src/jscatter/smallanglescattering.py:1855  [transmissionCorrection]

--- calculate the scattering length density of a protein
  0.241  src/jscatter/examples/example_bio_proteinVolume.py:1
  0.244  src/jscatter/bio/mda.py:1902  [scatteringUniverse.__init__]
  0.274  src/jscatter/formel/physics.py:425  [scatteringLengthDensityCalc]

--- wie berechne ich den Strukturfaktor einer harten Kugel?
  0.218  src/jscatter/doc/source/structurefactor.rst:1 (doc)
  0.330  src/jscatter/structurefactor/fluid.py:1094  [PercusYevick]
```

## How it is put together

| File | |
| --- | --- |
| [`repository.py`](repository.py) | clone or refresh, resolve the revision, list the files worth indexing |
| [`chunking.py`](chunking.py) | one file → `list[CodeChunk]` |
| [`ingest.py`](ingest.py) | wires those to `Store.replace_source` |
| [`search.py`](search.py) | the other direction — `Store.search`, printed for a person |
| [`repositories.yaml`](repositories.yaml) | which repositories, and how to treat their files |

The splitting is carried over from the chatbot's indexer rather than invented: chonkie's
`CodeChunker` for Python and Markdown, langchain's `RecursiveCharacterTextSplitter` with
language-aware separators elsewhere. Both versions are **pinned exactly** in
`pyproject.toml`, and that is not caution — an unpinned chonkie resolved to a newer release
that split the same 170 files into 4,114 chunks instead of 3,293, silently.

Four things it does that the chatbot's version does not: a line range (derived from the
character offsets, because `formel.py:412-448` is what a citation shows a person), the
enclosing `def`/`class` from an `ast` walk, `chunk_kind` split between code and prose, and a
`docs_paths` that actually matches the repository — the chatbot's names `README.md` and
`docs/`, and jscatter has `README.rst` and `src/jscatter/doc/`, so every chunk in the live
index is marked `is_doc=False`.

## One source per repository

`source_id` is the repository id, `revision` is the git commit sha, and each file is a
chunk's `source_uri`. That is what `Store` already assumes — `delete_chunks(source_id,
uri=…)` means *"remove one file's chunks from a source"*.

**Any change re-ingests the whole repository — a choice, not a constraint.** `Store` supports
the other way: `delete_chunks(source_id, uri=…)` and `upsert_chunks` exist precisely for
*"one file of many changed"*, and search filters on visibility, source, category and kind but
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
uv run pytest use-cases/
```

No network, no git, no database — they build directory trees in a temporary folder and chunk
strings. They skip cleanly if the `use-cases` group is not installed, the same way the
database tests skip without a database.
