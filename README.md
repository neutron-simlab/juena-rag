# juena-rag

Chunk schema, storage and retrieval for the JCNS knowledge base.

The package owns one thing: **a chunk, once it has been cleaned, gets stored,
versioned, searched and cited correctly.** It deliberately does *not* parse or chunk
documents — that stays with whoever knows the format.

> Early scaffolding. The contract below is the agreed direction; almost none of it is
> implemented yet.

## The boundary

| Stage | Who owns it |
| --- | --- |
| Fetch the file (git, share, wiki, upload) | the caller's connector |
| Parse it — PDF, DOCX, LaTeX, HTML → text | [docling](https://docling-project.github.io/docling/), a Markdown parser, or a hand-written pipeline |
| Split into chunks and clean them — including tables into rows | the caller |
| **Validate against the schema** | **juena-rag** |
| **Embed, store, version, replace** | **juena-rag** |
| **Search, filter, cite** | **juena-rag** |

Everything above the line is plural and messy by nature — a neutron-scattering manual
chunks differently from a Fortran file. Everything below it must be *exactly one*
implementation, or the corpus becomes a pile of incompatible islands.

So a new document type is a new script in someone's own repository plus a schema
subclass. It is not a change to this package.

## Design in brief

**The schema has three layers.** A core that every chunk carries and that becomes real
database columns; a per-category extension added by subclassing and kept in one JSON
column; and a batch object that crosses the API. One rule keeps it from rotting: *if you
want to filter on it, it belongs in the core.*

The core is short on purpose, and shorter still to write: you supply a source id, a
category, a revision and a list of texts. Identifiers and hashes are derived, the
source's answers are copied onto its chunks for you, and anything true of only some
categories — line numbers, page provenance, a section heading — lives in the extension,
so nobody indexing PDFs is ever asked for a line number.

**Clean Markdown is a first-class input, not a fallback.** Docling is the default answer
for PDFs and Office formats, but if your documentation is already Markdown you need
neither Docling nor a converter: the `.md` file is its own artifact, identified by the
sha256 of its bytes, and chunks anchor by line range where a PDF would anchor by page and
bounding box. Heading levels give the same section context Docling extracts. The only
thing you give up is page-region highlighting.

**Tables are worth splitting by row.** A sixty-row parameter table stored as one chunk is
one embedding standing for sixty separate facts, and a question about one of them has to
outrank the whole corpus while the other fifty-nine dilute it. Split it, render each row
with its column names — `"Parameter: min. y | Unit: cm | Option: -y"` — and mark the chunk
`chunk_kind="table_row"`. The splitting is yours to do; what the package supplies is the
agreed shape for the result, so every pipeline describes a row the same way.

**One table, not one collection per category.** `category` is an indexed column, so
reclassifying a source is an `UPDATE` rather than a re-index. A physical split is
reserved for the one case that forces it — a different embedding model.

**pgvector is the store.** It reuses a database already required on both deployment
targets, its filtering is plain SQL, and it turns "rebuild without downtime" into a
single transaction: insert the new revision beside the old, move the source's pointer,
delete what the pointer no longer names.

## Where it runs

Two things ship from this repository, and it matters which one a caller uses.

| | What it is | Who uses it |
| --- | --- | --- |
| **The package** | `pip install juena-rag` — the schema, and optionally the store | ingestion pipelines, and anything validating chunks |
| **The service** | a container running `juena-rag serve` beside a pgvector instance | applications that search the corpus |

Both deployment targets — a single-node cluster on a VM, and the institute's OpenShift —
run Kubernetes, so the service is one Deployment, one Service and one Route, described
once and sized differently per environment. pgvector is a StatefulSet with its own
storage claim, deliberately its own instance rather than a schema inside an application's
database.

**Applications search over HTTP; they do not open the database.** Talking to pgvector
directly from an application would work — it is a real database server and concurrent
access is safe. The objection is architectural. Direct access puts schema migrations in
lockstep across every consumer, duplicates permission filtering into each of them, and
replaces the API contract with a table layout that nobody can change afterwards. The
`[store]` extra exists for this service and for ingestion pipelines, not for consumers.

## Getting started

Requires [uv](https://docs.astral.sh/uv/).

```bash
git clone <this repo> && cd juena-rag
uv sync                 # creates .venv, installs deps and the dev group
uv run pytest           # once there are tests
```

Add a dependency with `uv add <package>`; add a dev-only one with `uv add --dev`. Both
update `pyproject.toml` and `uv.lock` together — commit the lockfile.

Extras, so a pipeline author installs only what they need:

| Install | Gets you |
| --- | --- |
| `pip install juena-rag` | the schema alone — enough to write an ingestion pipeline |
| `pip install juena-rag[store]` | pgvector access (psycopg, pgvector) |
| `pip install juena-rag[serve]` | the HTTP service (FastAPI, uvicorn) |

## Layout

```
src/juena_rag/
├── schema/        the chunk schema -- the part other people code against
└── store/         storage backends, and the SQL schema in ddl.sql
tests/
```

## Working on this

**The import direction is the rule that matters.** `juena_rag` must never import
`juena` (the chatbot). The chatbot may depend on this package; never the reverse. Getting
that backwards is the one mistake that is expensive to undo.

Two things to know before touching the schema:

- **It is the expensive thing** — because other people's pipelines are written against
  it, not because of the database. Adding a category is free forever. Adding a required
  core field breaks every caller that does not set it. The storage cost is mild by
  comparison: promoting a field to a column is a backfill, while a genuine re-index
  (re-embedding everything) is forced only by a change to chunk text, chunk boundaries
  or the embedding model.
- **Filtered vector search silently loses recall.** An HNSW index finds the nearest
  vectors first and applies the filter afterwards, so asking for 10 results with a
  category filter can quietly return 2. `SET hnsw.iterative_scan = strict_order` fixes
  it, and it has to be on from the first filtered query.

## Status

| | |
| --- | --- |
| Schema | drafted, under review |
| pgvector store | not started — `store/ddl.sql` is the intended table |
| HTTP service | not started |
| Ingestion pipelines | out of scope by design; they live in their own repositories |
