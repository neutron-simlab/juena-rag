# juena-rag

Chunk schema, storage and retrieval for the JCNS knowledge base.

The package owns one thing: **a chunk, once it has been cleaned, gets stored,
versioned, searched and cited correctly.** It does *not* convert formats, and it does not
decide where a chunk ends — that stays with whoever knows the document.

**One deliberate exception:** `juena_rag.parsers` reads a source file's *structure* — which
function, subroutine or section a given line sits inside. That answer is identical for every
pipeline that will ever read Fortran, and getting it wrong is visible in every citation the
corpus produces, so it belongs here rather than in each caller.

> The schema, the store, the read-only HTTP service and its client are built and
> carry a real corpus. What the contract below still describes ahead of the code is
> the deployment; see [Status](#status).

## The boundary

| Stage | Who owns it |
| --- | --- |
| Fetch the file (git, share, wiki, upload) | the caller's connector |
| Parse it — PDF, DOCX, LaTeX, HTML → text | [docling](https://docling-project.github.io/docling/), a Markdown parser, or a hand-written pipeline |
| Split into chunks and clean them — including tables into rows | the caller |
| **Read the structure — which function or section a line is in** | **juena-rag** |
| **Validate against the schema** | **juena-rag** |
| **Embed, store, version, replace** | **juena-rag** |
| **Search, filter, cite** | **juena-rag** |

Everything above the line is plural and messy by nature — a neutron-scattering manual
chunks differently from a Fortran file. Everything below it must be *exactly one*
implementation, or the corpus becomes a pile of incompatible islands.

So a new document type is a new script in someone's own repository plus a schema
subclass. It is not a change to this package — with the honest caveat that indexing a
Fortran codebase *did* change it once, to add that language's parser, and the next
unfamiliar language will too.

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
run Kubernetes, so the service is one Deployment, one Service and one internal Route —
reached over the institute VPN, never the public internet — described
once and sized differently per environment. pgvector is a StatefulSet with its own
storage claim, deliberately its own instance rather than a schema inside an application's
database.

**Applications search over HTTP; they do not open the database.** Talking to pgvector
directly from an application would work — it is a real database server and concurrent
access is safe. The objection is architectural. Direct access puts schema migrations in
lockstep across every consumer, hands each of them the embedding API key, and replaces the
API contract with a table layout that nobody can change afterwards. The
`[store]` extra exists for this service and for ingestion pipelines, not for consumers.

### Running the service

```bash
set -a; source .env; set +a           # JUENA_RAG_DSN, and a key for [embed]
uv run juena-rag serve --port 8000    # --host 0.0.0.0 in a container
```

It opens one `Store` at startup and refuses to start unless both widths agree: the
model has to return vectors of the configured size, and the `embedding` column has
to be that wide. Five endpoints, each of them one call to a `Store` method:

| | |
| --- | --- |
| `POST /v1/search` | the question and its filters, in the body |
| `GET /v1/sources` | the catalogue |
| `GET /v1/sources/{id}/structure?revision=…` | one document, and **only** at a revision you name |
| `GET /v1/status` | what is stored — scans the table, so not a probe |
| `GET /v1/health` | can this pod reach the database |

```bash
curl -s localhost:8000/v1/search -H 'content-type: application/json' \
    -d '{"query": "wie berechne ich den Strukturfaktor einer harten Kugel?", "n": 3}'
```

**Consumers use the client, not the endpoints.** It ships in this package so
nobody hand-writes a request body against the schema and gets it subtly wrong:

```python
from juena_rag.store.pgvector import Store  # [store] -- reads and writes
from juena_rag.store.http import HttpStore  # [client] -- the read half only

store = Store(dsn, embedder=BlabladorEmbedder.from_env())
store = HttpStore("https://juena-rag.jcns.internal")

result = store.search("how do I fit a form factor to SANS data?", n=5)
```

Same five read methods, same argument names, same objects back — checked by a
shared test suite that runs every read against both. `HttpStore` has none of the
five writing methods, and the suite asserts their absence too.

**Reads only, and pointers only.** The five writing methods have no endpoint at
all — ingestion opens the database directly, as a job — and a hit carries its own
text plus where that text came from, never the original file. `git show
<sha>:<path>` resolves the pointer, and the sha the hit already carries is what
keeps the two in step.

Two more settings, both optional: `JUENA_RAG_POOL_MAX_SIZE` (4) is how many
database connections there are, and `JUENA_RAG_MAX_CONCURRENCY` (40) is how many
requests may be in flight. They are deliberately not equal — a search embeds the
question *before* it borrows a connection, so the round trip to Blablador occupies
a thread and no connection.

## Getting started

Requires [uv](https://docs.astral.sh/uv/).

```bash
git clone <this repo> && cd juena-rag
uv sync                     # creates .venv, installs deps and the dev group
cp .env.example .env        # then set POSTGRES_PASSWORD, and a key if you want [embed]
docker compose up -d        # PostgreSQL 18 with pgvector, on port 5433
uv run pytest               # 256 tests, 1 skip
```

Two things are optional, and each shows up as skips rather than failures:

| | Passes | Skips |
| --- | ---: | ---: |
| `uv run pytest` | 256 | 1 |
| `uv run pytest`, container stopped | 181 | 76 |
| `uv run --group use-cases pytest` | **272** | 0 |
| `uv run --group use-cases pytest`, container stopped | 197 | 75 |

An unreachable `JUENA_RAG_TEST_DSN` is treated the same as an unset one, so the database
tests skip on a plane. Each of them works in its own throwaway schema and drops it
afterwards. The single skip in the first row is the chunking suite in `use-cases/`, which
needs the optional group that carries a code chunker.

The container is a development database and nothing else; it is not the shape anything
deploys in. See [`deploy/kube/`](deploy/kube/) for that.

Add a dependency with `uv add <package>`; add a dev-only one with `uv add --dev`. Both
update `pyproject.toml` and `uv.lock` together — commit the lockfile.

Extras, so a pipeline author installs only what they need:

| Install | Gets you |
| --- | --- |
| `pip install juena-rag` | the schema alone — enough to write an ingestion pipeline |
| `pip install juena-rag[store]` | pgvector access (psycopg, pgvector) |
| `pip install juena-rag[embed]` | the Blablador embedding client (openai, tokenizers) |
| `pip install juena-rag[client]` | `HttpStore` — search the service over HTTP (httpx) |
| `pip install juena-rag[serve]` | the HTTP service — FastAPI and uvicorn, **and the two extras above**, since it is the thing that reaches the database and embeds the question |

`[embed]` is separate from `[store]` on purpose: a store handed an embedder needs no
network stack of its own. The `Embedder` protocol and its checks are in the base
install; only the two things that reach outside — the HTTP client and the model's
vocabulary — sit behind the extra.

`BlabladorEmbedder.from_env()` reads these; only the first has no default:

| Variable | Default |
| --- | --- |
| `BLABLADOR_API_KEY` | — |
| `BLABLADOR_BASE_URL` | `https://api.blablador.fz-juelich.de/v1` |
| `BLABLADOR_EMBEDDING_MODEL` | `alias-qwen3-8b-embeddings` |
| `BLABLADOR_EMBEDDING_DIMENSION` | `4096` |
| `BLABLADOR_EMBEDDING_MAX_INPUT_TOKENS` | `8192` |

**Text arrives ready to embed.** The embedder sends what it is given, whole — there is no
chunker in this package and there is not going to be one, because where a chunk should end
depends on the format it came from and the format is exactly what this package does not
know. Past `max_input_tokens` the model truncates and returns a normal-looking vector, so
staying under it matters; it is the chunker's job, and `ModelTokenCounter` is here so the
job is measurable rather than guessed at:

```python
embedder = BlabladorEmbedder.from_env()
tokens = ModelTokenCounter()  # or ModelTokenCounter("path/to/tokenizer.json")

counts = tokens.count_many([chunk.text for chunk in chunks])
too_big = [c for c, n in zip(chunks, counts) if n > embedder.metadata.max_input_tokens]
```

Pass a local `tokenizer.json` in a container image; left unset the vocabulary is fetched
from Hugging Face on first use and cached.

## Layout

```
src/juena_rag/
├── schema/        the chunk schema -- the part other people code against
├── parsers/       what a source file declares, per language -- one module each,
│                  all returning the same shape
├── embedding/     turning text into vectors; the parts that need a network
│                  (the client, the vocabulary) are behind [embed]
├── store/         base.py is the read contract and the errors, and needs no driver;
│                  pgvector.py is [store], http.py is [client], ddl.sql is the table
└── service/       the HTTP service -- five endpoints and `juena-rag serve`, [serve]
deploy/kube/       the deployed database: a CloudNativePG cluster,
                   one base and a `vm` overlay for the single node
docker-compose.yml the local one, for tests
use-cases/         worked ingestion pipelines built *on* the package, not in it --
                   outside src/ so the base install stays pydantic-only
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
| Schema | built — models, registry, validation rules, 76 tests |
| Parsers | built — Python via `ast`; Fortran, C, C++ and LaTeX via tree-sitter |
| Embedder | built — Blablador client, token counting, dimension pinned at 4096 |
| pgvector store | built — `Store` with a two-stage search and an atomic source replace |
| Local database | built — `docker compose up -d` |
| Deployed database | built — CloudNativePG, verified on a real cluster; the OpenShift arbitrary-UID question is still open |
| HTTP service | built — five read endpoints under `/v1`, `juena-rag serve`, no write side |
| The client | built — `HttpStore`, the same five methods over HTTP; a shared suite runs every read against both |
| Ingestion pipelines | out of scope by design — but [`use-cases/code-repository/`](use-cases/code-repository/) is one worked example, and eight repositories and 38,477 chunks are the corpus it now holds |
