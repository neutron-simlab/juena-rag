# JüNA RAG

JüNA RAG makes a project's knowledge searchable. It stores passages from documents
and code, finds material relevant to a question, and returns references to the
sources. An assistant can use these results to answer questions about your
project's own material.

RAG means **retrieval-augmented generation**: finding relevant information before
asking a language model to write an answer. This package handles the search and
source information. The application decides how to present or use the results.

## How the JüNA projects fit together

| Project | What it provides | Start here when you want to… |
| --- | --- | --- |
| **juena-rag** | Searchable knowledge with source references | Add document or code search to a project |
| [juena-core](https://github.com/neutron-simlab/juena-core) | Shared building blocks for conversations, agents, and files | Build your own assistant application |
| [juena-chatbot](https://github.com/neutron-simlab/juena-chatbot) | A complete research assistant using core and RAG | Run JüNA or adapt an existing chat application |

You can use juena-rag with any Python application; it does not require
juena-core or juena-chatbot. If you are running the chatbot, its
[setup guide](https://github.com/neutron-simlab/juena-chatbot/blob/main/deploy/LOCAL-DEVELOPMENT.md)
already includes the search service and database.

## What it provides

- Search by meaning, with filters for particular sources or document categories.
- Results that retain the source, revision, and available location information,
  such as file paths, line numbers, or pages.
- A common way to describe code, papers, manuals, reports, datasets,
  presentations, wiki content, and books, with room for your own categories.
- A ready-to-use importer for Git repositories, including code and documentation.
- A search service that multiple applications can share, plus a small Python client.
- Optional reranking: a second model reorders the search results for a question.

The search database uses PostgreSQL with pgvector. An embedding model turns text
into numerical representations so related passages can be found. A Blablador
connection is included; the Python interfaces also allow other embedding providers.

## Choose your starting point

**Your group already has a search service:** use the Python client below. Your
application needs the service address, without database credentials or an
embedding key.

**You want to build a knowledge collection:** follow the local setup below,
then import a repository or prepare your own document pipeline.

**You only need the document data types:** use the base package to describe and
validate your material without running a database.

### Search an existing service

In your application's uv project, install the client:

```bash
uv add "juena-rag[client] @ git+https://github.com/neutron-simlab/juena-rag.git"
```

Keep `uv.lock` in version control so collaborators use the same resolved version.
Replace the address below with your service's address:

```python
from juena_rag.store.http import HttpStore

with HttpStore("http://127.0.0.1:8000") as knowledge:
    result = knowledge.search("How do I fit a sphere model to scattering data?", n=3)
    for hit in result.hits:
        source = result.source_of(hit)
        print(source.title or source.source_id)
        print(hit.text)
```

The result contains passages and source information. Writing an answer from those
passages is the calling application's job.

## Run a local search service

You need **Python 3.11 or newer**, **uv** for managing Python dependencies,
**Docker with Compose** for the database, and a **Blablador API key**.
The following setup creates a new local knowledge database.

### 1. Install and configure

```bash
git clone https://github.com/neutron-simlab/juena-rag.git
cd juena-rag
uv sync --extra serve --extra ingest
cp .env.example .env
```

Edit `.env`: set `BLABLADOR_API_KEY` and choose `POSTGRES_PASSWORD`. The supplied
settings use separate databases for your knowledge collection and the tests.
Then load the settings into your terminal:

```bash
set -a
source .env
set +a
```

Repeat that loading step in each new terminal. The service and importer read
environment variables; they do not load `.env` themselves.

### 2. Prepare the database once

Using the default database user from `.env.example`:

```bash
docker compose up -d --wait
docker compose exec pgvector createdb -U postgres juena_rag_corpus
```

Create the tables in that new database:

```bash
uv run --extra serve --extra ingest python - <<'PY'
import os
import psycopg
from juena_rag.store import apply_ddl

with psycopg.connect(os.environ["JUENA_RAG_DSN"]) as connection:
    apply_ddl(
        connection,
        dimension=int(os.environ["BLABLADOR_EMBEDDING_DIMENSION"]),
        create_extension=True,
    )
PY
```

This is a first-time setup, not an upgrade command. Existing tables are refused.
The database's embedding dimension must match the model used to index and search.

### 3. Add some knowledge

The example configuration indexes jscatter, a scientific Python repository:

```bash
uv run --extra serve --extra ingest juena-rag ingest-repo jscatter \
    --config use-cases/code-repository/repositories.yaml
```

This downloads the repository, splits its files into searchable passages, and
sends those passages to the embedding service. Add `--dry-run` to preview the
splitting without embedding or writing to the database; it can still download
the repository. See the [repository import guide](use-cases/code-repository/README.md)
for configuration and further options.

### 4. Start searching

```bash
uv run --extra serve --extra ingest juena-rag serve --port 8000
```

Keep this terminal open and use the client example above from your application.
The service checks its database and embedding configuration at startup. Its
health endpoint is `http://127.0.0.1:8000/v1/health`.

Stop the service with Ctrl+C. `docker compose down` stops the database while
keeping the stored collection.

## Bring your project's material

For **code repositories**, keep a repository configuration in your own project
and pass it to `ingest-repo`. The [example configuration](use-cases/code-repository/repositories.yaml)
shows how to choose files and passage sizes. Updating a repository replaces its
stored revision, so searches see a complete version of the source.

For **documents**, your project handles reading the original files and splitting
them into useful passages. There is no general PDF or Office-document importer
in this package. A document pipeline prepares:

| Part | What it describes |
| --- | --- |
| [SourceMetadata](src/juena_rag/schema/models.py) | The document's identity, revision, title, authors, and original file when available |
| [A chunk](src/juena_rag/schema/registry.py) | One searchable passage and its source location or category-specific details |
| [ChunkBatch](src/juena_rag/schema/boundary.py) | The document and passages to store together |

Use the existing categories or register your own, such as an instrument
procedure. Keep that project's converter and category extensions alongside its
application. The shared package handles validation, storage, and retrieval.

## Use only the parts you need

These extras select which components are installed. For example, replace
`[client]` in the installation command with `[ingest]` for a repository importer.

| Package | Purpose |
| --- | --- |
| `juena-rag` | Document and chunk data types, validation, and source-code structure parsers |
| `juena-rag[client]` | Search an existing service from Python |
| `juena-rag[store]` | Read and write a PostgreSQL/pgvector collection |
| `juena-rag[embed]` | Use the included embedding and reranking clients |
| `juena-rag[serve]` | Run the search service, including its database and model clients |
| `juena-rag[ingest]` | Import Git repositories, including embedding and storage |

Applications normally use `HttpStore` to search. Importers use the database
store to add or update material. The HTTP service exposes only reads and does
not provide user sign-in or per-source permissions; run it on a trusted network
or behind access controls provided by your deployment.

The [local Compose file](docker-compose.yml) runs the database only. Projects
choose how to host the service; the chatbot provides a bundled deployment, and
[Kubernetes database manifests](deploy/kube/) are also available here.

## Contributing

Keep project-specific source lists and document conversion in the project that
uses this package. Changes here should help other consumers too. When extending
the shared data model, consider existing importers and stored collections.

Run the tests from this repository:

```bash
uv run --extra serve --extra ingest pytest -q
```

Database tests use `JUENA_RAG_TEST_DSN`; keep it separate from the collection in
`JUENA_RAG_DSN`. Tests that require a database skip when it is unavailable. The
live embedding test needs `BLABLADOR_API_KEY`; leave it empty for offline work.

## License

[MIT](LICENSE)
