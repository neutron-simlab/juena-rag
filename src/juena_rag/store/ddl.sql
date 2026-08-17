-- The two tables the package is built around, and their indexes.
--
-- `<dim>` is not SQL. `juena_rag.store.schema.render_ddl()` replaces it with the
-- embedding width before this text reaches the database, because pgvector needs
-- that number fixed when the column is created -- so it cannot be a query
-- parameter the way an ordinary value would be.
--
-- Deliberately `CREATE TABLE` rather than `CREATE TABLE IF NOT EXISTS`: applying
-- this twice is a mistake and should say so. `apply_ddl()` sends the whole file
-- inside one transaction, so a failure anywhere leaves the database untouched
-- rather than half migrated.

CREATE TABLE sources (
    source_id       text PRIMARY KEY,
    category        text        NOT NULL,
    title           text,
    visibility      text        NOT NULL DEFAULT 'internal',
    active_revision text,

    -- the artifact, for documents that have one; all NULL for code sources
    artifact_uri    text,
    artifact_format text,
    artifact_hash   text,
    parser          text,

    updated_at      timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE chunks (
    chunk_id        text PRIMARY KEY,
    source_id       text NOT NULL REFERENCES sources(source_id) ON DELETE CASCADE,
    source_revision text NOT NULL,
    chunk_index     int  NOT NULL,

    category        text NOT NULL,          -- filtered
    visibility      text NOT NULL,
    language        text,
    chunk_kind      text NOT NULL DEFAULT 'text',

    source_uri      text,                   -- provenance

    text            text NOT NULL,          -- content
    content_hash    text NOT NULL,

    extra           jsonb NOT NULL DEFAULT '{}',   -- the extension

    embedding       vector(<dim>) NOT NULL,
    embedding_model text NOT NULL,
    schema_version  int  NOT NULL,
    ingested_at     timestamptz NOT NULL DEFAULT now()
);

-- One bit per dimension, not the vector: 4096 floats are 16 kB and over every
-- index size cap, while their signs are 512 bytes. The column keeps the full
-- float32 vector and re-ranks the candidates exactly.
--
-- A query's ORDER BY must apply the same function and the same cast to the same
-- column. Parentheses and `CAST(x AS y)` spelling are free -- Postgres compares
-- the parsed expression, not the text -- but drop the `::bit(<dim>)` and it plans
-- a sequential scan over every row without saying so.
CREATE INDEX chunks_embedding_bits_idx ON chunks USING hnsw
    ((binary_quantize(embedding)::bit(<dim>)) bit_hamming_ops);

-- UNIQUE, and it costs nothing to be: `chunk_id` is a hash of exactly these three
-- values, so the primary key already forbids a repeat. Saying it here puts the
-- rule where a reader of the table can see it, and the leftmost columns still
-- serve the revision lookups on their own.
CREATE UNIQUE INDEX chunks_revision_idx
    ON chunks (source_id, source_revision, chunk_index);

CREATE INDEX chunks_permission_idx ON chunks (category, visibility);

-- Partial on purpose. Most chunks are ordinary prose, and an index over a value
-- that matches most rows earns nothing while still costing space on every write.
CREATE INDEX chunks_kind_idx ON chunks (chunk_kind) WHERE chunk_kind <> 'text';

CREATE INDEX chunks_extra_idx ON chunks USING gin (extra jsonb_path_ops);
