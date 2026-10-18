-- Open Brain wiki schema (additive; the thoughts table is not touched).
-- Idempotent: the server and the compile script apply it on first use.

CREATE TABLE IF NOT EXISTS wiki_pages (
    slug TEXT PRIMARY KEY CHECK (slug ~ '^[a-z0-9]+(-[a-z0-9]+)*$' AND length(slug) <= 80),
    title TEXT NOT NULL,
    summary TEXT NOT NULL DEFAULT '',
    body TEXT NOT NULL DEFAULT '',
    tags TEXT[] NOT NULL DEFAULT '{}',
    links TEXT[] NOT NULL DEFAULT '{}',
    sources BIGINT[] NOT NULL DEFAULT '{}',
    embedding vector(1536),
    search_tsv TSVECTOR,
    revision INT NOT NULL DEFAULT 1,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_by TEXT NOT NULL DEFAULT 'mcp'
);

CREATE INDEX IF NOT EXISTS idx_wiki_pages_search ON wiki_pages USING GIN (search_tsv);
CREATE INDEX IF NOT EXISTS idx_wiki_pages_tags ON wiki_pages USING GIN (tags);
CREATE INDEX IF NOT EXISTS idx_wiki_pages_links ON wiki_pages USING GIN (links);
CREATE INDEX IF NOT EXISTS idx_wiki_pages_updated_at ON wiki_pages (updated_at DESC);

CREATE TABLE IF NOT EXISTS wiki_revisions (
    id BIGSERIAL PRIMARY KEY,
    slug TEXT NOT NULL,
    revision INT NOT NULL,
    title TEXT NOT NULL,
    summary TEXT NOT NULL,
    body TEXT NOT NULL,
    tags TEXT[] NOT NULL,
    links TEXT[] NOT NULL,
    sources BIGINT[] NOT NULL,
    author TEXT NOT NULL,
    note TEXT NOT NULL DEFAULT '',
    compile_run_id BIGINT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (slug, revision)
);

CREATE INDEX IF NOT EXISTS idx_wiki_revisions_created_at ON wiki_revisions (created_at DESC);

CREATE TABLE IF NOT EXISTS wiki_compile_runs (
    id BIGSERIAL PRIMARY KEY,
    started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at TIMESTAMPTZ,
    status TEXT NOT NULL DEFAULT 'running' CHECK (status IN ('running', 'ok', 'partial', 'failed')),
    model TEXT NOT NULL DEFAULT '',
    from_thought_id BIGINT NOT NULL,
    to_thought_id BIGINT,
    thoughts_seen INT NOT NULL DEFAULT 0,
    pages_created INT NOT NULL DEFAULT 0,
    pages_updated INT NOT NULL DEFAULT 0,
    summary TEXT NOT NULL DEFAULT '',
    error TEXT
);
