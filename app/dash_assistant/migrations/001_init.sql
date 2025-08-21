-- app/dash_assistant/migrations/001_init.sql
-- Initial schema for dash assistant

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE TABLE IF NOT EXISTS bi_entity (
  entity_id           BIGSERIAL PRIMARY KEY,
  entity_type         TEXT NOT NULL CHECK (entity_type IN ('dashboard','chart')),
  superset_id         TEXT,
  dashboard_slug      TEXT,
  title               TEXT NOT NULL,
  description         TEXT,
  domain              TEXT,
  owner               TEXT,
  tags                TEXT[],
  url                 TEXT,
  usage_score         REAL DEFAULT 0,
  last_refresh_ts     TIMESTAMPTZ,
  metadata            JSONB DEFAULT '{}'::jsonb
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_bi_entity_superset_id
  ON bi_entity (superset_id) WHERE superset_id IS NOT NULL;
CREATE UNIQUE INDEX IF NOT EXISTS uq_bi_entity_slug
  ON bi_entity (dashboard_slug) WHERE dashboard_slug IS NOT NULL;

CREATE TABLE IF NOT EXISTS bi_chart (
  chart_id            BIGSERIAL PRIMARY KEY,
  parent_dashboard_id BIGINT REFERENCES bi_entity(entity_id) ON DELETE CASCADE,
  superset_chart_id   TEXT,
  title               TEXT NOT NULL,
  viz_type            TEXT,
  sql_text            TEXT,
  metrics             JSONB,
  dimensions          JSONB,
  filters_default     JSONB,
  url                 TEXT
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_bi_chart_superset_chart_id
  ON bi_chart (superset_chart_id) WHERE superset_chart_id IS NOT NULL;

CREATE TABLE IF NOT EXISTS bi_chunk (
  chunk_id            BIGSERIAL PRIMARY KEY,
  entity_id           BIGINT REFERENCES bi_entity(entity_id) ON DELETE CASCADE,
  chart_id            BIGINT REFERENCES bi_chart(chart_id) ON DELETE CASCADE,
  scope               TEXT NOT NULL,       -- 'title'|'desc'|'chart_title'|'sql'
  content             TEXT NOT NULL,
  lang                TEXT,
  tsv_en              tsvector,
  tsv_ru              tsvector,
  embedding           vector(3072)
);

CREATE OR REPLACE FUNCTION make_tsvectors()
RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
  NEW.tsv_en := setweight(to_tsvector('english', coalesce(NEW.content,'')), 'A');
  NEW.tsv_ru := setweight(to_tsvector('russian', coalesce(NEW.content,'')), 'A');
  RETURN NEW;
END$$;

DROP TRIGGER IF EXISTS bi_chunk_tsvectors ON bi_chunk;
CREATE TRIGGER bi_chunk_tsvectors
BEFORE INSERT OR UPDATE ON bi_chunk
FOR EACH ROW EXECUTE FUNCTION make_tsvectors();

-- Vector index will be created separately after confirming pgvector version
-- CREATE INDEX IF NOT EXISTS idx_bi_chunk_emb_hnsw
-- ON bi_chunk USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS idx_bi_chunk_tsv_en ON bi_chunk USING GIN (tsv_en);
CREATE INDEX IF NOT EXISTS idx_bi_chunk_tsv_ru ON bi_chunk USING GIN (tsv_ru);

CREATE INDEX IF NOT EXISTS idx_bi_entity_title_trgm ON bi_entity USING GIN (title gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_bi_chart_title_trgm  ON bi_chart  USING GIN (title gin_trgm_ops);

CREATE TABLE IF NOT EXISTS term_dict (
  term            TEXT PRIMARY KEY,
  normalized      TEXT NOT NULL,
  lang            TEXT,
  synonyms        TEXT[]
);

CREATE TABLE IF NOT EXISTS query_log (
  qid             BIGSERIAL PRIMARY KEY,
  ts              TIMESTAMPTZ DEFAULT now(),
  user_id         TEXT,
  query_text      TEXT,
  intent_json     JSONB,
  chosen_entity   BIGINT REFERENCES bi_entity(entity_id),
  chosen_chart    BIGINT REFERENCES bi_chart(chart_id),
  scores          JSONB,
  feedback        TEXT         -- 'up'|'down' (Slack 👍/👎)
);
