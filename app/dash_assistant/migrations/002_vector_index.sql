-- app/dash_assistant/migrations/002_vector_index.sql
-- Create vector index and optimize for mass loading

-- Create HNSW index for vector similarity search
-- Using HNSW for better performance on large datasets
CREATE INDEX IF NOT EXISTS idx_bi_chunk_emb_hnsw
ON bi_chunk USING hnsw (embedding vector_cosine_ops) 
WITH (m = 16, ef_construction = 64);

-- Alternative IVFFlat index (commented out, use if HNSW is not available)
-- CREATE INDEX IF NOT EXISTS idx_bi_chunk_emb_ivfflat
-- ON bi_chunk USING ivfflat (embedding vector_cosine_ops) 
-- WITH (lists = 100);

-- Create function to analyze tables after mass loading
CREATE OR REPLACE FUNCTION analyze_dash_assistant_tables()
RETURNS void LANGUAGE plpgsql AS $$
BEGIN
    -- Analyze all dash assistant tables to update statistics
    -- This is critical for query performance after mass data loading
    ANALYZE bi_entity;
    ANALYZE bi_chart;
    ANALYZE bi_chunk;
    ANALYZE term_dict;
    ANALYZE query_log;
    
    -- Log the analysis
    RAISE NOTICE 'Analyzed all dash assistant tables for optimal query performance';
END$$;

-- Create function to optimize vector index after mass loading
CREATE OR REPLACE FUNCTION optimize_vector_index()
RETURNS void LANGUAGE plpgsql AS $$
BEGIN
    -- Reindex vector index for optimal performance
    -- This is important after mass loading of embeddings
    REINDEX INDEX idx_bi_chunk_emb_hnsw;
    
    -- Analyze the chunk table specifically for vector operations
    ANALYZE bi_chunk;
    
    RAISE NOTICE 'Optimized vector index and updated statistics';
END$$;

-- Create convenience function to run full optimization
CREATE OR REPLACE FUNCTION optimize_after_mass_loading()
RETURNS void LANGUAGE plpgsql AS $$
BEGIN
    PERFORM analyze_dash_assistant_tables();
    PERFORM optimize_vector_index();
    
    RAISE NOTICE 'Full optimization completed after mass loading';
END$$;
