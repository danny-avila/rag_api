"""Integration tests: verify metadata filter SQL uses indexes on a real pgvector database.

These tests spin up a real PostgreSQL+pgvector container via testcontainers and
confirm that our ExtendedPgVector._handle_field_filter override produces queries
that use the B-tree expression index rather than sequential scans.

Run with:  pytest tests/integration/ -m integration -v
"""

import json
import math
import random
import uuid

import pytest
import sqlalchemy
from sqlalchemy import text
from sqlalchemy.dialects import postgresql
from sqlalchemy.orm import Session

from app.services.vector_store.extended_pg_vector import ExtendedPgVector
from langchain_community.vectorstores.pgvector import PGVector

pytestmark = pytest.mark.integration

ROW_COUNT = 50_000
BATCH_SIZE = 2_000
NUM_DISTINCT_FILES = 500


def _seed_embeddings(conn, collection_id, count=50, file_id="file-aaa"):
    """Insert test embeddings with a given file_id (small batches)."""
    for i in range(count):
        conn.execute(
            text(
                "INSERT INTO langchain_pg_embedding "
                "(collection_id, embedding, document, cmetadata, custom_id) "
                "VALUES (:cid, :emb, :doc, :meta, :cust)"
            ),
            {
                "cid": collection_id,
                "emb": f"[{i % 10 * 0.1},{i % 5 * 0.2},{i % 3 * 0.3}]",
                "doc": f"Document {i}",
                "meta": json.dumps({"file_id": file_id, "index": i}),
                "cust": f"custom-{file_id}-{i}",
            },
        )


def _bulk_seed(conn, collection_id, total=ROW_COUNT, num_files=NUM_DISTINCT_FILES):
    """Bulk-insert rows across many file_ids using multi-row VALUES for speed."""
    file_ids = [f"file-{uuid.uuid4().hex[:12]}" for _ in range(num_files)]
    rng = random.Random(42)

    for offset in range(0, total, BATCH_SIZE):
        batch_end = min(offset + BATCH_SIZE, total)
        values_parts = []
        params = {"cid": collection_id}
        for i in range(offset, batch_end):
            fid = rng.choice(file_ids)
            idx = f"_{i}"
            params[f"emb{idx}"] = (
                f"[{rng.random():.4f},{rng.random():.4f},{rng.random():.4f}]"
            )
            params[f"doc{idx}"] = f"Document {i}"
            params[f"meta{idx}"] = json.dumps({"file_id": fid, "index": i})
            params[f"cust{idx}"] = f"cust-{i}"
            values_parts.append(
                f"(:cid, :emb{idx}\\:\\:vector, :doc{idx}, :meta{idx}\\:\\:jsonb, :cust{idx})"
            )
        sql = (
            "INSERT INTO langchain_pg_embedding "
            "(collection_id, embedding, document, cmetadata, custom_id) VALUES "
            + ", ".join(values_parts)
        )
        conn.execute(text(sql), params)

    return file_ids


def _get_plan_node_type(plan_json):
    """Extract the top-level node type from EXPLAIN JSON output."""
    if isinstance(plan_json, list):
        plan_json = plan_json[0]
    node = plan_json.get("Plan", plan_json)
    return node.get("Node Type", "Unknown")


def _get_plan_details(plan_json):
    """Extract node type, actual time, and full plan text for diagnostics."""
    if isinstance(plan_json, list):
        plan_json = plan_json[0]
    node = plan_json.get("Plan", plan_json)
    return {
        "node_type": node.get("Node Type", "Unknown"),
        "actual_time_ms": node.get("Actual Total Time", -1),
        "actual_rows": node.get("Actual Rows", -1),
        "plan_rows": node.get("Plan Rows", -1),
    }


def _walk_plan_nodes(plan_json):
    """Yield every node in the EXPLAIN JSON tree."""
    if isinstance(plan_json, list):
        plan_json = plan_json[0]
    node = plan_json.get("Plan", plan_json)
    yield node
    for child in node.get("Plans", []):
        yield from _walk_plan_nodes(child)


@pytest.fixture(scope="module")
def seeded_data(engine, collection_id):
    """Seed the database once for the module with a realistic row count.

    Returns (file_ids, target_file_id) where target_file_id is one specific
    file_id to query for.
    """
    with engine.begin() as conn:
        row_count = conn.execute(
            text("SELECT count(*) FROM langchain_pg_embedding")
        ).scalar()

        if row_count < ROW_COUNT:
            file_ids = _bulk_seed(conn, collection_id, total=ROW_COUNT)
        else:
            result = conn.execute(
                text(
                    "SELECT DISTINCT cmetadata->>'file_id' "
                    "FROM langchain_pg_embedding LIMIT :lim"
                ),
                {"lim": NUM_DISTINCT_FILES},
            )
            file_ids = [r[0] for r in result]

        conn.execute(text("ANALYZE langchain_pg_embedding"))

        target = conn.execute(
            text(
                "SELECT cmetadata->>'file_id', count(*) as cnt "
                "FROM langchain_pg_embedding "
                "GROUP BY cmetadata->>'file_id' "
                "ORDER BY cnt DESC LIMIT 1"
            )
        ).fetchone()
        target_file_id = target[0]

    return file_ids, target_file_id


class TestQueryPlanRegression:
    """Prove the bug: jsonb_path_match forces seq scan, astext uses index.

    Seeds ~50K rows across 500 file_ids so the planner has real statistics.
    Uses EXPLAIN (ANALYZE) for actual execution times, not just estimates.
    """

    def test_astext_eq_uses_index(self, engine, seeded_data):
        """Our fix: cmetadata->>'file_id' = ... MUST use an index scan."""
        _, target_file_id = seeded_data

        with engine.begin() as conn:
            plan_json = conn.execute(
                text(
                    "EXPLAIN (ANALYZE, FORMAT JSON) "
                    "SELECT * FROM langchain_pg_embedding "
                    "WHERE (cmetadata->>'file_id') = :fid"
                ),
                {"fid": target_file_id},
            ).scalar()

        plan_str = json.dumps(plan_json).lower()
        has_index = any(
            "index" in n.get("Node Type", "").lower()
            for n in _walk_plan_nodes(plan_json)
        )
        assert has_index, (
            f"Expected index scan for ->>'file_id' = ...\n"
            f"Got plan: {json.dumps(plan_json, indent=2)}"
        )

    def test_jsonb_path_match_forces_seq_scan(self, engine, seeded_data):
        """The bug: jsonb_path_match() cannot use the expression index."""
        _, target_file_id = seeded_data

        with engine.begin() as conn:
            plan_json = conn.execute(
                text(
                    "EXPLAIN (ANALYZE, FORMAT JSON) "
                    "SELECT * FROM langchain_pg_embedding "
                    "WHERE jsonb_path_match("
                    "  cmetadata, '$.file_id == $value',"
                    "  jsonb_build_object('value', :fid \\:\\:text)"
                    ")"
                ),
                {"fid": target_file_id},
            ).scalar()

        node_types = [
            n.get("Node Type", "").lower() for n in _walk_plan_nodes(plan_json)
        ]
        has_seq_scan = any("seq scan" in nt for nt in node_types)
        has_no_index = not any("index" in nt for nt in node_types)
        assert has_seq_scan and has_no_index, (
            f"Expected seq scan (no index) for jsonb_path_match.\n"
            f"Node types found: {node_types}\n"
            f"Full plan: {json.dumps(plan_json, indent=2)}"
        )

    def test_astext_eq_faster_than_jsonb_path_match(self, engine, seeded_data):
        """Measure actual execution time: astext should be significantly faster."""
        _, target_file_id = seeded_data

        with engine.begin() as conn:
            astext_plan = conn.execute(
                text(
                    "EXPLAIN (ANALYZE, FORMAT JSON) "
                    "SELECT * FROM langchain_pg_embedding "
                    "WHERE (cmetadata->>'file_id') = :fid"
                ),
                {"fid": target_file_id},
            ).scalar()

            jpm_plan = conn.execute(
                text(
                    "EXPLAIN (ANALYZE, FORMAT JSON) "
                    "SELECT * FROM langchain_pg_embedding "
                    "WHERE jsonb_path_match("
                    "  cmetadata, '$.file_id == $value',"
                    "  jsonb_build_object('value', :fid \\:\\:text)"
                    ")"
                ),
                {"fid": target_file_id},
            ).scalar()

        astext_details = _get_plan_details(astext_plan)
        jpm_details = _get_plan_details(jpm_plan)

        astext_ms = astext_details["actual_time_ms"]
        jpm_ms = jpm_details["actual_time_ms"]

        report = (
            f"\n{'='*70}\n"
            f"  QUERY PERFORMANCE COMPARISON ({ROW_COUNT:,} rows)\n"
            f"{'='*70}\n"
            f"  FIX  (cmetadata->>'file_id' = ...)\n"
            f"    Plan node : {astext_details['node_type']}\n"
            f"    Time      : {astext_ms:.3f} ms\n"
            f"    Rows      : {astext_details['actual_rows']}\n"
            f"{'  -'*23}\n"
            f"  BUG  (jsonb_path_match)\n"
            f"    Plan node : {jpm_details['node_type']}\n"
            f"    Time      : {jpm_ms:.3f} ms\n"
            f"    Rows      : {jpm_details['actual_rows']}\n"
            f"{'  -'*23}\n"
            f"  Speedup     : {jpm_ms / astext_ms:.1f}x faster with fix\n"
            f"{'='*70}"
        )
        print(report)

        assert astext_ms < jpm_ms, (
            f"Expected astext ({astext_ms:.3f}ms) to be faster than "
            f"jsonb_path_match ({jpm_ms:.3f}ms).\n{report}"
        )

    def test_both_return_same_results(self, engine, seeded_data):
        """Sanity check: both query shapes return identical rows."""
        _, target_file_id = seeded_data

        with engine.begin() as conn:
            astext_rows = conn.execute(
                text(
                    "SELECT uuid FROM langchain_pg_embedding "
                    "WHERE (cmetadata->>'file_id') = :fid "
                    "ORDER BY uuid"
                ),
                {"fid": target_file_id},
            ).fetchall()

            jpm_rows = conn.execute(
                text(
                    "SELECT uuid FROM langchain_pg_embedding "
                    "WHERE jsonb_path_match("
                    "  cmetadata, '$.file_id == $value',"
                    "  jsonb_build_object('value', :fid \\:\\:text)"
                    ") ORDER BY uuid"
                ),
                {"fid": target_file_id},
            ).fetchall()

        astext_ids = [r[0] for r in astext_rows]
        jpm_ids = [r[0] for r in jpm_rows]
        assert astext_ids == jpm_ids, (
            f"Result mismatch: astext returned {len(astext_ids)} rows, "
            f"jsonb_path_match returned {len(jpm_ids)} rows"
        )


class TestFilterCorrectness:
    """Verify that filters return correct results from a real database."""

    def test_eq_returns_matching_documents(self, engine, collection_id):
        target_file = f"file-{uuid.uuid4().hex[:8]}"
        other_file = f"file-{uuid.uuid4().hex[:8]}"

        with engine.begin() as conn:
            _seed_embeddings(conn, collection_id, count=10, file_id=target_file)
            _seed_embeddings(conn, collection_id, count=20, file_id=other_file)

            result = conn.execute(
                text(
                    "SELECT count(*) FROM langchain_pg_embedding "
                    "WHERE (cmetadata->>'file_id') = :fid"
                ),
                {"fid": target_file},
            )
            assert result.scalar() == 10

    def test_ne_excludes_documents(self, engine, collection_id):
        target_file = f"file-{uuid.uuid4().hex[:8]}"
        other_file = f"file-{uuid.uuid4().hex[:8]}"

        with engine.begin() as conn:
            _seed_embeddings(conn, collection_id, count=5, file_id=target_file)
            _seed_embeddings(conn, collection_id, count=15, file_id=other_file)

            result = conn.execute(
                text(
                    "SELECT count(*) FROM langchain_pg_embedding "
                    "WHERE (cmetadata->>'file_id') != :fid "
                    "AND collection_id = :cid"
                ),
                {"fid": target_file, "cid": str(collection_id)},
            )
            count = result.scalar()
            assert count >= 15

    def test_in_filter_returns_multiple_file_ids(self, engine, collection_id):
        file_a = f"file-{uuid.uuid4().hex[:8]}"
        file_b = f"file-{uuid.uuid4().hex[:8]}"
        file_c = f"file-{uuid.uuid4().hex[:8]}"

        with engine.begin() as conn:
            _seed_embeddings(conn, collection_id, count=5, file_id=file_a)
            _seed_embeddings(conn, collection_id, count=5, file_id=file_b)
            _seed_embeddings(conn, collection_id, count=5, file_id=file_c)

            result = conn.execute(
                text(
                    "SELECT count(*) FROM langchain_pg_embedding "
                    "WHERE (cmetadata->>'file_id') IN (:fa, :fb)"
                ),
                {"fa": file_a, "fb": file_b},
            )
            assert result.scalar() == 10


class TestExtendedPgVectorSQL:
    """Test that ExtendedPgVector._handle_field_filter produces the right SQL
    and that the SQL actually works against a real database.
    """

    @pytest.fixture()
    def store(self):
        from langchain_community.vectorstores.pgvector import (
            _get_embedding_collection_store,
        )

        class TestableStore(ExtendedPgVector):
            def __init__(self):
                self._bind = None
                EmbeddingStore, _ = _get_embedding_collection_store(
                    vector_dimension=3, use_jsonb=True
                )
                self.EmbeddingStore = EmbeddingStore

        return TestableStore()

    def test_eq_clause_runs_on_real_pg(self, engine, collection_id, store):
        """Compile the $eq clause and execute it on the real database."""
        target_file = f"file-{uuid.uuid4().hex[:8]}"

        with engine.begin() as conn:
            _seed_embeddings(conn, collection_id, count=10, file_id=target_file)
            _seed_embeddings(conn, collection_id, count=10, file_id="file-noise")

        clause = store._handle_field_filter("file_id", {"$eq": target_file})

        with Session(engine) as session:
            query = (
                session.query(store.EmbeddingStore)
                .filter(store.EmbeddingStore.collection_id == collection_id)
                .filter(clause)
            )
            results = query.all()
            assert len(results) == 10
            assert all(r.cmetadata["file_id"] == target_file for r in results)

    def test_ne_clause_runs_on_real_pg(self, engine, collection_id, store):
        target_file = f"file-{uuid.uuid4().hex[:8]}"
        noise_file = f"file-{uuid.uuid4().hex[:8]}"

        with engine.begin() as conn:
            _seed_embeddings(conn, collection_id, count=5, file_id=target_file)
            _seed_embeddings(conn, collection_id, count=8, file_id=noise_file)

        clause = store._handle_field_filter("file_id", {"$ne": target_file})

        with Session(engine) as session:
            query = (
                session.query(store.EmbeddingStore)
                .filter(store.EmbeddingStore.collection_id == collection_id)
                .filter(clause)
            )
            results = query.all()
            assert all(r.cmetadata["file_id"] != target_file for r in results)

    def test_in_clause_runs_on_real_pg(self, engine, collection_id, store):
        file_a = f"file-{uuid.uuid4().hex[:8]}"
        file_b = f"file-{uuid.uuid4().hex[:8]}"

        with engine.begin() as conn:
            _seed_embeddings(conn, collection_id, count=3, file_id=file_a)
            _seed_embeddings(conn, collection_id, count=7, file_id=file_b)
            _seed_embeddings(conn, collection_id, count=5, file_id="file-ignored")

        clause = store._handle_field_filter("file_id", {"$in": [file_a, file_b]})

        with Session(engine) as session:
            query = (
                session.query(store.EmbeddingStore)
                .filter(store.EmbeddingStore.collection_id == collection_id)
                .filter(clause)
            )
            results = query.all()
            assert len(results) == 10
            assert all(r.cmetadata["file_id"] in (file_a, file_b) for r in results)


def _compile_clause(clause):
    """Compile a SQLAlchemy clause to a PostgreSQL SQL string."""
    return str(
        clause.compile(
            dialect=postgresql.dialect(),
            compile_kwargs={"literal_binds": True},
        )
    )


class TestWithoutOverride:
    """Prove the bug exists in LangChain's default code path (without our fix).

    Calls PGVector._handle_field_filter (the parent class method) directly,
    bypassing ExtendedPgVector's override, to confirm:
      1. LangChain's default $eq emits jsonb_path_match()
      2. That SQL forces a sequential scan on real PostgreSQL
      3. Our override produces different (faster) SQL for the same input

    These tests document that the issue is real and upstream, not hypothetical.
    """

    @pytest.fixture(scope="class")
    def store(self):
        from langchain_community.vectorstores.pgvector import (
            _get_embedding_collection_store,
        )

        class TestableStore(ExtendedPgVector):
            def __init__(self):
                self._bind = None
                self.use_jsonb = True
                EmbeddingStore, _ = _get_embedding_collection_store(
                    vector_dimension=3, use_jsonb=True
                )
                self.EmbeddingStore = EmbeddingStore

        return TestableStore()

    def test_langchain_default_eq_emits_jsonb_path_match(self, store):
        """Without our override, LangChain generates jsonb_path_match for $eq."""
        upstream_clause = PGVector._handle_field_filter(
            store, "file_id", {"$eq": "test-id"}
        )
        sql = _compile_clause(upstream_clause)
        assert "jsonb_path_match" in sql, (
            f"Expected LangChain default to use jsonb_path_match.\n"
            f"Got: {sql}\n"
            f"If this fails, LangChain may have fixed the issue upstream — "
            f"review whether our override is still needed."
        )
        assert "->>" not in sql, (
            f"LangChain default unexpectedly uses ->> for $eq.\n"
            f"Got: {sql}\n"
            f"The upstream bug may have been fixed."
        )

    def test_our_override_emits_astext_for_same_input(self, store):
        """Our override produces ->> instead of jsonb_path_match for $eq."""
        our_clause = store._handle_field_filter("file_id", {"$eq": "test-id"})
        our_sql = _compile_clause(our_clause)

        upstream_clause = PGVector._handle_field_filter(
            store, "file_id", {"$eq": "test-id"}
        )
        upstream_sql = _compile_clause(upstream_clause)

        assert (
            our_sql != upstream_sql
        ), "Override produces identical SQL to parent — override may be a no-op"
        assert "->>" in our_sql
        assert "jsonb_path_match" not in our_sql

    def test_langchain_default_causes_seq_scan_on_real_pg(
        self, engine, seeded_data, store
    ):
        """The upstream jsonb_path_match SQL seq-scans on real PostgreSQL."""
        _, target_file_id = seeded_data

        upstream_clause = PGVector._handle_field_filter(
            store, "file_id", {"$eq": target_file_id}
        )

        with Session(engine) as session:
            sa_query = session.query(store.EmbeddingStore).filter(upstream_clause)
            compiled = sa_query.statement.compile(
                dialect=engine.dialect,
                compile_kwargs={"literal_binds": True},
            )
            full_sql = str(compiled)

        with engine.begin() as conn:
            plan_json = conn.execute(
                text(f"EXPLAIN (ANALYZE, FORMAT JSON) {full_sql}")
            ).scalar()

        node_types = [
            n.get("Node Type", "").lower() for n in _walk_plan_nodes(plan_json)
        ]
        has_seq_scan = any("seq scan" in nt for nt in node_types)
        has_index = any("index" in nt for nt in node_types)
        assert has_seq_scan and not has_index, (
            f"Expected seq scan (no index) from LangChain's default filter.\n"
            f"Node types: {node_types}\n"
            f"SQL: {full_sql}\n"
            f"If this fails, LangChain may have fixed the issue upstream."
        )

    def test_our_override_uses_index_on_real_pg(self, engine, seeded_data, store):
        """Our override uses an index scan for the same filter on real PG."""
        _, target_file_id = seeded_data

        our_clause = store._handle_field_filter("file_id", {"$eq": target_file_id})

        with Session(engine) as session:
            sa_query = session.query(store.EmbeddingStore).filter(our_clause)
            compiled = sa_query.statement.compile(
                dialect=engine.dialect,
                compile_kwargs={"literal_binds": True},
            )
            full_sql = str(compiled)

        with engine.begin() as conn:
            plan_json = conn.execute(
                text(f"EXPLAIN (ANALYZE, FORMAT JSON) {full_sql}")
            ).scalar()

        node_types = [
            n.get("Node Type", "").lower() for n in _walk_plan_nodes(plan_json)
        ]
        has_index = any("index" in nt for nt in node_types)
        assert has_index, (
            f"Expected index scan from our override.\n"
            f"Node types: {node_types}\n"
            f"SQL: {full_sql}"
        )

    def test_performance_comparison_with_and_without_override(
        self, engine, seeded_data, store
    ):
        """Side-by-side: our override vs LangChain default, actual execution time."""
        _, target_file_id = seeded_data

        our_clause = store._handle_field_filter("file_id", {"$eq": target_file_id})
        upstream_clause = PGVector._handle_field_filter(
            store, "file_id", {"$eq": target_file_id}
        )

        with Session(engine) as session:
            our_sql = str(
                session.query(store.EmbeddingStore)
                .filter(our_clause)
                .statement.compile(
                    dialect=engine.dialect,
                    compile_kwargs={"literal_binds": True},
                )
            )
            upstream_sql = str(
                session.query(store.EmbeddingStore)
                .filter(upstream_clause)
                .statement.compile(
                    dialect=engine.dialect,
                    compile_kwargs={"literal_binds": True},
                )
            )

        with engine.begin() as conn:
            our_plan = conn.execute(
                text(f"EXPLAIN (ANALYZE, FORMAT JSON) {our_sql}")
            ).scalar()
            upstream_plan = conn.execute(
                text(f"EXPLAIN (ANALYZE, FORMAT JSON) {upstream_sql}")
            ).scalar()

        our_details = _get_plan_details(our_plan)
        upstream_details = _get_plan_details(upstream_plan)

        report = (
            f"\n{'='*70}\n"
            f"  WITH vs WITHOUT override ({ROW_COUNT:,} rows)\n"
            f"{'='*70}\n"
            f"  WITH override (ExtendedPgVector)\n"
            f"    Plan node : {our_details['node_type']}\n"
            f"    Time      : {our_details['actual_time_ms']:.3f} ms\n"
            f"{'  -'*23}\n"
            f"  WITHOUT override (LangChain default)\n"
            f"    Plan node : {upstream_details['node_type']}\n"
            f"    Time      : {upstream_details['actual_time_ms']:.3f} ms\n"
            f"{'  -'*23}\n"
            f"  Speedup     : "
            f"{upstream_details['actual_time_ms'] / max(our_details['actual_time_ms'], 0.001):.1f}x\n"
            f"{'='*70}"
        )
        print(report)

        assert (
            our_details["actual_time_ms"] < upstream_details["actual_time_ms"]
        ), f"Override should be faster than LangChain default.\n{report}"


class TestLangChainUpgradeGuardrails:
    """Catch regressions if a LangChain upgrade changes filter SQL generation.

    These tests exercise the full _create_filter_clause -> _handle_field_filter
    pipeline using the exact same filter dicts our /query and /query_multiple
    routes pass. They verify:

      1. The compiled SQL uses ->> (not jsonb_path_match)
      2. The compiled SQL actually uses an index on real PostgreSQL
      3. The results are correct

    If someone bumps langchain-community and the parent class changes behavior,
    or if our override is accidentally removed, these tests will fail.
    """

    @pytest.fixture(scope="class")
    def store(self):
        from langchain_community.vectorstores.pgvector import (
            _get_embedding_collection_store,
        )

        class TestableStore(ExtendedPgVector):
            def __init__(self):
                self._bind = None
                self.use_jsonb = True
                EmbeddingStore, _ = _get_embedding_collection_store(
                    vector_dimension=3, use_jsonb=True
                )
                self.EmbeddingStore = EmbeddingStore

        return TestableStore()

    def test_query_route_filter_sql_shape(self, store):
        """The /query route filter must compile to ->> not jsonb_path_match."""
        route_filter = {"file_id": {"$eq": "test-file-id"}}
        clause = store._create_filter_clause(route_filter)
        sql = _compile_clause(clause)

        assert "->>" in sql, (
            f"Expected ->> operator in compiled SQL for /query filter.\n"
            f"Got: {sql}\n"
            f"This likely means a LangChain upgrade changed the filter path "
            f"or the ExtendedPgVector override was removed."
        )
        assert "jsonb_path_match" not in sql, (
            f"jsonb_path_match detected in compiled SQL for /query filter.\n"
            f"Got: {sql}\n"
            f"This will cause sequential scans on large tables. "
            f"The ExtendedPgVector._handle_field_filter override may be broken."
        )

    def test_query_multiple_route_filter_sql_shape(self, store):
        """The /query_multiple route filter must compile to ->> IN(...)."""
        route_filter = {"file_id": {"$in": ["file-a", "file-b", "file-c"]}}
        clause = store._create_filter_clause(route_filter)
        sql = _compile_clause(clause)

        assert "->>" in sql, (
            f"Expected ->> operator in compiled SQL for /query_multiple filter.\n"
            f"Got: {sql}"
        )
        assert "IN" in sql, (
            f"Expected IN clause in compiled SQL for /query_multiple filter.\n"
            f"Got: {sql}"
        )
        assert "jsonb_path_match" not in sql, (
            f"jsonb_path_match detected in /query_multiple filter SQL.\n" f"Got: {sql}"
        )

    def test_query_route_filter_uses_index_on_real_pg(self, engine, seeded_data, store):
        """End-to-end: /query filter compiles, runs on PG, uses index."""
        _, target_file_id = seeded_data

        route_filter = {"file_id": {"$eq": target_file_id}}
        clause = store._create_filter_clause(route_filter)

        with Session(engine) as session:
            sa_query = session.query(store.EmbeddingStore).filter(clause)
            compiled = sa_query.statement.compile(
                dialect=engine.dialect,
                compile_kwargs={"literal_binds": True},
            )
            full_sql = str(compiled)

        with engine.begin() as conn:
            plan_json = conn.execute(
                text(f"EXPLAIN (ANALYZE, FORMAT JSON) {full_sql}")
            ).scalar()

        node_types = [
            n.get("Node Type", "").lower() for n in _walk_plan_nodes(plan_json)
        ]
        has_index = any("index" in nt for nt in node_types)
        assert has_index, (
            f"Expected index scan for /query route filter on real PostgreSQL.\n"
            f"Node types: {node_types}\n"
            f"SQL: {full_sql}\n"
            f"This means the filter SQL no longer matches available indexes."
        )

    def test_query_multiple_route_filter_uses_index_on_real_pg(
        self, engine, seeded_data, store
    ):
        """End-to-end: /query_multiple filter compiles, runs on PG, uses index."""
        file_ids, _ = seeded_data
        target_ids = file_ids[:3]

        route_filter = {"file_id": {"$in": target_ids}}
        clause = store._create_filter_clause(route_filter)

        with Session(engine) as session:
            sa_query = session.query(store.EmbeddingStore).filter(clause)
            compiled = sa_query.statement.compile(
                dialect=engine.dialect,
                compile_kwargs={"literal_binds": True},
            )
            full_sql = str(compiled)

        with engine.begin() as conn:
            plan_json = conn.execute(
                text(f"EXPLAIN (ANALYZE, FORMAT JSON) {full_sql}")
            ).scalar()

        node_types = [
            n.get("Node Type", "").lower() for n in _walk_plan_nodes(plan_json)
        ]
        has_index = any("index" in nt for nt in node_types)
        assert has_index, (
            f"Expected index scan for /query_multiple route filter.\n"
            f"Node types: {node_types}\n"
            f"SQL: {full_sql}"
        )

    def test_query_route_filter_returns_correct_results(
        self, engine, collection_id, store
    ):
        """End-to-end: /query filter returns only matching file_id rows."""
        target_file = f"file-guard-{uuid.uuid4().hex[:8]}"
        noise_file = f"file-guard-{uuid.uuid4().hex[:8]}"

        with engine.begin() as conn:
            _seed_embeddings(conn, collection_id, count=12, file_id=target_file)
            _seed_embeddings(conn, collection_id, count=20, file_id=noise_file)

        route_filter = {"file_id": {"$eq": target_file}}
        clause = store._create_filter_clause(route_filter)

        with Session(engine) as session:
            results = (
                session.query(store.EmbeddingStore)
                .filter(store.EmbeddingStore.collection_id == collection_id)
                .filter(clause)
                .all()
            )
        assert len(results) == 12
        assert all(r.cmetadata["file_id"] == target_file for r in results)

    def test_query_multiple_route_filter_returns_correct_results(
        self, engine, collection_id, store
    ):
        """End-to-end: /query_multiple filter returns rows for all requested file_ids."""
        file_a = f"file-guard-{uuid.uuid4().hex[:8]}"
        file_b = f"file-guard-{uuid.uuid4().hex[:8]}"
        noise = f"file-guard-{uuid.uuid4().hex[:8]}"

        with engine.begin() as conn:
            _seed_embeddings(conn, collection_id, count=6, file_id=file_a)
            _seed_embeddings(conn, collection_id, count=4, file_id=file_b)
            _seed_embeddings(conn, collection_id, count=15, file_id=noise)

        route_filter = {"file_id": {"$in": [file_a, file_b]}}
        clause = store._create_filter_clause(route_filter)

        with Session(engine) as session:
            results = (
                session.query(store.EmbeddingStore)
                .filter(store.EmbeddingStore.collection_id == collection_id)
                .filter(clause)
                .all()
            )
        result_fids = {r.cmetadata["file_id"] for r in results}
        assert len(results) == 10
        assert result_fids == {file_a, file_b}
