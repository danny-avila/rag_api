#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
run_id="$(date +%s)-$$"
ch="ragv2-ch-$run_id"
runner="ragv2-bun-$run_id"
cleanup() {
  docker rm -f "$runner" "$ch" >/dev/null 2>&1 || true
}
trap cleanup EXIT
# No published database port or application credentials.
docker run -d --name "$ch" --network none --cpus 2 --memory 2g \
  -e CLICKHOUSE_SKIP_USER_SETUP=1 clickhouse/clickhouse-server:26.8 >/dev/null
for attempt in $(seq 1 60); do
  if docker exec "$ch" clickhouse-client --query 'SELECT 1' >/dev/null 2>&1; then break; fi
  if [ "$attempt" = 60 ]; then printf 'ClickHouse did not become ready\n' >&2; exit 1; fi
  sleep 1
done
docker run -d --name "$runner" --network "container:$ch" --entrypoint sh \
  -e RAG_V2_TEST_CLICKHOUSE_URL=http://127.0.0.1:8123 oven/bun:1.4.2-debian -c 'sleep 300' >/dev/null
docker exec "$runner" mkdir /app
docker cp . "$runner:/app/v2"
docker exec -w /app/v2 "$runner" bun test test/integration.test.ts
if [ "${RAG_V2_RUN_BENCH:-false}" = true ]; then
  docker exec -w /app/v2 "$runner" bun scripts/benchmark.ts
fi
