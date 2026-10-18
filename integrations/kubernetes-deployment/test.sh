#!/usr/bin/env bash
# Runs wiki_test.ts against a throwaway pgvector container and removes it afterwards.
# Set DENO_CACHE_VOLUME to reuse a named Deno cache volume between runs.
set -euo pipefail
cd "$(dirname "$0")"

name="ob1-wiki-test-$$"
cache="${DENO_CACHE_VOLUME:-$name-deno}"

cleanup() {
  docker rm -f "$name-db" >/dev/null 2>&1 || true
  docker network rm "$name" >/dev/null 2>&1 || true
  if [ -z "${DENO_CACHE_VOLUME:-}" ]; then docker volume rm "$cache" >/dev/null 2>&1 || true; fi
}
trap cleanup EXIT

docker network create "$name" >/dev/null
docker run -d --name "$name-db" --network "$name" \
  -e POSTGRES_USER=openbrain -e POSTGRES_PASSWORD=test -e POSTGRES_DB=openbrain \
  -v "$PWD/k8s/init.sql:/docker-entrypoint-initdb.d/init.sql:ro" \
  pgvector/pgvector:pg17 >/dev/null

# TCP only opens after the init scripts have run, so this waits for a ready schema.
for _ in $(seq 1 60); do
  if docker exec "$name-db" psql -h 127.0.0.1 -U openbrain -d openbrain -c "SELECT 1 FROM thoughts LIMIT 1" >/dev/null 2>&1; then break; fi
  sleep 1
done

docker run --rm --network "$name" \
  -v "$PWD:/app:ro" -v "$cache:/deno-dir" -e DENO_DIR=/deno-dir \
  -e WIKI_TEST_DB_HOST="$name-db" -e WIKI_TEST_DB_PASSWORD=test \
  -w /app denoland/deno:2.3.3 \
  deno test --no-lock --allow-net --allow-env --allow-read --allow-run --allow-write=/tmp wiki_test.ts "$@"
