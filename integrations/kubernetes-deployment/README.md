# Kubernetes Self-Hosted Deployment

<div align="center">

![Community Contribution](https://img.shields.io/badge/OB1_COMMUNITY-Approved_Contribution-2ea44f?style=for-the-badge&logo=github)

**Created by [@velo](https://github.com/velo)**

*Reviewed and merged by the Open Brain maintainer team — thank you for building the future of AI memory!*

</div>

> Deploy Open Brain on Kubernetes with self-hosted PostgreSQL + pgvector, replacing Supabase with fully self-managed infrastructure.

## What It Does

This integration provides Kubernetes manifests and a modified MCP server that connects directly to PostgreSQL instead of Supabase. Your thoughts database, embeddings, and MCP endpoint all run on your own cluster. The MCP HTTP endpoint is served via Kubernetes Ingress, making it a remote endpoint accessible by URL from any MCP client.

## Prerequisites

- Working Kubernetes cluster (tested on K3s v1.31, works with any K8s distribution)
- `kubectl` configured for your cluster
- Docker installed (for building the MCP server image)
- An embedding/chat API provider (OpenRouter, OpenAI, or a local model with OpenAI-compatible API)
- An ingress controller (Traefik, nginx-ingress, etc.) if you want external access

## Credential Tracker

Copy this block into a text editor and fill it in as you go.

```text
KUBERNETES DEPLOYMENT -- CREDENTIAL TRACKER
--------------------------------------------

POSTGRESQL
  Password:              ____________

MCP SERVER
  Access key:            ____________

EMBEDDING/CHAT API
  API base URL:          ____________
  API key:               ____________
  Embedding model:       ____________
  Chat model:            ____________

--------------------------------------------
```

## Steps

1. Build the MCP server Docker image
2. Configure secrets
3. Deploy to Kubernetes
4. Verify the deployment
5. Connect your MCP client

### 1. Build the MCP Server Docker Image

From this directory, build and import the image:

```bash
docker build -t openbrain-mcp-server:latest .

# For K3s:
docker save openbrain-mcp-server:latest | sudo k3s ctr images import -

# For minikube:
minikube image load openbrain-mcp-server:latest

# For other clusters, push to your registry:
docker tag openbrain-mcp-server:latest your-registry/openbrain-mcp-server:latest
docker push your-registry/openbrain-mcp-server:latest
```

### 2. Configure Secrets

```bash
cp k8s/secrets.yml.example k8s/secrets.yml
```

Edit `k8s/secrets.yml` with your actual credentials. **Never commit this file.**

### 3. Deploy to Kubernetes

```bash
kubectl apply -f k8s/secrets.yml
kubectl apply -f k8s/openbrain.yml
```

### 4. Verify Deployment

```bash
# Check pod status
kubectl get pods -n openbrain

# Check database is initialized
kubectl exec -n openbrain openbrain-0 -c db -- \
  psql -U postgres -d openbrain -c '\dt'

# Test MCP endpoint (via port-forward)
kubectl port-forward -n openbrain svc/openbrain 8000:8000 &
curl -X POST http://localhost:8000 \
  -H "x-brain-key: YOUR_ACCESS_KEY" \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","method":"tools/list","id":1}'
```

### 5. Connect Your MCP Client

For Claude Desktop or any MCP-compatible client, configure the remote MCP endpoint:

```json
{
  "mcpServers": {
    "openbrain": {
      "url": "http://openbrain.openbrain.svc.cluster.local:8000",
      "transport": "http",
      "headers": {
        "x-brain-key": "YOUR_ACCESS_KEY"
      }
    }
  }
}
```

If you've configured an Ingress, use your external URL instead:

```json
{
  "mcpServers": {
    "openbrain": {
      "url": "https://brain.yourdomain.com",
      "transport": "http",
      "headers": {
        "x-brain-key": "YOUR_ACCESS_KEY"
      }
    }
  }
}
```

## Using a Local LLM Instead of OpenRouter

To use a local model (e.g., Ollama, BitNet, llama.cpp) for embeddings and chat, update the environment variables in `k8s/openbrain.yml`:

```yaml
- name: EMBEDDING_API_BASE
  value: "http://your-local-model:8080/v1"
- name: EMBEDDING_API_KEY
  value: "not-needed"
- name: EMBEDDING_MODEL
  value: "your-model-name"
- name: CHAT_API_BASE
  value: "http://your-local-model:8080/v1"
- name: CHAT_API_KEY
  value: "not-needed"
- name: CHAT_MODEL
  value: "your-model-name"
```

If your embedding model produces a different vector dimension than 1536, update the `vector(1536)` in the init SQL to match.

## Expected Outcome

After deployment you should see:

- `openbrain-0` pod running with 2 containers (db + mcp-server)
- PostgreSQL with `thoughts` table and `match_thoughts` function
- MCP endpoint responding to `tools/list` with 4 tools: `search_thoughts`, `list_thoughts`, `thought_stats`, `capture_thought`
- Thoughts captured via any MCP client are stored in your self-hosted database

> **Tool hygiene:** This integration adds MCP tools to your AI's context window. As your deployment grows, the total tool count grows — and with it, the context cost and risk of your AI picking the wrong tool. See the [MCP Tool Audit & Optimization Guide](../../docs/05-tool-audit.md) for strategies on auditing, merging, and scoping your tools.

## Wiki

The server also keeps a compiled wiki: markdown pages about the durable subjects in your thoughts (projects, clients, people, servers, rules), current state first, every fact cited as `[#thought-id]`. Thoughts stay the append-only source of truth; the wiki is rebuilt from them by a compile job. The tables (`wiki_pages`, `wiki_revisions`, `wiki_compile_runs`, see `wiki_schema.sql`) are created automatically on first use; `thoughts` is not changed.

MCP tools on the same endpoint and key:

| Tool | Purpose |
|------|---------|
| `wiki_index` | Table of contents grouped by category, sub-pages nested under their parent, plus compile status and the number of thoughts not compiled yet |
| `wiki_list` | Page list with one-line summaries; filter by tag or category, or pages updated in the last N days |
| `wiki_read` | One page with sources, links and backlinks; an older `revision` or `history=true`. Built-in pages: `index`, `log` (changes), `pending` (thoughts no page cites yet) |
| `wiki_search` | Full-text (Postgres `simple` config, works for any language) plus embedding similarity, merged by reciprocal rank fusion |
| `wiki_write` | Create or update a page; every write stores a revision. `expected_revision` prevents lost updates, `append=true` adds to the end |

The server also sends MCP `instructions` telling clients when to use the wiki and when to use raw thoughts.

### Compile job

`wiki_compile.ts` reads the thoughts captured after the last successful run (a high-water mark on thought ids), asks the chat model to plan pages and route each thought to one to three pages, then rewrites each affected page with the new thoughts (oldest first, `SUPERSEDES` and `RETRACTED` notes applied). Citations to ids the model was not given and links to unknown pages are stripped before the write. It ends with a lint pass (near-duplicate pages, broken links, pages nothing links to) recorded in the run summary and the `log` page. A page that fails, or a thought the model did not assign, stays pending for the next run (the run is recorded as `partial`; a thought left unassigned by two runs in a row is given up and named in the log); thoughts a page already cites are skipped, so re-runs are safe. Thoughts the model marks as noise are listed in the run summary and are not retried. Thoughts younger than `WIKI_COMPILE_SETTLE_SECONDS` (default 60) wait for the next run. Pages that have no embedding yet are embedded at the start of each run. Runs take an advisory lock, so two never overlap.

```bash
# inside the running server container (it has the database and API settings)
deno run --allow-net --allow-env --allow-read wiki_compile.ts --dry-run   # plan only
deno run --allow-net --allow-env --allow-read wiki_compile.ts             # compile
deno run --allow-net --allow-env --allow-read wiki_compile.ts --status
deno run --allow-net --allow-env --allow-read --allow-write=/tmp/wiki wiki_compile.ts --export /tmp/wiki
```

The model is `--model`, else `WIKI_COMPILE_MODEL`, else `CHAT_MODEL`; use the provider's model id (`gpt-5.4-mini` on the OpenAI API, `openai/gpt-5.4-mini` on OpenRouter). `systemd/` holds a service and weekly timer that run the compile with `docker exec` in the container matched by `OB1_CONTAINER_FILTER`:

```bash
cp systemd/ob1-wiki-compile.* /etc/systemd/system/
systemctl daemon-reload && systemctl enable --now ob1-wiki-compile.timer
journalctl -u ob1-wiki-compile.service   # last run
```

### Tests

`./test.sh` starts a throwaway pgvector container and runs `wiki_test.ts`: unit tests, then the real server and compile job against the database with a fake OpenAI-compatible API (old tools, every wiki tool, revisions, conflicts, search fallback, compile recovery, export). Everything it starts is removed afterwards.

## Troubleshooting

**Pod stuck in CrashLoopBackOff (mcp-server)**
- Check logs: `kubectl logs -n openbrain openbrain-0 -c mcp-server`
- Most common cause: invalid API key or unreachable embedding API base URL
- For local models, ensure the model service is running and accessible from the cluster

**Database not initialized / tables missing**
- The init SQL runs only on first startup. If the data volume already exists with an old database, the init script is skipped.
- To re-initialize: delete the data volume directory and restart the pod
- `kubectl delete pod openbrain-0 -n openbrain` (StatefulSet will recreate it)

**Embedding dimension mismatch**
- If you see errors about vector dimensions, your embedding model produces vectors of a different size than expected
- Check your model's output dimension and update `vector(1536)` in the init SQL ConfigMap
- Drop and recreate the `thoughts` table if changing dimensions on an existing database

**Connection refused to database**
- Containers in the same pod communicate via `127.0.0.1` — this is normal Kubernetes multi-container pod behavior
- Check that the `db` container is ready: `kubectl logs -n openbrain openbrain-0 -c db`
