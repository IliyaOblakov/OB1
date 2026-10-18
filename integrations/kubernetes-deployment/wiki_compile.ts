// Open Brain wiki compiler: folds the thoughts captured since the last run into wiki pages.
// Runs inside the openbrain-mcp container with the server's environment; see --help.

import { Pool } from "postgres";
import {
  buildIndex,
  CATEGORIES,
  CITATION_PATTERN,
  compileStatus,
  embedMissing,
  ensureWikiSchema,
  getPage,
  LINK_PATTERN,
  listPages,
  normalizeTags,
  oneLine,
  renderLog,
  slugError,
  slugify,
  stamp,
  type WikiPage,
  WikiError,
  withClient,
  writePage,
} from "./wiki.ts";

const env = (name: string, fallback = "") => Deno.env.get(name) || fallback;

const EMBEDDING_API_BASE = env("EMBEDDING_API_BASE", "https://openrouter.ai/api/v1");
const EMBEDDING_API_KEY = env("EMBEDDING_API_KEY") || env("OPENROUTER_API_KEY");
const EMBEDDING_MODEL = env("EMBEDDING_MODEL", "openai/text-embedding-3-small");
const CHAT_API_BASE = env("CHAT_API_BASE", EMBEDDING_API_BASE);
const CHAT_API_KEY = env("CHAT_API_KEY", EMBEDDING_API_KEY);
const CHAT_MODEL = env("CHAT_MODEL", "openai/gpt-4o-mini");

const COMPILE_LOCK = 4_271_312;
const PLAN_BATCH = 1_500;
const PLAN_EXCERPT_CHARS = 200;
const ROUTE_BATCH = 60;
const ROUTE_EXCERPT_CHARS = 500;
const CHUNK_CHARS = 40_000;
const DUPLICATE_SIMILARITY = 0.9;
// Thoughts younger than this are left for the next run, so a capture still committing is never skipped.
const settle = Number(env("WIKI_COMPILE_SETTLE_SECONDS", "60"));
const SETTLE_SECONDS = Number.isFinite(settle) && settle >= 0 ? settle : 60;

const USAGE = `Open Brain wiki compiler

  deno run --allow-net --allow-env --allow-read wiki_compile.ts [options]

Options:
  --model NAME        chat model (default: WIKI_COMPILE_MODEL, then CHAT_MODEL)
  --limit N           compile at most N pending thoughts (oldest first)
  --concurrency N     pages written in parallel (default 4)
  --dry-run           plan and route only; print the assignment, write nothing
  --status            print the compile status and exit
  --export DIR        write every page plus index.md and log.md to DIR (needs --allow-write=DIR)
  --help              show this text`;

const pool = new Pool({
  hostname: env("DB_HOST", "127.0.0.1"),
  port: parseInt(env("DB_PORT", "5432"), 10),
  database: env("DB_NAME", "openbrain"),
  user: env("DB_USER", "postgres"),
  password: env("DB_PASSWORD"),
}, 8, true);

type Options = {
  model: string;
  limit: number;
  concurrency: number;
  dryRun: boolean;
  status: boolean;
  exportDir: string | null;
  help: boolean;
};

type Thought = { id: number; content: string; metadata: Record<string, unknown>; created_at: Date };

type Entry = {
  slug: string;
  title: string;
  category: string;
  summary: string;
  count: number;
  parent?: string;
  proposed: boolean;
};

type Draft = { title: string; summary: string; tags: string[]; body: string };

type Run = {
  id: number;
  model: string;
  thoughts: Map<number, Thought>;
  catalog: Map<string, Entry>;
  linkable: Set<string>;
};

const usage = { calls: 0, prompt: 0, completion: 0, reasoning: 0 };

function log(message: string): void {
  console.log(`${new Date().toISOString().slice(11, 19)} ${message}`);
}

function positiveInt(value: string, flag: string): number {
  const n = Number(value);
  if (!Number.isInteger(n) || n < 1) throw new Error(`${flag} needs a positive integer`);
  return n;
}

function parseOptions(args: string[]): Options {
  const opts: Options = {
    model: env("WIKI_COMPILE_MODEL", CHAT_MODEL),
    limit: 1_000_000,
    concurrency: 4,
    dryRun: false,
    status: false,
    exportDir: null,
    help: false,
  };
  for (let i = 0; i < args.length; i++) {
    const flag = args[i];
    const value = () => {
      const v = args[++i];
      if (v === undefined) throw new Error(`${flag} needs a value`);
      return v;
    };
    switch (flag) {
      case "--model": opts.model = value(); break;
      case "--limit": opts.limit = positiveInt(value(), flag); break;
      case "--concurrency": opts.concurrency = positiveInt(value(), flag); break;
      case "--dry-run": opts.dryRun = true; break;
      case "--status": opts.status = true; break;
      case "--export": opts.exportDir = value(); break;
      case "--help": case "-h": opts.help = true; break;
      default: throw new Error(`unknown argument ${flag} (see --help)`);
    }
  }
  return opts;
}

// --- Model calls ---

async function getEmbedding(text: string): Promise<number[]> {
  const r = await fetch(`${EMBEDDING_API_BASE}/embeddings`, {
    method: "POST",
    headers: { Authorization: `Bearer ${EMBEDDING_API_KEY}`, "Content-Type": "application/json" },
    body: JSON.stringify({ model: EMBEDDING_MODEL, input: text }),
    signal: AbortSignal.timeout(60_000),
  });
  if (!r.ok) throw new Error(`Embedding API failed: ${r.status} ${await r.text().catch(() => "")}`);
  const d = await r.json();
  return d.data[0].embedding;
}

class FatalApiError extends Error {}

async function chatJSON<T>(model: string, effort: "low" | "medium", system: string, user: string): Promise<T> {
  const body: Record<string, unknown> = {
    model,
    response_format: { type: "json_object" },
    max_completion_tokens: 32_000,
    messages: [{ role: "system", content: system }, { role: "user", content: user }],
  };
  if (/^(openai\/)?(gpt-5|o\d)/.test(model)) body.reasoning_effort = effort;
  else body.temperature = 0.2;

  let lastError: Error = new Error("chat API was not called");
  for (let attempt = 1; attempt <= 3; attempt++) {
    try {
      const response = await fetch(`${CHAT_API_BASE}/chat/completions`, {
        method: "POST",
        headers: { Authorization: `Bearer ${CHAT_API_KEY}`, "Content-Type": "application/json" },
        body: JSON.stringify(body),
        signal: AbortSignal.timeout(600_000),
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) {
        const message = `chat API ${response.status}: ${data?.error?.message ?? "no details"}`;
        if (response.status !== 429 && response.status < 500) throw new FatalApiError(message);
        throw new Error(message);
      }
      usage.calls++;
      usage.prompt += data.usage?.prompt_tokens ?? 0;
      usage.completion += data.usage?.completion_tokens ?? 0;
      usage.reasoning += data.usage?.completion_tokens_details?.reasoning_tokens ?? 0;
      const choice = data.choices?.[0];
      if (choice?.finish_reason === "length") throw new Error("chat response was truncated");
      return JSON.parse(choice?.message?.content ?? "") as T;
    } catch (err) {
      if (err instanceof FatalApiError) throw err;
      lastError = err as Error;
      if (attempt < 3) await new Promise((r) => setTimeout(r, attempt * 15_000));
    }
  }
  throw lastError;
}

// --- Prompts ---

const SUBJECTS =
  "a project or product, a client or business, a person, an organization, a server, account or other infrastructure, a tool, a recurring process, a set of standing rules or preferences, or a recurring topic";

const NAMING =
  `slug: lowercase ASCII kebab-case English, 1-5 words, no dates. title: the subject's real name. category: one of ${CATEGORIES.join(", ")}. summary: one line under 140 characters saying what the subject is.`;

const PLAN_SYSTEM = `You maintain the Open Brain wiki: markdown pages compiled from the captured notes ("thoughts") of Iliya Oblakov's AI sessions. You get the existing page catalog and the opening lines of every new thought. Decide which NEW pages the wiki needs so that these thoughts can be filed. Do not assign thoughts yet.

Rules:
- A page covers one durable subject: ${SUBJECTS}.
- Reuse existing pages. Propose a page only for a subject that appears in several thoughts or carries facts worth keeping.
- Aim for pages that will gather roughly 3 to 40 thoughts. Split a large subject into an overview page plus sub-pages for its distinct durable areas (slug <parent-slug>-<area>, parent set to the overview slug). Fold tiny subjects into the page of the subject they belong to.
- ${NAMING}

Return JSON only:
{"new_pages": [{"slug": "...", "title": "...", "category": "...", "summary": "...", "parent": "<optional slug>"}]}`;

const ROUTE_SYSTEM = `You maintain the Open Brain wiki: markdown pages compiled from the captured notes ("thoughts") of Iliya Oblakov's AI sessions. You get the page catalog and a batch of new thoughts. Assign every thought to the wiki pages it adds durable knowledge to.

Rules:
- A page covers one durable subject: ${SUBJECTS}.
- Use an existing or proposed page whenever one fits. Propose a new page only when none fits and the subject carries facts worth keeping.
- Give each thought one primary page (listed first) and at most two more pages it adds real facts to.
- A thought that starts "SUPERSEDES earlier notes on X" or "RETRACTED" belongs to the page about X.
- When a page already has many sources (n=40 or more) and the thought is about a distinct durable area of it, propose a sub-page with slug <parent-slug>-<area> and set parent.
- One-off events go to the page of the subject they affected, not to a new page. Use an empty page list only for noise (tests, empty or meaningless notes).
- ${NAMING}

Return JSON only:
{"new_pages": [{"slug": "...", "title": "...", "category": "...", "summary": "...", "parent": "<optional slug>"}],
 "assignments": [{"id": 123, "pages": ["primary-slug", "other-slug"]}]}
Every thought id in the batch appears exactly once in assignments.`;

const WRITE_SYSTEM = `You compile one page of the Open Brain wiki, a markdown knowledge base built from the captured notes ("thoughts") of Iliya Oblakov's AI sessions (Claude Code, Codex and others). You get the page's current text (empty for a new page), the list of other pages, and new thoughts for this page, oldest first. Return the complete updated page.

Content rules:
- Use only facts stated in the current page or in the thoughts, and only the parts of a thought that concern this page's subject. No speculation, filler or advice.
- Newer information wins. "SUPERSEDES earlier notes on X" replaces the older claims it names: update the fact, do not list both versions. "RETRACTED" means remove the content it names.
- End every factual bullet or sentence with citations of the thoughts it comes from, like [#123] or [#123][#140]. Keep the citations of facts you keep. Cite only ids you were given.
- Write dates as YYYY-MM-DD (add a time only when it matters). Keep exact names, paths, URLs, versions, numbers and amounts. Keep commit hashes, session ids and file hashes only when someone will need them later (the current main commit, a live deployment), not for superseded intermediate steps. Never include secrets (keys, passwords, tokens), even if a thought contains one.
- Refer to the owner as Iliya (the notes often say "the user").
- Link other wiki pages as [[slug]], only with slugs from the list, where the text mentions that subject. Do not link the page to itself.
- Keep it current and compact, at most about 1200 words: drop intermediate statuses that later notes made obsolete; keep durable facts, decisions, rules and milestone dates.
- Write in English; keep Bulgarian names and terms as written.

Body format (leave out empty sections; no title heading):
<1-3 sentences: what this subject is and why it matters>

## Current state
## Key facts
## Decisions and rules
## Open items
## History
<dated milestones, newest first, one line each>

Return JSON only:
{"title": "...", "summary": "<one line under 140 characters: what it is and its current state>", "tags": ["<category>", "<2-5 short lowercase topic tags>"], "body": "<markdown>"}`;

function catalogLines(catalog: Map<string, Entry>, withCounts: boolean): string {
  const entries = [...catalog.values()].sort((a, b) => a.slug.localeCompare(b.slug));
  if (!entries.length) return "(no pages yet)";
  return entries
    .map((e) =>
      withCounts
        ? `${e.slug} | ${e.category} | ${e.title} | n=${e.count}${e.parent ? ` | parent=${e.parent}` : ""} | ${e.summary}`
        : `${e.slug} | ${e.title}`
    )
    .join("\n");
}

function topics(thought: Thought): string {
  const t = thought.metadata?.topics;
  return Array.isArray(t) ? t.join(", ") : "";
}

function excerpt(thought: Thought, chars: number): string {
  return `[#${thought.id}] ${thought.created_at.toISOString().slice(0, 10)} | topics: ${topics(thought)}\n${oneLine(thought.content, chars)}`;
}

function writePrompt(slug: string, entry: Entry, page: WikiPage | null, part: Thought[], run: Run): string {
  const others = new Map([...run.catalog].filter(([s]) => s !== slug && run.linkable.has(s)));
  const lines = [
    `Page slug: ${slug}`,
    `Category: ${entry.category} (keep it as the first tag)`,
  ];
  if (entry.parent) lines.push(`Parent page: [[${entry.parent}]] (link to it)`);
  if (page) {
    lines.push(
      `Current revision ${page.revision}. Title: ${page.title}. Summary: ${page.summary}. Tags: ${page.tags.join(", ")}`,
      "Current body:",
      "<<<",
      page.body,
      ">>>",
    );
  } else {
    lines.push(`New page. Proposed title: ${entry.title}. Proposed summary: ${entry.summary}`);
  }
  lines.push("", "Other wiki pages (slug | title):", catalogLines(others, false), "");
  lines.push(`New thoughts for this page (${part.length}, oldest first):`);
  for (const t of part) {
    lines.push("", `=== [#${t.id}] captured ${stamp(t.created_at)} ===`, t.content.trim());
  }
  return lines.join("\n");
}

// --- Compile steps ---

function addProposed(raw: unknown, catalog: Map<string, Entry>): void {
  if (!raw || typeof raw !== "object") return;
  const p = raw as Record<string, unknown>;
  const title = typeof p.title === "string" ? p.title.replace(/\s+/g, " ").trim().slice(0, 200) : "";
  let slug = typeof p.slug === "string" ? p.slug.trim().toLowerCase() : "";
  if (slugError(slug)) slug = slugify(slug || title);
  if (!slug || slugError(slug) || catalog.has(slug)) return;
  const category = typeof p.category === "string" && (CATEGORIES as readonly string[]).includes(p.category)
    ? p.category
    : "topic";
  const parent = typeof p.parent === "string" && catalog.has(p.parent) ? p.parent : undefined;
  catalog.set(slug, {
    slug,
    title: title || slug,
    category,
    summary: oneLine(String(p.summary ?? ""), 200),
    count: 0,
    parent,
    proposed: true,
  });
}

async function planPages(thoughts: Thought[], catalog: Map<string, Entry>, model: string): Promise<void> {
  for (let i = 0; i < thoughts.length; i += PLAN_BATCH) {
    const batch = thoughts.slice(i, i + PLAN_BATCH);
    const user = [
      "Existing pages (slug | category | title | n=sources | summary):",
      catalogLines(catalog, true),
      "",
      `New thoughts (${batch.length}):`,
      ...batch.map((t) => excerpt(t, PLAN_EXCERPT_CHARS)),
    ].join("\n");
    const result = await chatJSON<{ new_pages?: unknown[] }>(model, "medium", PLAN_SYSTEM, user);
    const before = catalog.size;
    for (const p of result.new_pages ?? []) addProposed(p, catalog);
    log(`plan: ${catalog.size - before} new page(s) proposed for ${batch.length} thought(s)`);
  }
}

async function routeThoughts(
  thoughts: Thought[],
  catalog: Map<string, Entry>,
  model: string,
): Promise<{ assignments: Map<string, number[]>; noise: number[]; unassigned: number[] }> {
  const assignments = new Map<string, number[]>();
  const noise: number[] = [];
  const unassigned: number[] = [];

  for (let i = 0; i < thoughts.length; i += ROUTE_BATCH) {
    let todo = thoughts.slice(i, i + ROUTE_BATCH);
    for (let pass = 1; pass <= 2 && todo.length; pass++) {
      const user = [
        "Pages (slug | category | title | n=sources so far | summary):",
        catalogLines(catalog, true),
        "",
        `New thoughts (${todo.length}):`,
        ...todo.map((t) => excerpt(t, ROUTE_EXCERPT_CHARS)),
      ].join("\n\n");
      const result = await chatJSON<{ new_pages?: unknown[]; assignments?: { id?: unknown; pages?: unknown }[] }>(
        model,
        "low",
        ROUTE_SYSTEM,
        user,
      );
      for (const p of result.new_pages ?? []) addProposed(p, catalog);
      const ids = new Set(todo.map((t) => t.id));
      const done = new Set<number>();
      for (const a of result.assignments ?? []) {
        const id = Number(a?.id);
        if (!ids.has(id) || done.has(id) || !Array.isArray(a.pages)) continue;
        const pages = [...new Set(a.pages.map((s) => String(s).trim().toLowerCase()))].filter((s) => catalog.has(s)).slice(0, 3);
        if (a.pages.length && !pages.length) continue;
        done.add(id);
        if (!pages.length) noise.push(id);
        for (const slug of pages) {
          assignments.set(slug, [...(assignments.get(slug) ?? []), id]);
          catalog.get(slug)!.count++;
        }
      }
      todo = todo.filter((t) => !done.has(t.id));
    }
    unassigned.push(...todo.map((t) => t.id));
    log(`route: thoughts ${Math.min(i + ROUTE_BATCH, thoughts.length)}/${thoughts.length}, ${assignments.size} page(s) so far`);
  }
  return { assignments, noise, unassigned };
}

function cleanDraft(
  raw: Partial<Draft>,
  ctx: { slug: string; category: string; fallbackTitle: string; allowedIds: Set<number>; linkable: Set<string> },
): Draft {
  let body = typeof raw.body === "string" ? raw.body.trim() : "";
  if (!body) throw new Error(`model returned an empty body for ${ctx.slug}`);
  body = body
    .replace(/^#\s[^\n]*\n+/, "")
    .replace(CITATION_PATTERN, (m, id) => (ctx.allowedIds.has(Number(id)) ? m : ""))
    .replace(LINK_PATTERN, (_m, target: string, label?: string) => {
      const slug = target.trim();
      if (slug !== ctx.slug && ctx.linkable.has(slug)) return label ? `[[${slug}|${label}]]` : `[[${slug}]]`;
      return label?.trim() || slug.replace(/-/g, " ");
    })
    .replace(/[ \t]+\n/g, "\n")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
  const title = typeof raw.title === "string" && raw.title.trim() ? oneLine(raw.title, 200) : ctx.fallbackTitle;
  const summary = typeof raw.summary === "string" ? oneLine(raw.summary, 200) : "";
  const extra = Array.isArray(raw.tags)
    ? raw.tags.filter((t): t is string => typeof t === "string" && !(CATEGORIES as readonly string[]).includes(t.trim().toLowerCase()))
    : [];
  return { title, summary, tags: normalizeTags([ctx.category, ...extra]).slice(0, 7), body };
}

function chunkByChars(thoughts: Thought[], budget: number): Thought[][] {
  const chunks: Thought[][] = [];
  let current: Thought[] = [];
  let size = 0;
  for (const t of thoughts) {
    if (current.length && size + t.content.length > budget) {
      chunks.push(current);
      current = [];
      size = 0;
    }
    current.push(t);
    size += t.content.length;
  }
  if (current.length) chunks.push(current);
  return chunks;
}

async function compilePage(slug: string, ids: number[], run: Run): Promise<"created" | "updated" | "skipped"> {
  const entry = run.catalog.get(slug)!;
  let page = await withClient(pool, (c) => getPage(c, slug));
  const existed = page !== null;
  const known = new Set(page?.sources ?? []);
  const todo = ids.filter((id) => !known.has(id)).sort((a, b) => a - b).map((id) => run.thoughts.get(id)!);
  if (!todo.length) return "skipped";

  for (const part of chunkByChars(todo, CHUNK_CHARS)) {
    for (let attempt = 1;; attempt++) {
      try {
        const raw = await chatJSON<Partial<Draft>>(run.model, "medium", WRITE_SYSTEM, writePrompt(slug, entry, page, part, run));
        const draft = cleanDraft(raw, {
          slug,
          category: entry.category,
          fallbackTitle: page?.title ?? entry.title,
          allowedIds: new Set([...(page?.sources ?? []), ...part.map((t) => t.id)]),
          linkable: run.linkable,
        });
        const result = await writePage(pool, getEmbedding, {
          slug,
          ...draft,
          sources: part.map((t) => t.id),
          expected_revision: page?.revision ?? 0,
          author: "compile",
          note: `compile #${run.id}: +${part.length} thought(s)`,
          compile_run_id: run.id,
        });
        page = result.page;
        break;
      } catch (err) {
        if (err instanceof WikiError && err.message.includes("revision conflict") && attempt < 2) {
          page = await withClient(pool, (c) => getPage(c, slug));
          continue;
        }
        throw err;
      }
    }
  }
  return existed ? "updated" : "created";
}

async function runPool<T>(items: T[], concurrency: number, fn: (item: T) => Promise<void>): Promise<void> {
  const queue = [...items];
  const workers = Array.from({ length: Math.min(concurrency, queue.length) }, async () => {
    for (let item = queue.shift(); item !== undefined; item = queue.shift()) await fn(item);
  });
  await Promise.all(workers);
}

async function lint(): Promise<string[]> {
  return await withClient(pool, async (c) => {
    const findings: string[] = [];
    const dupes = await c.queryObject<{ a: string; b: string; similarity: number }>(
      `SELECT a.slug AS a, b.slug AS b, 1 - (a.embedding <=> b.embedding) AS similarity
       FROM wiki_pages a JOIN wiki_pages b ON a.slug < b.slug
       WHERE a.embedding IS NOT NULL AND b.embedding IS NOT NULL
         AND 1 - (a.embedding <=> b.embedding) >= $1
       ORDER BY similarity DESC LIMIT 10`,
      [DUPLICATE_SIMILARITY],
    );
    if (dupes.rows.length) {
      findings.push(`Possible duplicates: ${dupes.rows.map((r) => `[[${r.a}]] and [[${r.b}]] (${r.similarity.toFixed(2)})`).join(", ")}.`);
    }
    const broken = await c.queryObject<{ slug: string; link: string }>(
      `SELECT p.slug, l AS link FROM wiki_pages p, unnest(p.links) AS l
       WHERE l NOT IN ('index', 'log') AND NOT EXISTS (SELECT 1 FROM wiki_pages q WHERE q.slug = l)
       ORDER BY 1, 2 LIMIT 20`,
    );
    if (broken.rows.length) {
      findings.push(`Broken links: ${broken.rows.map((r) => `[[${r.slug}]] -> ${r.link}`).join(", ")}.`);
    }
    const orphans = await c.queryObject<{ slug: string }>(
      `SELECT p.slug FROM wiki_pages p
       WHERE NOT EXISTS (SELECT 1 FROM wiki_pages q WHERE q.slug <> p.slug AND q.links @> ARRAY[p.slug])
       ORDER BY 1 LIMIT 30`,
    );
    if (orphans.rows.length) {
      findings.push(`Not linked from any other page: ${orphans.rows.map((r) => `[[${r.slug}]]`).join(", ")}.`);
    }
    return findings;
  });
}

async function loadThoughts(after: number, limit: number): Promise<Thought[]> {
  const result = await withClient(pool, (c) =>
    c.queryObject<{ id: bigint; content: string; metadata: Record<string, unknown> | null; created_at: Date }>(
      `SELECT id, content, metadata, created_at FROM thoughts
       WHERE id > $1 AND created_at < now() - make_interval(secs => $3::double precision)
       ORDER BY id LIMIT $2`,
      [after, limit, SETTLE_SECONDS],
    )
  );
  return result.rows.map((r) => ({ ...r, id: Number(r.id), metadata: r.metadata ?? {} }));
}

async function loadCatalog(): Promise<Map<string, Entry>> {
  const pages = await withClient(pool, (c) => listPages(c));
  return new Map(pages.map((p) => [p.slug, {
    slug: p.slug,
    title: p.title,
    category: p.tags.find((t) => (CATEGORIES as readonly string[]).includes(t)) ?? "topic",
    summary: p.summary,
    count: p.source_count,
    proposed: false,
  }]));
}

function usageLine(): string {
  return `${usage.calls} model call(s), ${usage.prompt} prompt + ${usage.completion} completion tokens (${usage.reasoning} reasoning)`;
}

async function compile(opts: Options): Promise<void> {
  await withClient(pool, (c) =>
    c.queryArray(
      "UPDATE wiki_compile_runs SET status = 'failed', finished_at = now(), error = 'abandoned: the process ended before finishing' WHERE status = 'running'",
    )
  );
  const reembedded = opts.dryRun ? 0 : await embedMissing(pool, getEmbedding);
  if (reembedded) log(`re-embedded ${reembedded} page(s) that had no embedding`);
  const status = await withClient(pool, (c) => compileStatus(c));
  const thoughts = await loadThoughts(status.high_water_mark, opts.limit);
  if (!thoughts.length) {
    log(`No new thoughts since the last compile (high-water mark #${status.high_water_mark}).`);
    return;
  }
  const first = thoughts[0].id;
  const last = thoughts[thoughts.length - 1].id;
  log(`${thoughts.length} new thought(s), #${first}-#${last}; model ${opts.model}${opts.dryRun ? "; dry run" : ""}`);

  const catalog = await loadCatalog();
  const previous = await withClient(pool, (c) =>
    c.queryObject<{ unassigned: unknown[] }>(
      "SELECT unassigned FROM wiki_compile_runs WHERE status IN ('ok', 'partial') ORDER BY id DESC LIMIT 1",
    )
  );
  const unassignedBefore = new Set((previous.rows[0]?.unassigned ?? []).map(Number));
  let runId = 0;
  if (!opts.dryRun) {
    const started = await withClient(pool, (c) =>
      c.queryObject<{ id: bigint }>(
        "INSERT INTO wiki_compile_runs (model, from_thought_id, thoughts_seen) VALUES ($1, $2, $3) RETURNING id",
        [opts.model, status.high_water_mark, thoughts.length],
      )
    );
    runId = Number(started.rows[0].id);
  }

  try {
    await planPages(thoughts, catalog, opts.model);
    const routing = await routeThoughts(thoughts, catalog, opts.model);
    const order = [...routing.assignments].sort((a, b) => b[1].length - a[1].length);

    if (opts.dryRun) {
      for (const [slug, ids] of order) {
        const e = catalog.get(slug)!;
        console.log(
          `${e.proposed ? "NEW " : "    "}${slug} | ${e.category}${e.parent ? ` | parent ${e.parent}` : ""} | ${ids.length} thought(s) | ${e.title}: ${e.summary}`,
        );
      }
      const unused = [...catalog.values()].filter((e) => e.proposed && !routing.assignments.has(e.slug)).map((e) => e.slug);
      console.log(`proposed but unused: ${unused.join(", ") || "none"}`);
      console.log(`noise: ${routing.noise.map((id) => `#${id}`).join(", ") || "none"}`);
      console.log(`unassigned: ${routing.unassigned.map((id) => `#${id}`).join(", ") || "none"}`);
      console.log(usageLine());
      return;
    }

    const run: Run = {
      id: runId,
      model: opts.model,
      thoughts: new Map(thoughts.map((t) => [t.id, t])),
      catalog,
      linkable: new Set([...catalog.values()].filter((e) => !e.proposed || routing.assignments.has(e.slug)).map((e) => e.slug)),
    };
    const created: string[] = [];
    const updated: string[] = [];
    const failed: string[] = [];
    const failedIds = new Set<number>();
    await runPool(order, opts.concurrency, async ([slug, ids]) => {
      try {
        const outcome = await compilePage(slug, ids, run);
        if (outcome === "created") created.push(slug);
        if (outcome === "updated") updated.push(slug);
        log(`${outcome} ${slug} (${ids.length} thought(s))`);
      } catch (err) {
        failed.push(slug);
        ids.forEach((id) => failedIds.add(id));
        log(`FAILED ${slug}: ${(err as Error).message}`);
      }
    });

    const findings = await lint();
    // A thought left unassigned by two runs in a row is given up, so it cannot pin the mark forever.
    const givenUp = routing.unassigned.filter((id) => unassignedBefore.has(id));
    const unassigned = routing.unassigned.filter((id) => !unassignedBefore.has(id));
    const retry = [...failedIds, ...unassigned];
    const highWaterMark = retry.length ? Math.min(last, Math.min(...retry) - 1) : last;
    const outcome = retry.length ? "partial" : "ok";
    const summary = [
      `${created.length} created, ${updated.length} updated${failed.length ? `, ${failed.length} failed (${failed.join(", ")})` : ""}.`,
      routing.noise.length ? `Skipped as noise: ${routing.noise.map((id) => `#${id}`).join(", ")}.` : "",
      unassigned.length ? `Not assigned (retried next run): ${unassigned.map((id) => `#${id}`).join(", ")}.` : "",
      givenUp.length ? `Not assigned twice, given up: ${givenUp.map((id) => `#${id}`).join(", ")}.` : "",
      reembedded ? `Re-embedded ${reembedded} page(s).` : "",
      ...findings,
      `Usage: ${usageLine()}.`,
    ].filter(Boolean).join(" ");
    await withClient(pool, (c) =>
      c.queryObject(
        `UPDATE wiki_compile_runs SET status = $2, finished_at = now(), to_thought_id = $3,
           pages_created = $4, pages_updated = $5, summary = $6, unassigned = $7::bigint[] WHERE id = $1`,
        [runId, outcome, highWaterMark, created.length, updated.length, summary, unassigned],
      )
    );
    log(`compile #${runId} ${outcome}: ${summary}`);
  } catch (err) {
    if (runId) {
      await withClient(pool, (c) =>
        c.queryObject(
          "UPDATE wiki_compile_runs SET status = 'failed', finished_at = now(), error = $2 WHERE id = $1",
          [runId, `${(err as Error).message} (${usageLine()})`],
        )
      );
    }
    throw err;
  }
}

async function printStatus(): Promise<void> {
  const { status, pages } = await withClient(pool, async (c) => ({
    status: await compileStatus(c),
    pages: await listPages(c),
  }));
  console.log(`pages: ${pages.length}`);
  console.log(
    status.last_ok
      ? `last compile: run #${status.last_ok.id} ${status.last_ok.status} at ${status.last_ok.finished_at ? stamp(status.last_ok.finished_at) : "?"}, thoughts up to #${status.last_ok.to_thought_id}, model ${status.last_ok.model}`
      : "last compile: never",
  );
  if (status.last_attempt?.status === "failed") {
    console.log(`latest attempt: run #${status.last_attempt.id} failed: ${status.last_attempt.error ?? ""}`);
  }
  console.log(`pending thoughts: ${status.pending}`);
}

function toMarkdown(page: WikiPage): string {
  const front = [
    "---",
    `title: ${JSON.stringify(page.title)}`,
    `summary: ${JSON.stringify(page.summary)}`,
    `tags: [${page.tags.map((t) => JSON.stringify(t)).join(", ")}]`,
    `sources: [${page.sources.join(", ")}]`,
    `revision: ${page.revision}`,
    `updated: ${page.updated_at.toISOString()}`,
    "---",
  ];
  return `${front.join("\n")}\n\n# ${page.title}\n\n${page.body}\n`;
}

async function exportWiki(dir: string): Promise<void> {
  const { pages, index, changes } = await withClient(pool, async (c) => {
    const full: WikiPage[] = [];
    for (const p of await listPages(c)) full.push((await getPage(c, p.slug))!);
    return { pages: full, index: await buildIndex(c), changes: await renderLog(c, 500) };
  });
  await Deno.mkdir(dir, { recursive: true });
  for (const page of pages) await Deno.writeTextFile(`${dir}/${page.slug}.md`, toMarkdown(page));
  await Deno.writeTextFile(`${dir}/index.md`, `${index}\n`);
  await Deno.writeTextFile(`${dir}/log.md`, `${changes}\n`);
  log(`exported ${pages.length} page(s) plus index.md and log.md to ${dir}`);
}

async function main(): Promise<void> {
  const opts = parseOptions(Deno.args);
  if (opts.help) {
    console.log(USAGE);
    return;
  }
  await ensureWikiSchema(pool);
  if (opts.status) return await printStatus();
  if (opts.exportDir) return await exportWiki(opts.exportDir);
  if (!CHAT_API_KEY) {
    throw new Error("no chat API key configured (CHAT_API_KEY or EMBEDDING_API_KEY); compile by hand through the MCP tools instead (wiki_read 'pending', then wiki_write)");
  }

  const lock = await pool.connect();
  try {
    const got = await lock.queryObject<{ ok: boolean }>("SELECT pg_try_advisory_lock($1) AS ok", [COMPILE_LOCK]);
    if (!got.rows[0]?.ok) {
      log("another compile is running; nothing to do");
      return;
    }
    try {
      await compile(opts);
    } finally {
      await lock.queryObject("SELECT pg_advisory_unlock($1)", [COMPILE_LOCK]).catch(() => {});
    }
  } finally {
    lock.release();
  }
  log(`done; ${usageLine()}`);
}

if (import.meta.main) {
  try {
    await main();
    await pool.end();
  } catch (err) {
    console.error(`wiki_compile: ${(err as Error).message}`);
    await pool.end().catch(() => {});
    Deno.exit(1);
  }
}
