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
  extractCitations,
  getPage,
  LINK_PATTERN,
  listPages,
  MERGED_TAG,
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
const CHUNK_CHARS = 160_000;
const DUPLICATE_SIMILARITY = 0.9;
// An update may drop fewer than max(this, a quarter of the page's cited thoughts) citations.
const DROP_LIMIT = 4;
// Exit status when a page failed: the systemd unit then fails, which the sites-1 watchdog reports.
const FAILED_PAGES_EXIT = 3;
const RELATED_LIMIT = 20;
const relatedFloor = Number(env("WIKI_COMPILE_RELATED_FLOOR", "0.6"));
const RELATED_FLOOR = Number.isFinite(relatedFloor) ? relatedFloor : 0.6;
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
  --rebuild SLUGS     rewrite pages from all their sources: comma-separated slugs or "all" (--dry-run prints them)
  --export DIR        write every page plus index.md and log.md to DIR (needs --allow-write=DIR)
  --help              show this text

Pages tagged "merged" (pointers left after a merge) take no new thoughts and are not linked or rebuilt.
Exit status: 0 done, 1 error, 3 some pages failed (their thoughts stay pending for the next run).`;

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
  rebuild: string | null;
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
  maxThoughtId: number;
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
    rebuild: null,
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
      case "--rebuild": opts.rebuild = value(); break;
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

const WRITE_SYSTEM = `You compile one page of the Open Brain wiki, a markdown knowledge base built from the captured notes ("thoughts") of Iliya Oblakov's AI sessions (Claude Code, Codex and others). You get the page's current text (empty for a new page), the list of other pages, and thoughts for this page, oldest first. Return the complete updated page.

Content rules:
- Use only facts stated in the current page or in the thoughts, and only the parts of a thought that concern this page's subject. No speculation, filler or advice.
- Unless the request says it is a rebuild, the current page was already checked against its sources: edit it, do not rewrite it. Add the new facts, change only the lines that a new thought updates, supersedes or retracts, add History entries, and copy every other line exactly as it is, with its citations.
- Newer information wins. "SUPERSEDES earlier notes on X" replaces the older claims it names: update the fact, do not present both versions as current. When the old value still matters to a reader (a price, a fee, a date, a reversed decision), keep it once, marked, for example "79 EUR a month (an earlier note said 80 EUR) [#526]". "RETRACTED" means remove the content it names.
- End every factual bullet or sentence with citations of the thoughts it comes from, like [#123] or [#123][#140]. Keep the citations of facts you keep. Cite only ids you were given.
- Write dates as YYYY-MM-DD (add a time only when it matters). Keep exact names, paths, URLs, versions, numbers and amounts. Keep commit hashes, session ids and file hashes only when someone will need them later (the current main commit, a live deployment), not for superseded intermediate steps. Never include secrets (keys, passwords, tokens), even if a thought contains one.
- Iliya is the person these notes belong to. Notes often say "the user": write "Iliya" only where the note names him or it is certain, otherwise keep "the user". In client projects "the owner" and "the client" mean the client's owner, never Iliya. Keep who decided, who asked and who did what exactly as the note says.
- Date history entries by when the event happened (the note's "As of" time or a date in its text, Sofia time), not by when a later note mentions it.
- Collapse a chain of status notes (PR numbers, commit heads, render stages, review rounds) into the current state plus one History line per milestone.
- Sub-pages hold the detail and status of their areas: for an area that has a sub-page, keep one summary line with its [[slug]] link and do not repeat its status.
- Thoughts marked "related" were filed under other pages: use them to correct or complete facts about this page's subject (the newest wins) and cite them when you use them.
- Before answering, check each Current state and Open items line against the newest thought on that point, own or related. If a later thought says it changed, was done or was fixed, update or remove the line.
- Link other wiki pages as [[slug]], only with slugs from the list, where the text mentions that subject. Do not link the page to itself.
- Keep every durable fact already on the page unless a newer thought supersedes or retracts it: new thoughts add to the page, they do not replace it. Durable means it still matters next month: what the subject is, where it lives (paths, URLs, servers, repositories), who is involved, prices and terms, decisions, rules and milestones.
- Drop intermediate statuses that later notes made obsolete (a PR that was opened and later merged needs only the merge).
- Length follows the material: about 300 words for a small subject, up to about 1800 words for a large one with many thoughts.
- Write in English; keep Bulgarian names and terms as written. Do not use em dashes or en dashes: use a comma, a colon, parentheses or a plain hyphen.

Body format (leave out empty sections; no title heading):
<1-3 sentences: what this subject is and why it matters>

## Current state
## Key facts
## Decisions and rules
## Open items
## History
<dated milestones over the whole life of the subject, newest first, one line each; fold routine steps of the same week into one line instead of dropping them>

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

type PageText = Pick<WikiPage, "title" | "summary" | "tags" | "body">;

function writePrompt(
  slug: string,
  entry: Entry,
  page: PageText | null,
  part: Thought[],
  run: Run,
  opts: { total?: number; rebuild?: boolean; related?: Thought[] } = {},
): string {
  const others = new Map([...run.catalog].filter(([s]) => s !== slug && run.linkable.has(s)));
  const lines = [
    `Page slug: ${slug}`,
    `Category: ${entry.category} (keep it as the first tag)`,
  ];
  if (entry.parent) lines.push(`Parent page: [[${entry.parent}]] (link to it)`);
  if (opts.rebuild) {
    lines.push("Rebuild: the current text may be incomplete. Rebuild the page from all of its thoughts below, and keep corrections from the current text unless a thought contradicts them.");
  }
  if (opts.total && opts.total > part.length) {
    lines.push(`This page draws on ${opts.total} thoughts in total; this request contains ${part.length} of them.`);
  }
  if (page) {
    lines.push(
      `Current title: ${page.title}. Summary: ${page.summary}. Tags: ${page.tags.join(", ")}`,
      "Current body:",
      "<<<",
      page.body,
      ">>>",
    );
  } else {
    lines.push(`New page. Proposed title: ${entry.title}. Proposed summary: ${entry.summary}`);
  }
  lines.push("", "Other wiki pages (slug | title):", catalogLines(others, false), "");
  const children = [...run.catalog.values()].filter((e) =>
    e.slug.startsWith(`${slug}-`) && ![...run.catalog.keys()].some((o) => o !== e.slug && o.startsWith(`${slug}-`) && e.slug.startsWith(`${o}-`))
  );
  if (children.length) {
    lines.push("Sub-pages of this page (slug | title | summary):", ...children.map((e) => `${e.slug} | ${e.title} | ${e.summary}`), "");
  }
  const related = new Set((opts.related ?? []).map((t) => t.id));
  const stream = [...part, ...(opts.related ?? [])].sort((a, b) => a.id - b.id);
  lines.push(
    `Thoughts (${part.length} for this page${related.size ? `, ${related.size} related ones filed under other pages` : ""}), oldest first:`,
  );
  for (const t of stream) {
    const label = related.has(t.id) ? " (related: filed under another page)" : "";
    lines.push("", `=== [#${t.id}] captured ${stamp(t.created_at)}${label} ===`, t.content.trim());
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
    .replace(/^(\s*-\s+\d{4}-\d{2}-\d{2}[^\n\u2013\u2014]*?)\s+[\u2013\u2014]\s+/gm, "$1: ")
    .replace(/(\d)\s*[\u2013\u2014]\s*(\d)/g, "$1-$2")
    .replace(/\s+[\u2013\u2014]\s+/g, ", ")
    .replace(/[\u2013\u2014]/g, "-")
    .replace(/[ \t]+\n/g, "\n")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
  const undash = (s: string) => s.replace(/\s+[\u2013\u2014]\s+/g, ", ").replace(/[\u2013\u2014]/g, "-");
  const title = typeof raw.title === "string" && raw.title.trim() ? undash(oneLine(raw.title, 200)) : ctx.fallbackTitle;
  const summary = typeof raw.summary === "string" ? undash(oneLine(raw.summary, 200)) : "";
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

async function relatedThoughts(slug: string, exclude: number[], maxId: number): Promise<Thought[]> {
  const result = await withClient(pool, (c) =>
    c.queryObject<{ id: bigint; content: string; metadata: Record<string, unknown> | null; created_at: Date; similarity: unknown }>(
      `SELECT t.id, t.content, t.metadata, t.created_at, 1 - (t.embedding <=> p.embedding) AS similarity
       FROM thoughts t JOIN wiki_pages p ON p.slug = $1
       WHERE p.embedding IS NOT NULL AND t.embedding IS NOT NULL AND t.id <= $3 AND NOT (t.id = ANY($2::bigint[]))
       ORDER BY t.embedding <=> p.embedding LIMIT $4`,
      [slug, exclude, maxId, RELATED_LIMIT],
    )
  );
  return result.rows
    .filter((r) => Number(r.similarity) >= RELATED_FLOOR)
    .map((r) => ({ id: Number(r.id), content: r.content, metadata: r.metadata ?? {}, created_at: r.created_at }))
    .sort((a, b) => a.id - b.id);
}

function droppedCitations(before: string, after: string): number[] {
  const kept = new Set(extractCitations(after));
  return extractCitations(before).filter((id) => !kept.has(id));
}

async function draftUpdate(
  slug: string,
  entry: Entry,
  page: WikiPage | null,
  part: Thought[],
  run: Run,
  related: Thought[],
): Promise<Draft> {
  const allowedIds = new Set([...(page?.sources ?? []), ...part.map((t) => t.id), ...related.map((t) => t.id)]);
  let prompt = writePrompt(slug, entry, page, part, run, { related });
  for (let attempt = 1;; attempt++) {
    const raw = await chatJSON<Partial<Draft>>(run.model, "medium", WRITE_SYSTEM, prompt);
    const draft = cleanDraft(raw, { slug, category: entry.category, fallbackTitle: page?.title ?? entry.title, allowedIds, linkable: run.linkable });
    if (!page) return draft;
    // A routine update must not quietly drop facts a checked page already cites: retry once, then leave the page alone.
    const lost = droppedCitations(page.body, draft.body);
    if (lost.length < Math.max(DROP_LIMIT, Math.ceil(extractCitations(page.body).length / 4))) return draft;
    if (attempt === 2) {
      throw new Error(
        `the update dropped ${lost.length} cited thoughts twice; page left unchanged. Fold its pending thoughts in by hand with wiki_write; once the page cites them, the next run moves on`,
      );
    }
    log(`${slug}: update dropped ${lost.length} cited thoughts; asking again`);
    prompt += `\n\nYour previous answer removed the facts cited as ${lost.map((id) => `[#${id}]`).join("")}. Return the complete page again and keep those facts unless a thought above supersedes or retracts them.`;
  }
}

async function compilePage(slug: string, ids: number[], run: Run): Promise<"created" | "updated" | "skipped"> {
  const entry = run.catalog.get(slug)!;
  let page = await withClient(pool, (c) => getPage(c, slug));
  const existed = page !== null;
  const known = new Set(page?.sources ?? []);
  const todo = ids.filter((id) => !known.has(id)).sort((a, b) => a - b).map((id) => run.thoughts.get(id)!);
  if (!todo.length) return "skipped";
  const related = await relatedThoughts(slug, [...known, ...ids], run.maxThoughtId);

  for (const part of chunkByChars(todo, CHUNK_CHARS)) {
    for (let attempt = 1;; attempt++) {
      try {
        const draft = await draftUpdate(slug, entry, page, part, run, related);
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

async function loadThoughtsById(ids: number[]): Promise<Thought[]> {
  const result = await withClient(pool, (c) =>
    c.queryObject<{ id: bigint; content: string; metadata: Record<string, unknown> | null; created_at: Date }>(
      "SELECT id, content, metadata, created_at FROM thoughts WHERE id = ANY($1::bigint[]) ORDER BY id",
      [ids],
    )
  );
  return result.rows.map((r) => ({ ...r, id: Number(r.id), metadata: r.metadata ?? {} }));
}

async function rebuildPage(slug: string, run: Run, dryRun: boolean): Promise<void> {
  const page = (await withClient(pool, (c) => getPage(c, slug)))!;
  const entry = run.catalog.get(slug)!;
  const thoughts = await loadThoughtsById(page.sources);
  const related = await relatedThoughts(slug, page.sources, run.maxThoughtId);
  let current: PageText = page;
  for (const part of chunkByChars(thoughts, CHUNK_CHARS)) {
    const raw = await chatJSON<Partial<Draft>>(
      run.model,
      "medium",
      WRITE_SYSTEM,
      writePrompt(slug, entry, current, part, run, { total: thoughts.length, rebuild: current === page, related }),
    );
    current = cleanDraft(raw, {
      slug,
      category: entry.category,
      fallbackTitle: page.title,
      allowedIds: new Set([...page.sources, ...related.map((t) => t.id)]),
      linkable: run.linkable,
    });
  }
  if (dryRun) {
    console.log(`\n===== ${slug} =====\n# ${current.title}\nsummary: ${current.summary}\ntags: ${current.tags.join(", ")}\n\n${current.body}`);
    return;
  }
  await writePage(pool, getEmbedding, {
    slug,
    ...current,
    expected_revision: page.revision,
    author: "compile",
    note: `compile #${run.id}: rebuilt from ${thoughts.length} thought(s)`,
    compile_run_id: run.id,
  });
}

async function rebuild(target: string, opts: Options): Promise<number> {
  const catalog = await loadCatalog();
  const slugs = target === "all" ? [...catalog.keys()] : target.split(",").map((s) => s.trim()).filter(Boolean);
  const unknown = slugs.filter((s) => !catalog.has(s));
  if (unknown.length) throw new Error(`no such page: ${unknown.join(", ")}`);

  const status = await withClient(pool, (c) => compileStatus(c));
  let runId = 0;
  if (!opts.dryRun) {
    const started = await withClient(pool, (c) =>
      c.queryObject<{ id: bigint }>(
        `INSERT INTO wiki_compile_runs (model, from_thought_id, to_thought_id, unassigned)
         SELECT $1, $2, $2, COALESCE((SELECT unassigned FROM wiki_compile_runs WHERE status IN ('ok', 'partial') ORDER BY id DESC LIMIT 1), '{}')
         RETURNING id`,
        [opts.model, status.high_water_mark],
      )
    );
    runId = Number(started.rows[0].id);
  }
  const run: Run = {
    id: runId,
    model: opts.model,
    thoughts: new Map(),
    catalog,
    linkable: new Set(catalog.keys()),
    maxThoughtId: status.high_water_mark,
  };
  const done: string[] = [];
  const failed: string[] = [];
  await runPool(slugs, opts.concurrency, async (slug) => {
    try {
      await rebuildPage(slug, run, opts.dryRun);
      done.push(slug);
      log(`rebuilt ${slug}`);
    } catch (err) {
      failed.push(slug);
      log(`FAILED ${slug}: ${(err as Error).message}`);
    }
  });
  if (opts.dryRun) return failed.length;

  const findings = await lint().catch((err) => [`Lint failed: ${(err as Error).message}.`]);
  const summary = [
    `Rebuilt ${done.length} page(s) from their sources${failed.length ? `; ${failed.length} failed (${failed.join(", ")})` : ""}.`,
    ...findings,
    `Usage: ${usageLine()}.`,
  ].join(" ");
  await withClient(pool, (c) =>
    c.queryObject(
      "UPDATE wiki_compile_runs SET status = $2, finished_at = now(), pages_updated = $3, summary = $4 WHERE id = $1",
      [runId, failed.length ? "partial" : "ok", done.length, summary],
    )
  );
  log(`rebuild #${runId} ${failed.length ? "partial" : "ok"}: ${summary}`);
  return failed.length;
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
         AND NOT ($2 = ANY(a.tags)) AND NOT ($2 = ANY(b.tags))
         AND 1 - (a.embedding <=> b.embedding) >= $1
       ORDER BY similarity DESC LIMIT 10`,
      [DUPLICATE_SIMILARITY, MERGED_TAG],
    );
    if (dupes.rows.length) {
      findings.push(`Possible duplicates: ${dupes.rows.map((r) => `[[${r.a}]] and [[${r.b}]] (${Number(r.similarity).toFixed(2)})`).join(", ")}.`);
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
       WHERE NOT ($1 = ANY(p.tags))
         AND NOT EXISTS (SELECT 1 FROM wiki_pages q WHERE q.slug <> p.slug AND q.links @> ARRAY[p.slug])
         AND NOT EXISTS (SELECT 1 FROM wiki_pages q WHERE p.slug LIKE q.slug || '-%')
       ORDER BY 1 LIMIT 30`,
      [MERGED_TAG],
    );
    if (orphans.rows.length) {
      findings.push(`Top-level pages no other page links to: ${orphans.rows.map((r) => `[[${r.slug}]]`).join(", ")}.`);
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
  return new Map(pages.filter((p) => !p.tags.includes(MERGED_TAG)).map((p) => [p.slug, {
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

async function compile(opts: Options): Promise<number> {
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
    return 0;
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
      return 0;
    }

    const run: Run = {
      id: runId,
      model: opts.model,
      thoughts: new Map(thoughts.map((t) => [t.id, t])),
      catalog,
      linkable: new Set([...catalog.values()].filter((e) => !e.proposed || routing.assignments.has(e.slug)).map((e) => e.slug)),
      maxThoughtId: last,
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

    const findings = await lint().catch((err) => [`Lint failed: ${(err as Error).message}.`]);
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
    return failed.length;
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

async function main(): Promise<number> {
  const opts = parseOptions(Deno.args);
  if (opts.help) {
    console.log(USAGE);
    return 0;
  }
  await ensureWikiSchema(pool);
  if (opts.status) {
    await printStatus();
    return 0;
  }
  if (opts.exportDir) {
    await exportWiki(opts.exportDir);
    return 0;
  }
  if (!CHAT_API_KEY) {
    throw new Error("no chat API key configured (CHAT_API_KEY or EMBEDDING_API_KEY); compile by hand through the MCP tools instead (wiki_read 'pending', then wiki_write)");
  }

  let failedPages = 0;
  const lock = await pool.connect();
  try {
    const got = await lock.queryObject<{ ok: boolean }>("SELECT pg_try_advisory_lock($1) AS ok", [COMPILE_LOCK]);
    if (!got.rows[0]?.ok) {
      log("another compile is running; nothing to do");
      return 0;
    }
    try {
      failedPages = opts.rebuild ? await rebuild(opts.rebuild, opts) : await compile(opts);
    } finally {
      await lock.queryObject("SELECT pg_advisory_unlock($1)", [COMPILE_LOCK]).catch(() => {});
    }
  } finally {
    lock.release();
  }
  log(`done; ${usageLine()}`);
  return failedPages ? FAILED_PAGES_EXIT : 0;
}

if (import.meta.main) {
  try {
    const code = await main();
    await pool.end();
    if (code) Deno.exit(code);
  } catch (err) {
    console.error(`wiki_compile: ${(err as Error).message}`);
    await pool.end().catch(() => {});
    Deno.exit(1);
  }
}
