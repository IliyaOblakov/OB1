// Open Brain wiki: compiled markdown pages stored next to the thoughts table,
// exposed as MCP tools on the same endpoint (see wiki_schema.sql, wiki_compile.ts).

import type { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { z } from "zod";
import type { Pool, PoolClient } from "postgres";

export type Embedder = (text: string) => Promise<number[]>;

export type WikiPage = {
  slug: string;
  title: string;
  summary: string;
  body: string;
  tags: string[];
  links: string[];
  sources: number[];
  revision: number;
  created_at: Date;
  updated_at: Date;
  updated_by: string;
};

export type PageListing = {
  slug: string;
  title: string;
  summary: string;
  tags: string[];
  revision: number;
  updated_at: Date;
  updated_by: string;
  source_count: number;
};

export type WriteInput = {
  slug: string;
  title?: string;
  summary?: string;
  body?: string;
  tags?: string[];
  sources?: number[];
  drop_sources?: number[];
  append?: boolean;
  expected_revision?: number;
  author?: string;
  note?: string;
  compile_run_id?: number | null;
};

export type WriteResult = {
  status: "created" | "updated" | "unchanged";
  page: WikiPage;
  missing_links: string[];
};

export type CompileStatus = {
  last_ok: {
    id: number;
    status: string;
    finished_at: Date | null;
    to_thought_id: number;
    model: string;
  } | null;
  last_attempt: { id: number; status: string; started_at: Date; error: string | null } | null;
  high_water_mark: number;
  pending: number;
};

export class WikiError extends Error {}

export const CATEGORIES = [
  "project",
  "client",
  "person",
  "organization",
  "infrastructure",
  "tool",
  "process",
  "rules",
  "topic",
] as const;

const CATEGORY_HEADINGS: Record<string, string> = {
  project: "Projects",
  client: "Clients",
  person: "People",
  organization: "Organizations",
  infrastructure: "Infrastructure",
  tool: "Tools",
  process: "Processes",
  rules: "Rules and preferences",
  topic: "Topics",
  other: "Other",
};

export const VIRTUAL_PAGES = ["index", "log", "pending"];

const SLUG_PATTERN = /^[a-z0-9]+(-[a-z0-9]+)*$/;
export const LINK_PATTERN = /\[\[([^\]|\n]+)(?:\|([^\]\n]*))?\]\]/g;
export const CITATION_PATTERN = /\[#(\d+)\]/g;

const LIMITS = { slug: 80, title: 200, summary: 300, body: 100_000, tags: 12, tag: 60 };
const SCHEMA_LOCK = 4_271_311;
const EMBED_CHARS = 10_000;
const SEMANTIC_FLOOR = 0.25;

export const WIKI_INSTRUCTIONS = [
  "Open Brain has two layers. Thoughts (capture_thought, search_thoughts, list_thoughts) are the append-only raw notes and the source of truth.",
  "The wiki (wiki_index, wiki_search, wiki_read) is compiled from thoughts into one page per project, client, person, system or rule set, current state first, with [#id] citations.",
  "For background on a subject, read the wiki first, then search_thoughts for anything newer than the page's updated date.",
  "Record new facts with capture_thought; use wiki_write only to correct or restructure a page.",
].join(" ");

// --- Pure helpers ---

export function slugError(slug: string): string | null {
  if (!slug) return "slug is required";
  if (slug.length > LIMITS.slug || !SLUG_PATTERN.test(slug)) {
    return `invalid slug "${slug}": use lowercase ASCII words joined by hyphens, at most ${LIMITS.slug} characters (for example "hotel-sofia")`;
  }
  if (VIRTUAL_PAGES.includes(slug)) return `"${slug}" is a built-in page and cannot be written`;
  return null;
}

export function slugify(text: string): string {
  return text
    .normalize("NFKD")
    .replace(/[\u0300-\u036f]/g, "")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, LIMITS.slug)
    .replace(/-+$/, "");
}

export function extractLinks(body: string): string[] {
  const links = new Set<string>();
  for (const m of body.matchAll(LINK_PATTERN)) {
    const slug = m[1].trim();
    if (SLUG_PATTERN.test(slug)) links.add(slug);
  }
  return [...links].sort();
}

export function extractCitations(body: string): number[] {
  const ids = new Set<number>();
  for (const m of body.matchAll(CITATION_PATTERN)) ids.add(Number(m[1]));
  return [...ids].sort((a, b) => a - b);
}

export function normalizeTags(tags: string[]): string[] {
  const out: string[] = [];
  for (const tag of tags) {
    const value = String(tag).trim().toLowerCase().replace(/\s+/g, " ").slice(0, LIMITS.tag);
    if (value && !out.includes(value)) out.push(value);
  }
  return out.slice(0, LIMITS.tags);
}

export function categoryOf(tags: string[]): string {
  return tags.find((t) => (CATEGORIES as readonly string[]).includes(t)) ?? "other";
}

export function fuseRankings(lists: string[][], k = 60): { slug: string; score: number }[] {
  const scores = new Map<string, number>();
  for (const list of lists) {
    list.forEach((slug, rank) => scores.set(slug, (scores.get(slug) ?? 0) + 1 / (k + rank + 1)));
  }
  return [...scores]
    .map(([slug, score]) => ({ slug, score }))
    .sort((a, b) => b.score - a.score || a.slug.localeCompare(b.slug));
}

function uniqueSortedIds(ids: number[]): number[] {
  return [...new Set(ids.filter((id) => Number.isSafeInteger(id) && id > 0))].sort((a, b) => a - b);
}

function toIds(value: unknown): number[] {
  return Array.isArray(value) ? value.map((v) => Number(v)) : [];
}

function day(value: Date | string): string {
  return new Date(value).toISOString().slice(0, 10);
}

export function stamp(value: Date | string): string {
  return `${new Date(value).toISOString().slice(0, 16).replace("T", " ")} UTC`;
}

export function oneLine(text: string, max: number): string {
  const flat = text.replace(/\s+/g, " ").trim();
  return flat.length > max ? `${flat.slice(0, max - 3).trimEnd()}...` : flat;
}

export function renderIndex(pages: PageListing[], status: CompileStatus): string {
  const lines = [
    "# Open Brain wiki",
    "",
    "Compiled from Open Brain thoughts: one page per durable subject, current state first, facts cited as [#thought-id]. Read a page with wiki_read, search with wiki_search; `log` lists recent changes and `pending` the thoughts no page covers yet.",
    "",
    `${pages.length} page(s). ${compileLine(status)}`,
  ];

  // A page whose slug extends another page's slug ("hotel-sofia-payments") is listed under it.
  const slugs = new Set(pages.map((p) => p.slug));
  const parentOf = (slug: string) => {
    for (let cut = slug.lastIndexOf("-"); cut > 0; cut = slug.lastIndexOf("-", cut - 1)) {
      if (slugs.has(slug.slice(0, cut))) return slug.slice(0, cut);
    }
    return null;
  };
  const children = new Map<string, PageListing[]>();
  const groups = new Map<string, PageListing[]>();
  for (const page of pages) {
    const parent = parentOf(page.slug);
    const key = parent ?? categoryOf(page.tags);
    const target = parent ? children : groups;
    target.set(key, [...(target.get(key) ?? []), page]);
  }
  const byTitle = (a: PageListing, b: PageListing) => a.title.localeCompare(b.title);
  const entry = (page: PageListing, depth: number): void => {
    lines.push(`${"  ".repeat(depth)}- [[${page.slug}]] ${page.title}${page.summary ? `: ${page.summary}` : ""}`);
    for (const child of (children.get(page.slug) ?? []).sort(byTitle)) entry(child, depth + 1);
  };
  for (const category of [...CATEGORIES, "other"]) {
    const group = groups.get(category);
    if (!group?.length) continue;
    lines.push("", `## ${CATEGORY_HEADINGS[category]} (${group.length})`);
    for (const page of group.sort(byTitle)) entry(page, 0);
  }

  const recent = [...pages].sort((a, b) => +b.updated_at - +a.updated_at).slice(0, 10);
  if (recent.length) {
    lines.push("", "## Recently updated");
    for (const page of recent) {
      lines.push(`- ${day(page.updated_at)} [[${page.slug}]] rev ${page.revision} by ${page.updated_by}`);
    }
  }
  return lines.join("\n");
}

function compileLine(status: CompileStatus): string {
  const parts: string[] = [];
  if (status.last_ok) {
    const run = status.last_ok;
    parts.push(
      `Last compile: ${run.finished_at ? stamp(run.finished_at) : "unfinished"} (run #${run.id}, ${run.status}, thoughts up to #${run.to_thought_id}).`,
    );
  } else {
    parts.push("Not compiled yet.");
  }
  if (status.last_attempt && status.last_attempt.status === "failed" && status.last_attempt.id > (status.last_ok?.id ?? 0)) {
    parts.push(`The latest compile attempt (run #${status.last_attempt.id}, ${stamp(status.last_attempt.started_at)}) failed.`);
  }
  parts.push(
    status.pending
      ? `${status.pending} thought(s) captured since then are not in the wiki yet; use search_thoughts for those.`
      : "No newer thoughts are waiting.",
  );
  return parts.join(" ");
}

// --- Database access ---

let schemaReady: Promise<void> | null = null;

export function ensureWikiSchema(pool: Pool): Promise<void> {
  schemaReady ??= (async () => {
    const sql = await Deno.readTextFile(new URL("./wiki_schema.sql", import.meta.url));
    await withClient(pool, async (client) => {
      try {
        await client.queryArray(`BEGIN; SELECT pg_advisory_xact_lock(${SCHEMA_LOCK}); ${sql}; COMMIT;`);
      } catch (err) {
        await client.queryArray("ROLLBACK").catch(() => {});
        throw err;
      }
    });
  })().catch((err) => {
    schemaReady = null;
    throw err;
  });
  return schemaReady;
}

export async function withClient<T>(pool: Pool, fn: (client: PoolClient) => Promise<T>): Promise<T> {
  const client = await pool.connect();
  try {
    return await fn(client);
  } finally {
    client.release();
  }
}

type PageRow = Omit<WikiPage, "sources"> & { sources: unknown };

const PAGE_COLUMNS =
  "slug, title, summary, body, tags, links, sources, revision, created_at, updated_at, updated_by";

export async function getPage(client: PoolClient, slug: string): Promise<WikiPage | null> {
  const result = await client.queryObject<PageRow>(
    `SELECT ${PAGE_COLUMNS} FROM wiki_pages WHERE slug = $1`,
    [slug],
  );
  const row = result.rows[0];
  return row ? { ...row, sources: toIds(row.sources) } : null;
}

export async function listPages(
  client: PoolClient,
  opts: { tag?: string; days?: number; sort?: "slug" | "updated"; limit?: number; offset?: number } = {},
): Promise<PageListing[]> {
  const where: string[] = [];
  const params: unknown[] = [];
  if (opts.tag) {
    params.push(opts.tag.trim().toLowerCase());
    where.push(`tags @> ARRAY[$${params.length}::text]`);
  }
  if (opts.days) {
    params.push(Math.max(1, Math.floor(opts.days)));
    where.push(`updated_at >= now() - make_interval(days => $${params.length}::int)`);
  }
  params.push(opts.limit ?? 10_000, opts.offset ?? 0);
  const result = await client.queryObject<PageListing>(
    `SELECT slug, title, summary, tags, revision, updated_at, updated_by,
            cardinality(sources)::int AS source_count
     FROM wiki_pages
     ${where.length ? `WHERE ${where.join(" AND ")}` : ""}
     ORDER BY ${opts.sort === "updated" ? "updated_at DESC, slug" : "slug"}
     LIMIT $${params.length - 1} OFFSET $${params.length}`,
    params,
  );
  return result.rows;
}

export async function backlinks(client: PoolClient, slug: string): Promise<string[]> {
  const result = await client.queryObject<{ slug: string }>(
    "SELECT slug FROM wiki_pages WHERE links @> ARRAY[$1::text] AND slug <> $1 ORDER BY slug",
    [slug],
  );
  return result.rows.map((r) => r.slug);
}

async function missingSlugs(client: PoolClient, slugs: string[]): Promise<string[]> {
  if (!slugs.length) return [];
  const result = await client.queryObject<{ slug: string }>(
    "SELECT slug FROM wiki_pages WHERE slug = ANY($1::text[])",
    [slugs],
  );
  const found = new Set(result.rows.map((r) => r.slug));
  return slugs.filter((s) => !found.has(s) && !VIRTUAL_PAGES.includes(s));
}

export async function compileStatus(client: PoolClient): Promise<CompileStatus> {
  const ok = await client.queryObject<{
    id: bigint;
    status: string;
    finished_at: Date | null;
    to_thought_id: bigint;
    model: string;
  }>(
    `SELECT id, status, finished_at, to_thought_id, model FROM wiki_compile_runs
     WHERE status IN ('ok', 'partial') ORDER BY id DESC LIMIT 1`,
  );
  const attempt = await client.queryObject<{ id: bigint; status: string; started_at: Date; error: string | null }>(
    "SELECT id, status, started_at, error FROM wiki_compile_runs ORDER BY id DESC LIMIT 1",
  );
  const lastOk = ok.rows[0];
  const highWaterMark = lastOk ? Number(lastOk.to_thought_id) : 0;
  const pending = await client.queryObject<{ count: number }>(
    "SELECT COUNT(*)::int AS count FROM thoughts WHERE id > $1",
    [highWaterMark],
  );
  const lastAttempt = attempt.rows[0];
  return {
    last_ok: lastOk
      ? { ...lastOk, id: Number(lastOk.id), to_thought_id: Number(lastOk.to_thought_id) }
      : null,
    last_attempt: lastAttempt ? { ...lastAttempt, id: Number(lastAttempt.id) } : null,
    high_water_mark: highWaterMark,
    pending: pending.rows[0]?.count ?? 0,
  };
}

type PageFields = Pick<WikiPage, "title" | "summary" | "body" | "tags" | "links" | "sources">;

function mergePage(slug: string, current: WikiPage | null, input: WriteInput): PageFields {
  if (!current && (!input.title?.trim() || !input.body?.trim())) {
    throw new WikiError(`page "${slug}" does not exist yet: title and body are required to create it`);
  }
  const title = (input.title ?? current?.title ?? "").replace(/\s+/g, " ").trim();
  const summary = (input.summary ?? current?.summary ?? "").replace(/\s+/g, " ").trim();
  let body = input.body ?? current?.body ?? "";
  if (input.append && current && input.body !== undefined) {
    body = `${current.body.trimEnd()}\n\n${input.body.trim()}`;
  }
  body = body.trim();
  const tags = input.tags ? normalizeTags(input.tags) : current?.tags ?? [];

  if (!title || title.length > LIMITS.title) throw new WikiError(`title must be 1-${LIMITS.title} characters`);
  if (summary.length > LIMITS.summary) throw new WikiError(`summary must be at most ${LIMITS.summary} characters (one line)`);
  if (!body) throw new WikiError("body must not be empty");
  if (body.length > LIMITS.body) throw new WikiError(`body must be at most ${LIMITS.body} characters; split the page`);

  const cited = extractCitations(body);
  const dropped = new Set((input.drop_sources ?? []).filter((id) => !cited.includes(id)));
  return {
    title,
    summary,
    body,
    tags,
    links: extractLinks(body).filter((s) => s !== slug),
    sources: uniqueSortedIds([...(current?.sources ?? []), ...(input.sources ?? []), ...cited]).filter((id) => !dropped.has(id)),
  };
}

function sameContent(a: PageFields, b: PageFields): boolean {
  return a.title === b.title && a.summary === b.summary && a.body === b.body && a.tags.join("\n") === b.tags.join("\n");
}

function embeddingText(page: PageFields): string {
  return `${page.title}\n${page.summary}\n${page.tags.join(", ")}\n\n${page.body}`.slice(0, EMBED_CHARS);
}

export async function embedMissing(pool: Pool, embed: Embedder): Promise<number> {
  const missing = await withClient(pool, (c) =>
    c.queryObject<PageFields & { slug: string; revision: number }>(
      "SELECT slug, title, summary, body, tags, links, sources, revision FROM wiki_pages WHERE embedding IS NULL ORDER BY slug",
    )
  );
  let fixed = 0;
  for (const page of missing.rows) {
    try {
      const vector = `[${(await embed(embeddingText(page))).join(",")}]`;
      const result = await withClient(pool, (c) =>
        c.queryObject(
          "UPDATE wiki_pages SET embedding = $3::vector WHERE slug = $1 AND revision = $2 AND embedding IS NULL",
          [page.slug, page.revision, vector],
        )
      );
      fixed += result.rowCount ?? 0;
    } catch (err) {
      console.error(`wiki: embedding failed for ${page.slug}: ${(err as Error).message}`);
    }
  }
  return fixed;
}

const SEARCH_VECTOR = `setweight(to_tsvector('simple', $2::text), 'A')
  || setweight(to_tsvector('simple', $3::text || ' ' || array_to_string($5::text[], ' ')), 'B')
  || setweight(to_tsvector('simple', $4::text), 'C')`;

export async function writePage(pool: Pool, embed: Embedder | null, input: WriteInput): Promise<WriteResult> {
  const slug = (input.slug ?? "").trim();
  const invalid = slugError(slug);
  if (invalid) throw new WikiError(invalid);
  const author = oneLine(input.author || "mcp", 60);
  const note = oneLine(input.note ?? "", 300);

  for (let attempt = 0; attempt < 3; attempt++) {
    const current = await withClient(pool, (c) => getPage(c, slug));
    const currentRevision = current?.revision ?? 0;
    if (input.expected_revision !== undefined && input.expected_revision !== currentRevision) {
      throw new WikiError(
        `revision conflict on "${slug}": expected ${input.expected_revision}, current is ${currentRevision} (0 means the page does not exist). Re-read the page and retry.`,
      );
    }
    const next = mergePage(slug, current, input);
    const contentChanged = !current || !sameContent(current, next);
    if (!contentChanged && next.sources.join(",") === current!.sources.join(",")) {
      return { status: "unchanged", page: current!, missing_links: [] };
    }

    const known = new Set(current?.sources ?? []);
    const newIds = next.sources.filter((id) => !known.has(id));
    if (newIds.length) {
      const found = await withClient(pool, (c) =>
        c.queryObject<{ id: bigint }>("SELECT id FROM thoughts WHERE id = ANY($1::bigint[])", [newIds])
      );
      const existing = new Set(found.rows.map((r) => Number(r.id)));
      const unknown = newIds.filter((id) => !existing.has(id));
      if (unknown.length) {
        throw new WikiError(
          `unknown thought id(s): ${unknown.map((id) => `#${id}`).join(", ")}. Cite only ids returned by search or fetch.`,
        );
      }
    }

    let embedding: string | null = null;
    if (embed && contentChanged) {
      try {
        embedding = `[${(await embed(embeddingText(next))).join(",")}]`;
      } catch (err) {
        console.error(`wiki: embedding failed for ${slug}: ${(err as Error).message}`);
      }
    }

    const revision = await withClient(pool, async (client) => {
      await client.queryArray("BEGIN");
      try {
        const locked = await client.queryObject<{ revision: number }>(
          "SELECT revision FROM wiki_pages WHERE slug = $1 FOR UPDATE",
          [slug],
        );
        if ((locked.rows[0]?.revision ?? 0) !== currentRevision) {
          await client.queryArray("ROLLBACK");
          return null;
        }
        const nextRevision = currentRevision + 1;
        const fields = [slug, next.title, next.summary, next.body, next.tags, next.links, next.sources, embedding, author];
        if (current) {
          await client.queryObject(
            `UPDATE wiki_pages SET title = $2, summary = $3, body = $4, tags = $5::text[], links = $6::text[],
               sources = $7::bigint[], embedding = COALESCE($8::vector, embedding), search_tsv = ${SEARCH_VECTOR},
               revision = $10, updated_at = now(), updated_by = $9
             WHERE slug = $1`,
            [...fields, nextRevision],
          );
        } else {
          const inserted = await client.queryObject(
            `INSERT INTO wiki_pages (slug, title, summary, body, tags, links, sources, embedding, search_tsv, updated_by)
             VALUES ($1, $2, $3, $4, $5::text[], $6::text[], $7::bigint[], $8::vector, ${SEARCH_VECTOR}, $9)
             ON CONFLICT (slug) DO NOTHING RETURNING slug`,
            fields,
          );
          if (!inserted.rows.length) {
            await client.queryArray("ROLLBACK");
            return null;
          }
        }
        await client.queryObject(
          `INSERT INTO wiki_revisions (slug, revision, title, summary, body, tags, links, sources, author, note, compile_run_id)
           VALUES ($1, $2, $3, $4, $5, $6::text[], $7::text[], $8::bigint[], $9, $10, $11::bigint)`,
          [
            slug,
            nextRevision,
            next.title,
            next.summary,
            next.body,
            next.tags,
            next.links,
            next.sources,
            author,
            note,
            input.compile_run_id ?? null,
          ],
        );
        await client.queryArray("COMMIT");
        return nextRevision;
      } catch (err) {
        await client.queryArray("ROLLBACK").catch(() => {});
        throw err;
      }
    });
    if (revision === null) {
      if (input.expected_revision !== undefined) {
        throw new WikiError(`revision conflict on "${slug}": the page changed during the write. Re-read the page and retry.`);
      }
      continue;
    }

    return await withClient(pool, async (client) => ({
      status: current ? "updated" : "created",
      page: (await getPage(client, slug))!,
      missing_links: await missingSlugs(client, next.links),
    }));
  }
  throw new WikiError(`"${slug}" kept changing during the write; retry`);
}

export type SearchHit = {
  slug: string;
  title: string;
  summary: string;
  updated_at: Date;
  matched: string[];
  snippet: string;
};

function likePattern(query: string): string {
  const words = query.toLowerCase().split(/[\s\-_]+/).filter(Boolean).map((w) => w.replace(/[\\%_]/g, "\\$&"));
  return `%${words.join("%")}%`;
}

export async function searchPages(
  pool: Pool,
  embed: Embedder | null,
  query: string,
  limit: number,
): Promise<{ hits: SearchHit[]; semantic: boolean }> {
  const q = query.trim();
  if (!q) throw new WikiError("query is required");
  const depth = Math.max(limit * 3, 20);

  let vector: string | null = null;
  if (embed) {
    try {
      vector = `[${(await embed(q)).join(",")}]`;
    } catch (err) {
      console.error(`wiki: search embedding failed: ${(err as Error).message}`);
    }
  }

  return await withClient(pool, async (client) => {
    const text = await client.queryObject<{ slug: string; snippet: string }>(
      `SELECT slug,
              ts_headline('simple', body, q, 'MaxFragments=2, MinWords=6, MaxWords=24, FragmentDelimiter=" ... ", StartSel=**, StopSel=**') AS snippet
       FROM wiki_pages, websearch_to_tsquery('simple', $1) AS q
       WHERE search_tsv @@ q
       ORDER BY ts_rank_cd(search_tsv, q) DESC, slug
       LIMIT $2`,
      [q, depth],
    );
    const names = await client.queryObject<{ slug: string }>(
      `SELECT slug FROM wiki_pages WHERE slug ILIKE $1 OR title ILIKE $1 ORDER BY length(slug), slug LIMIT $2`,
      [likePattern(q), depth],
    );
    let meaning: string[] = [];
    if (vector) {
      const semantic = await client.queryObject<{ slug: string; similarity: number }>(
        `SELECT slug, 1 - (embedding <=> $1::vector) AS similarity
         FROM wiki_pages WHERE embedding IS NOT NULL
         ORDER BY embedding <=> $1::vector LIMIT $2`,
        [vector, depth],
      );
      meaning = semantic.rows.filter((r) => r.similarity >= SEMANTIC_FLOOR).map((r) => r.slug);
    }

    const lists = { text: text.rows.map((r) => r.slug), name: names.rows.map((r) => r.slug), meaning };
    const ranked = fuseRankings([lists.text, lists.name, lists.meaning]).slice(0, limit);
    if (!ranked.length) return { hits: [], semantic: vector !== null };

    const meta = await client.queryObject<{ slug: string; title: string; summary: string; updated_at: Date; head: string }>(
      "SELECT slug, title, summary, updated_at, left(body, 400) AS head FROM wiki_pages WHERE slug = ANY($1::text[])",
      [ranked.map((r) => r.slug)],
    );
    const bySlug = new Map(meta.rows.map((r) => [r.slug, r]));
    const snippets = new Map(text.rows.map((r) => [r.slug, r.snippet]));
    const hits = ranked.flatMap(({ slug }) => {
      const row = bySlug.get(slug);
      if (!row) return [];
      return [{
        slug,
        title: row.title,
        summary: row.summary,
        updated_at: row.updated_at,
        matched: (Object.keys(lists) as (keyof typeof lists)[]).filter((k) => lists[k].includes(slug)),
        snippet: oneLine(snippets.get(slug) ?? row.head, 320),
      }];
    });
    return { hits, semantic: vector !== null };
  });
}

type RevisionRow = {
  slug: string;
  revision: number;
  title: string;
  summary: string;
  body: string;
  tags: string[];
  author: string;
  note: string;
  compile_run_id: bigint | null;
  created_at: Date;
};

export async function renderLog(client: PoolClient, limit = 50): Promise<string> {
  const runs = await client.queryObject<{
    id: bigint;
    status: string;
    model: string;
    started_at: Date;
    from_thought_id: bigint;
    to_thought_id: bigint | null;
    thoughts_seen: number;
    pages_created: number;
    pages_updated: number;
    summary: string;
    error: string | null;
  }>(
    `SELECT id, status, model, started_at, from_thought_id, to_thought_id, thoughts_seen,
            pages_created, pages_updated, summary, error
     FROM wiki_compile_runs ORDER BY id DESC LIMIT $1`,
    [limit],
  );
  const edits = await client.queryObject<RevisionRow>(
    `SELECT slug, revision, author, note, created_at FROM wiki_revisions
     WHERE compile_run_id IS NULL ORDER BY id DESC LIMIT $1`,
    [limit],
  );
  const runIds = runs.rows.map((r) => Number(r.id));
  type Touched = { run_id: bigint; slug: string; first_revision: number };
  const touched: Touched[] = runIds.length
    ? (await client.queryObject<Touched>(
      `SELECT compile_run_id AS run_id, slug, min(revision)::int AS first_revision
       FROM wiki_revisions WHERE compile_run_id = ANY($1::bigint[])
       GROUP BY compile_run_id, slug ORDER BY slug`,
      [runIds],
    )).rows
    : [];

  const events: { at: Date; text: string }[] = [];
  for (const run of runs.rows) {
    const pages = touched.filter((t) => Number(t.run_id) === Number(run.id));
    const created = pages.filter((p) => p.first_revision === 1).map((p) => `[[${p.slug}]]`);
    const updated = pages.filter((p) => p.first_revision > 1).map((p) => `[[${p.slug}]]`);
    const range = run.to_thought_id !== null
      ? ` thoughts #${Number(run.from_thought_id) + 1}-#${Number(run.to_thought_id)} (${run.thoughts_seen} read)`
      : ` from thought #${Number(run.from_thought_id) + 1}`;
    const parts = [`compile #${Number(run.id)} ${run.status}${run.model ? ` (${run.model})` : ""}:${range}.`];
    if (created.length) parts.push(`Created ${created.join(", ")}.`);
    if (updated.length) parts.push(`Updated ${updated.join(", ")}.`);
    if (run.summary) parts.push(run.summary);
    if (run.error) parts.push(`Error: ${oneLine(run.error, 300)}`);
    events.push({ at: run.started_at, text: parts.join(" ") });
  }
  for (const edit of edits.rows) {
    events.push({
      at: edit.created_at,
      text: `[[${edit.slug}]] rev ${edit.revision} by ${edit.author}${edit.note ? `: ${edit.note}` : ""}`,
    });
  }
  events.sort((a, b) => +b.at - +a.at);

  const lines = ["# Open Brain wiki: log", "", "Newest first. Compile runs list the pages they created or updated; other entries are direct wiki_write edits."];
  let currentDay = "";
  for (const event of events.slice(0, limit)) {
    const eventDay = day(event.at);
    if (eventDay !== currentDay) {
      lines.push("", `## ${eventDay}`);
      currentDay = eventDay;
    }
    lines.push(`- ${new Date(event.at).toISOString().slice(11, 16)} UTC ${event.text}`);
  }
  if (!events.length) lines.push("", "No changes yet.");
  return lines.join("\n");
}

export async function renderPending(client: PoolClient, limit = 50): Promise<string> {
  const { high_water_mark } = await compileStatus(client);
  const uncovered = `FROM thoughts t
     WHERE t.id > $1 AND NOT EXISTS (SELECT 1 FROM wiki_pages p WHERE p.sources @> ARRAY[t.id])`;
  const total = await client.queryObject<{ count: number }>(`SELECT COUNT(*)::int AS count ${uncovered}`, [high_water_mark]);
  const rows = await client.queryObject<{ id: bigint; content: string; created_at: Date }>(
    `SELECT t.id, t.content, t.created_at ${uncovered} ORDER BY t.id LIMIT $2`,
    [high_water_mark, limit],
  );
  const count = total.rows[0]?.count ?? 0;
  const lines = [
    "# Open Brain wiki: pending thoughts",
    "",
    `${count} thought(s) after the last compile (#${high_water_mark}) are not cited by any page yet.${count > rows.rows.length ? ` Showing the oldest ${rows.rows.length}.` : ""}`,
  ];
  for (const row of rows.rows) {
    lines.push("", `## [#${Number(row.id)}] ${stamp(row.created_at)}`, "", row.content.trim());
  }
  return lines.join("\n");
}

export async function buildIndex(client: PoolClient): Promise<string> {
  return renderIndex(await listPages(client), await compileStatus(client));
}

function formatIds(ids: number[], max = 120): string {
  const shown = ids.slice(0, max).map((id) => `#${id}`).join(", ");
  return ids.length > max ? `${shown} and ${ids.length - max} more` : shown;
}

async function readPageText(
  client: PoolClient,
  slug: string,
  opts: { revision?: number; history?: boolean; limit?: number },
): Promise<string> {
  const { revision, history } = opts;
  const limit = Math.min(Math.max(1, Math.floor(opts.limit ?? 50)), 500);
  if (slug === "index") return await buildIndex(client);
  if (slug === "log") return await renderLog(client, limit);
  if (slug === "pending") return await renderPending(client, limit);

  const page = await getPage(client, slug);
  if (!page) {
    const similar = await client.queryObject<{ slug: string }>(
      "SELECT slug FROM wiki_pages WHERE slug ILIKE $1 OR title ILIKE $1 ORDER BY slug LIMIT 8",
      [likePattern(slug.replace(/-/g, " "))],
    );
    const hint = similar.rows.length ? ` Similar: ${similar.rows.map((r) => `[[${r.slug}]]`).join(", ")}.` : "";
    return `No wiki page "${slug}".${hint} Use wiki_search or wiki_index to find pages.`;
  }

  if (history) {
    const revisions = await client.queryObject<RevisionRow>(
      "SELECT slug, revision, author, note, created_at FROM wiki_revisions WHERE slug = $1 ORDER BY revision DESC LIMIT $2",
      [slug, limit],
    );
    return [
      `# ${page.title}: revision history`,
      "",
      ...revisions.rows.map((r) => `- rev ${r.revision} ${stamp(r.created_at)} by ${r.author}${r.note ? `: ${r.note}` : ""}`),
      "",
      "Read an older version with wiki_read(slug, revision).",
    ].join("\n");
  }

  if (revision !== undefined && revision !== page.revision) {
    const old = await client.queryObject<RevisionRow & { sources: unknown }>(
      `SELECT slug, revision, title, summary, body, tags, sources, author, note, created_at
       FROM wiki_revisions WHERE slug = $1 AND revision = $2`,
      [slug, revision],
    );
    const row = old.rows[0];
    if (!row) return `Page "${slug}" has no revision ${revision} (current is ${page.revision}).`;
    return [
      `# ${row.title}`,
      `slug: ${slug} | revision ${row.revision} of ${page.revision} (old) | written ${stamp(row.created_at)} by ${row.author}${row.note ? ` (${row.note})` : ""}`,
      `summary: ${row.summary}`,
      `sources: ${formatIds(toIds(row.sources))}`,
      "",
      "---",
      "",
      row.body,
    ].join("\n");
  }

  const inbound = await backlinks(client, slug);
  const missing = new Set(await missingSlugs(client, page.links));
  const links = page.links.map((l) => (missing.has(l) ? `[[${l}]] (missing)` : `[[${l}]]`));
  return [
    `# ${page.title}`,
    `slug: ${page.slug} | revision ${page.revision} | updated ${stamp(page.updated_at)} by ${page.updated_by} | tags: ${page.tags.join(", ") || "none"}`,
    `summary: ${page.summary || "(none)"}`,
    `sources (${page.sources.length} thoughts): ${formatIds(page.sources) || "none"}`,
    `links: ${links.join(", ") || "none"} | linked from: ${inbound.map((l) => `[[${l}]]`).join(", ") || "none"}`,
    "",
    "---",
    "",
    page.body,
  ].join("\n");
}

// --- MCP tools ---

export type WikiContext = { pool: Pool; embed: Embedder };

export function registerWikiTools(server: McpServer, ctx: WikiContext): void {
  const respond = async (fn: () => Promise<string>) => {
    try {
      await ensureWikiSchema(ctx.pool);
      return { content: [{ type: "text" as const, text: await fn() }] };
    } catch (err: unknown) {
      return { content: [{ type: "text" as const, text: `Error: ${(err as Error).message}` }], isError: true };
    }
  };

  server.registerTool(
    "wiki_index",
    {
      title: "Wiki Index",
      description:
        "Start here for background on any project, client, person, server, tool or standing rule in Open Brain. Returns the wiki's table of contents: every compiled page with a one-line summary, grouped by category, plus when the wiki was last compiled and how many newer thoughts are not compiled yet. Wiki pages are deduplicated current state with [#thought-id] citations; raw notes stay in search_thoughts.",
      annotations: { readOnlyHint: true },
      inputSchema: {},
    },
    () => respond(() => withClient(ctx.pool, (client) => buildIndex(client))),
  );

  server.registerTool(
    "wiki_list",
    {
      title: "List Wiki Pages",
      description:
        "List wiki pages with their one-line summaries, optionally filtered by tag or category (project, client, person, organization, infrastructure, tool, process, rules, topic) or by pages updated in the last N days. Use it to find a slug for wiki_read; wiki_index gives the grouped overview.",
      annotations: { readOnlyHint: true },
      inputSchema: {
        tag: z.string().optional().describe("Only pages with this tag or category"),
        days: z.number().optional().describe("Only pages updated in the last N days"),
        sort: z.enum(["slug", "updated"]).optional().default("slug"),
        limit: z.number().optional().default(200),
        offset: z.number().optional().default(0),
      },
    },
    ({ tag, days, sort, limit, offset }) =>
      respond(() =>
        withClient(ctx.pool, async (client) => {
          const pages = await listPages(client, {
            tag,
            days,
            sort,
            limit: Math.min(Math.max(1, Math.floor(limit)), 1000),
            offset: Math.max(0, Math.floor(offset)),
          });
          if (!pages.length) return "No wiki pages match. Use wiki_index to see all pages.";
          return [
            `${pages.length} wiki page(s):`,
            ...pages.map((p) =>
              `- ${p.slug}: ${p.title}${p.summary ? `. ${p.summary}` : ""} (${categoryOf(p.tags)}, rev ${p.revision}, updated ${day(p.updated_at)}, ${p.source_count} sources)`
            ),
          ].join("\n");
        })
      ),
  );

  server.registerTool(
    "wiki_read",
    {
      title: "Read Wiki Page",
      description:
        "Read one wiki page by slug: compiled markdown (current state, key facts, decisions, history) with [#id] thought citations, [[slug]] links and backlinks. Built-in pages: 'index' (table of contents), 'log' (recent changes) and 'pending' (thoughts not compiled into any page yet, with ids and full text). Optional: an older revision number, or history=true for the revision list. For anything newer than the page's updated date, also check search_thoughts.",
      annotations: { readOnlyHint: true },
      inputSchema: {
        slug: z.string().describe("Page slug, for example 'hotel-sofia', or 'index' / 'log'"),
        revision: z.number().optional().describe("Read this older revision instead of the current one"),
        history: z.boolean().optional().default(false).describe("List the page's revisions instead of its text"),
        limit: z.number().optional().default(50).describe("Entries to show for 'log' and history"),
      },
    },
    ({ slug, revision, history, limit }) =>
      respond(() =>
        withClient(ctx.pool, (client) => readPageText(client, slug.trim().toLowerCase(), { revision, history, limit }))
      ),
  );

  server.registerTool(
    "wiki_search",
    {
      title: "Search Wiki",
      description:
        "Search wiki pages by keywords and by meaning (Postgres full-text plus embedding similarity, merged). Prefer it over search_thoughts when you need the consolidated, current picture of a subject; use search_thoughts for raw notes, exact wording, or anything captured after the last compile.",
      annotations: { readOnlyHint: true },
      inputSchema: {
        query: z.string().describe("Keywords or a question"),
        limit: z.number().optional().default(8),
      },
    },
    ({ query, limit }) =>
      respond(async () => {
        const { hits, semantic } = await searchPages(ctx.pool, ctx.embed, query, Math.min(Math.max(1, Math.floor(limit)), 25));
        if (!hits.length) return `No wiki pages match "${query}". Try search_thoughts for raw notes.`;
        const mode = semantic ? "full-text + semantic" : "full-text only (embedding unavailable)";
        return [
          `${hits.length} wiki page(s) for "${query}" (${mode}):`,
          ...hits.map((h, i) =>
            [
              `${i + 1}. [[${h.slug}]] ${h.title} (updated ${day(h.updated_at)}; matched: ${h.matched.join(", ")})`,
              h.summary ? `   ${h.summary}` : "",
              `   ${h.snippet}`,
            ].filter(Boolean).join("\n")
          ),
        ].join("\n");
      }),
  );

  server.registerTool(
    "wiki_write",
    {
      title: "Write Wiki Page",
      description:
        "Create or update a wiki page; every write keeps a revision. Use it to correct, merge or restructure compiled knowledge, not to log new events: record new facts with capture_thought and the compiler folds them in. Read the page first and pass expected_revision (0 for a new page) so you do not overwrite a concurrent edit. Cite thoughts as [#id], link pages as [[slug]]. Omitted fields keep their current values; append=true adds body to the end of the page.",
      annotations: { readOnlyHint: false, destructiveHint: false, idempotentHint: false, openWorldHint: false },
      inputSchema: {
        slug: z.string().describe("Lowercase kebab-case slug, for example 'hotel-sofia'"),
        title: z.string().optional().describe("Page title (required for a new page)"),
        summary: z.string().optional().describe("One line, under 140 characters: what the subject is and its current state"),
        body: z.string().optional().describe("Markdown body without the title heading (required for a new page)"),
        tags: z.array(z.string()).optional().describe("First tag is the category: project, client, person, organization, infrastructure, tool, process, rules or topic"),
        sources: z.array(z.number()).optional().describe("Thought ids this page draws on (ids cited as [#id] are added automatically)"),
        drop_sources: z.array(z.number()).optional().describe("Thought ids to remove from sources, for example after deleting a wrong citation (ids still cited in the body stay)"),
        append: z.boolean().optional().default(false).describe("Append body to the existing page instead of replacing it"),
        expected_revision: z.number().optional().describe("Current revision you read (0 for a new page); the write fails if the page changed"),
        author: z.string().optional().describe("Who is writing, for example 'claude-code' or 'codex'"),
        note: z.string().optional().describe("Short change note for the revision log"),
      },
    },
    (args) =>
      respond(async () => {
        const result = await writePage(ctx.pool, ctx.embed, { ...args, slug: args.slug.trim().toLowerCase() });
        const page = result.page;
        if (result.status === "unchanged") return `No change: "${page.slug}" is already at revision ${page.revision} with this content.`;
        const missing = result.missing_links.length
          ? ` Links to pages that do not exist yet: ${result.missing_links.map((l) => `[[${l}]]`).join(", ")}.`
          : "";
        return `${result.status === "created" ? "Created" : "Updated"} "${page.slug}" (revision ${page.revision}, ${page.sources.length} sources, ${page.links.length} links).${missing}`;
      }),
  );
}
