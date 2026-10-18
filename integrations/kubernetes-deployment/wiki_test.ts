// Wiki tests. Unit tests always run; the integration test needs a pgvector database
// (WIKI_TEST_DB_HOST) and is driven by test.sh.

import {
  assert,
  assertEquals,
  assertMatch,
  assertStringIncludes,
} from "jsr:@std/assert@1";
import {
  categoryOf,
  extractCitations,
  extractLinks,
  fuseRankings,
  normalizeTags,
  type PageListing,
  renderIndex,
  slugError,
  slugify,
} from "./wiki.ts";

Deno.test("slugError accepts kebab-case and rejects the rest", () => {
  assertEquals(slugError("hotel-sofia"), null);
  assertEquals(slugError("a1-b2"), null);
  assertMatch(slugError("Hotel Sofia")!, /invalid slug/);
  assertMatch(slugError("trailing-")!, /invalid slug/);
  assertMatch(slugError("x".repeat(81))!, /invalid slug/);
  assertMatch(slugError("index")!, /built-in/);
  assertMatch(slugError("log")!, /built-in/);
  assertMatch(slugError("pending")!, /built-in/);
  assertMatch(slugError("")!, /required/);
});

Deno.test("slugify makes ASCII kebab-case", () => {
  assertEquals(slugify("Hotel Sofia (Bansko)"), "hotel-sofia-bansko");
  assertEquals(slugify("  Café  Déjà vu "), "cafe-deja-vu");
  assertEquals(slugify("Пулт"), "");
});

Deno.test("extractLinks keeps valid unique slugs, with or without labels", () => {
  assertEquals(
    extractLinks("See [[hotel-sofia]], [[hotel-sofia|the hotel]], [[Bad Slug]] and [[sites-1]]."),
    ["hotel-sofia", "sites-1"],
  );
});

Deno.test("extractCitations returns sorted unique ids", () => {
  assertEquals(extractCitations("a [#12][#3] b [#12] c [#x] [# 4]"), [3, 12]);
});

Deno.test("normalizeTags lowercases, trims and dedupes", () => {
  assertEquals(normalizeTags([" Project ", "project", "Hotel  Management", ""]), ["project", "hotel management"]);
});

Deno.test("categoryOf finds the category tag", () => {
  assertEquals(categoryOf(["hotel", "client"]), "client");
  assertEquals(categoryOf(["hotel"]), "other");
});

Deno.test("fuseRankings rewards agreement across lists", () => {
  const fused = fuseRankings([["a", "b", "c"], ["b", "a"], ["b"]]);
  assertEquals(fused.map((f) => f.slug), ["b", "a", "c"]);
});

Deno.test("renderIndex groups pages by category and reports pending thoughts", () => {
  const page = (slug: string, tags: string[]): PageListing => ({
    slug,
    title: slug.toUpperCase(),
    summary: `about ${slug}`,
    tags,
    revision: 1,
    updated_at: new Date("2026-10-18T10:00:00Z"),
    updated_by: "compile",
    source_count: 2,
  });
  const pages = [
    page("hotel-sofia", ["client"]),
    page("hotel-sofia-payments", ["topic"]),
    page("hotel-sofia-payments-stripe", ["topic"]),
    page("hub", ["project"]),
    page("misc", []),
  ];
  const text = renderIndex(pages, {
    last_ok: { id: 4, status: "ok", finished_at: new Date("2026-10-18T10:00:00Z"), to_thought_id: 976, model: "m" },
    last_attempt: { id: 4, status: "ok", started_at: new Date("2026-10-18T09:00:00Z"), error: null },
    high_water_mark: 976,
    pending: 7,
  });
  assertStringIncludes(text, "5 page(s). Last compile: 2026-10-18 10:00 UTC (run #4, ok, thoughts up to #976).");
  assertStringIncludes(text, "7 thought(s) captured since then are not in the wiki yet");
  assert(text.indexOf("## Projects (1)") < text.indexOf("## Clients (1)"));
  assertStringIncludes(
    text,
    [
      "- [[hotel-sofia]] HOTEL-SOFIA: about hotel-sofia",
      "  - [[hotel-sofia-payments]] HOTEL-SOFIA-PAYMENTS: about hotel-sofia-payments",
      "    - [[hotel-sofia-payments-stripe]] HOTEL-SOFIA-PAYMENTS-STRIPE: about hotel-sofia-payments-stripe",
    ].join("\n"),
  );
  assert(!text.includes("## Topics"), "sub-pages should be nested, not listed under their own category");
  assertStringIncludes(text, "## Other (1)");
});

// --- Integration: real server process, real Postgres, fake OpenAI API ---

const DB_HOST = Deno.env.get("WIKI_TEST_DB_HOST");
const ACCESS_KEY = "test-access-key";

const fake = { failEmbeddings: false, breakWriter: false, dropOrphans: false, writePrompts: [] as string[] };

function fakeEmbedding(text: string): number[] {
  const v = new Array(1536).fill(0);
  v[0] = 0.01;
  for (const word of text.toLowerCase().match(/[\p{L}\p{N}]+/gu) ?? []) {
    let h = 2166136261;
    for (const ch of word) h = Math.imul(h ^ ch.codePointAt(0)!, 16777619);
    v[1 + (Math.abs(h) % 1535)] += 1;
  }
  const norm = Math.sqrt(v.reduce((s, x) => s + x * x, 0));
  return v.map((x) => x / norm);
}

function fakeChat(system: string, user: string): Record<string, unknown> {
  if (system.includes("Extract metadata")) {
    return { topics: ["test"], type: "observation", people: [], action_items: [], dates_mentioned: [] };
  }
  if (system.includes("Do not assign thoughts yet")) {
    const pages = [];
    if (user.includes("Hotel")) pages.push({ slug: "hotel-sofia", title: "Hotel Sofia", category: "client", summary: "Hotel in Bansko" });
    if (user.includes("Broken")) pages.push({ slug: "broken-subject", title: "Broken subject", category: "topic", summary: "x" });
    return { new_pages: pages };
  }
  if (system.includes("Assign every thought")) {
    const assignments = [...user.matchAll(/\[#(\d+)\] [^\n]*\n([^\n]*)/g)]
      .filter(([, , text]) => !(fake.dropOrphans && text.includes("Orphan")))
      .map(([, id, text]) => ({
        id: Number(id),
        pages: text.includes("Hotel") ? ["hotel-sofia"] : text.includes("Broken") ? ["broken-subject"] : [],
      }));
    return { new_pages: [], assignments };
  }
  if (system.includes("compile one page")) {
    fake.writePrompts.push(user);
    const slug = user.match(/^Page slug: (\S+)/)?.[1];
    if (slug === "broken-subject" && fake.breakWriter) return { title: "Broken", summary: "", tags: [], body: "" };
    const ids = [...user.matchAll(/=== \[#(\d+)\] captured [^\n]*UTC ===/g)].map((m) => m[1]);
    const current = user.match(/Current body:\n<<<\n([\s\S]*?)\n>>>/)?.[1];
    const facts = ids.map((id) => `- Fact from thought ${id} [#${id}]`).join("\n");
    const body = current
      ? `${current}\n${facts}`
      : `# Should be stripped\n\nIntro sentence. [#999999]\n\n## Current state\n${facts}\n\nSee [[test-page]] and [[no-such-page]].`;
    return { title: slug === "hotel-sofia" ? "Hotel Sofia" : "Broken subject", summary: `Summary of ${slug}`, tags: ["client", "Hotel"], body };
  }
  throw new Error(`unexpected chat prompt: ${system.slice(0, 60)}`);
}

function startFakeOpenAI(): Deno.HttpServer<Deno.NetAddr> {
  return Deno.serve({ hostname: "127.0.0.1", port: 0, onListen: () => {} }, async (req) => {
    const body = await req.json();
    const path = new URL(req.url).pathname;
    if (path.endsWith("/embeddings")) {
      if (fake.failEmbeddings) return new Response("down", { status: 503 });
      return Response.json({ data: [{ embedding: fakeEmbedding(String(body.input)) }] });
    }
    if (path.endsWith("/chat/completions")) {
      const content = JSON.stringify(fakeChat(body.messages[0].content, body.messages[1].content));
      return Response.json({
        choices: [{ message: { content }, finish_reason: "stop" }],
        usage: { prompt_tokens: 10, completion_tokens: 5, completion_tokens_details: { reasoning_tokens: 1 } },
      });
    }
    return new Response("not found", { status: 404 });
  });
}

function freePort(): number {
  const listener = Deno.listen({ hostname: "127.0.0.1", port: 0 });
  const { port } = listener.addr as Deno.NetAddr;
  listener.close();
  return port;
}

Deno.test({
  name: "integration: MCP endpoint, old tools, wiki tools and compile",
  ignore: !DB_HOST,
  sanitizeResources: false,
  sanitizeOps: false,
  async fn(t) {
    const openai = startFakeOpenAI();
    const port = freePort();
    const env = {
      DB_HOST: DB_HOST!,
      DB_PORT: "5432",
      DB_NAME: "openbrain",
      DB_USER: "openbrain",
      DB_PASSWORD: Deno.env.get("WIKI_TEST_DB_PASSWORD") ?? "test",
      EMBEDDING_API_BASE: `http://127.0.0.1:${openai.addr.port}/v1`,
      EMBEDDING_API_KEY: "fake-openai-key",
      EMBEDDING_MODEL: "text-embedding-3-small",
      CHAT_MODEL: "gpt-4o-mini",
      MCP_ACCESS_KEY: ACCESS_KEY,
      PORT: String(port),
    };
    const dir = new URL(".", import.meta.url).pathname;
    const server = new Deno.Command(Deno.execPath(), {
      args: ["run", "--no-lock", "--allow-net", "--allow-env", "--allow-read", "index.ts"],
      cwd: dir,
      env,
      stdout: "null",
      stderr: "inherit",
    }).spawn();
    const url = `http://127.0.0.1:${port}/`;

    const compile = async (...args: string[]) => {
      const out = await new Deno.Command(Deno.execPath(), {
        args: ["run", "--no-lock", "--allow-net", "--allow-env", "--allow-read", "--allow-write=/tmp", "wiki_compile.ts", ...args],
        cwd: dir,
        env: { ...env, WIKI_COMPILE_MODEL: "gpt-5.4-mini", WIKI_COMPILE_SETTLE_SECONDS: "0", WIKI_COMPILE_RELATED_FLOOR: "-1" },
      }).output();
      const text = new TextDecoder().decode(out.stdout) + new TextDecoder().decode(out.stderr);
      assertEquals(out.code, 0, text);
      return text;
    };

    let seq = 0;
    const rpc = async (method: string, params: unknown = {}, key = ACCESS_KEY) => {
      const r = await fetch(url, {
        method: "POST",
        headers: { "content-type": "application/json", accept: "application/json, text/event-stream", "x-brain-key": key },
        body: JSON.stringify({ jsonrpc: "2.0", id: ++seq, method, params }),
      });
      const text = await r.text();
      const data = text.split("\n").find((l) => l.startsWith("data: "));
      return { status: r.status, body: JSON.parse(data ? data.slice(6) : text) };
    };
    const call = async (name: string, args: Record<string, unknown> = {}) => {
      const { body } = await rpc("tools/call", { name, arguments: args });
      assert(body.result, JSON.stringify(body));
      return { text: body.result.content[0].text as string, isError: Boolean(body.result.isError) };
    };
    const ok = async (name: string, args: Record<string, unknown> = {}) => {
      const r = await call(name, args);
      assert(!r.isError, `${name} failed: ${r.text}`);
      return r.text;
    };

    try {
      for (let i = 0;; i++) {
        try {
          if ((await fetch(url, { method: "POST" })).status === 401) break;
        } catch { /* server still starting */ }
        if (i > 240) throw new Error("server did not start");
        await new Promise((r) => setTimeout(r, 500));
      }

      await t.step("auth and transport guards are unchanged", async () => {
        assertEquals((await rpc("tools/list", {}, "wrong")).status, 401);
        assertEquals((await fetch(url)).status, 405);
      });

      await t.step("tools/list has the old and the wiki tools; instructions mention the wiki", async () => {
        const { body } = await rpc("tools/list");
        const names = body.result.tools.map((tool: { name: string }) => tool.name).sort();
        assertEquals(names, [
          "capture_thought",
          "fetch",
          "list_thoughts",
          "search",
          "search_thoughts",
          "thought_stats",
          "wiki_index",
          "wiki_list",
          "wiki_read",
          "wiki_search",
          "wiki_write",
        ]);
        const init = await rpc("initialize", {
          protocolVersion: "2025-06-18",
          capabilities: {},
          clientInfo: { name: "wiki-test", version: "1" },
        });
        assertStringIncludes(init.body.result.instructions, "wiki_index");
      });

      const hotel1 = "As of 2026-10-01: Hotel Sofia in Bansko closes for the season from 2026-10-05.";
      await t.step("old tools still work (and search/fetch return ids)", async () => {
        assertStringIncludes(await ok("capture_thought", { content: hotel1 }), "Captured as observation");
        await ok("capture_thought", {
          content: "SUPERSEDES earlier notes on Hotel Sofia closure. As of 2026-10-02: Hotel Sofia closes from 2026-10-10.",
        });
        await ok("capture_thought", { content: "Random note about nothing in particular." });
        assertStringIncludes(await ok("list_thoughts"), "3 recent thought(s)");
        assertStringIncludes(await ok("search_thoughts", { query: hotel1 }), "Found");
        assertStringIncludes(await ok("thought_stats"), "Total thoughts: 3");
        const search = JSON.parse(await ok("search", { query: hotel1 }));
        assertEquals(search.results[0].id, "1");
        const doc = JSON.parse(await ok("fetch", { id: "1" }));
        assertEquals(doc.text, hotel1);
      });

      await t.step("wiki_index on an empty wiki", async () => {
        const text = await ok("wiki_index");
        assertStringIncludes(text, "0 page(s). Not compiled yet.");
        assertStringIncludes(text, "3 thought(s) captured since then");
      });

      await t.step("wiki_write validates input", async () => {
        assertMatch((await call("wiki_write", { slug: "index", title: "x", body: "y" })).text, /built-in/);
        assertMatch((await call("wiki_write", { slug: "Bad Slug", title: "x", body: "y" })).text, /invalid slug/);
        assertMatch((await call("wiki_write", { slug: "new-page", body: "y" })).text, /title and body are required/);
        assertMatch((await call("wiki_write", { slug: "x-page", title: "x", body: "cites [#999999]" })).text, /unknown thought id/);
        assertMatch((await call("wiki_write", { slug: "x-page", title: "x", body: "y", expected_revision: 3 })).text, /revision conflict/);
      });

      await t.step("wiki_write creates, updates, appends and keeps revisions", async () => {
        const created = await ok("wiki_write", {
          slug: "test-page",
          title: "Test page",
          summary: "A page written by the test",
          body: "Closure date is 2026-10-10 [#2]. See [[hotel-sofia]].",
          tags: ["topic", "Testing"],
          expected_revision: 0,
          author: "wiki-test",
          note: "first",
        });
        assertStringIncludes(created, 'Created "test-page" (revision 1, 1 sources, 1 links)');
        assertStringIncludes(created, "do not exist yet: [[hotel-sofia]]");

        assertMatch((await call("wiki_write", { slug: "test-page", body: "z", expected_revision: 0 })).text, /revision conflict/);
        assertStringIncludes(
          await ok("wiki_write", { slug: "test-page", body: "Closure date is 2026-10-10 [#2][#1]. See [[hotel-sofia]].", expected_revision: 1 }),
          "revision 2, 2 sources",
        );
        assertStringIncludes(
          await ok("wiki_write", { slug: "test-page", body: "Closure date is 2026-10-10 [#2][#1]. See [[hotel-sofia]]." }),
          "No change",
        );
        assertStringIncludes(await ok("wiki_write", { slug: "test-page", body: "## Notes\nUnique word zebracorn.", append: true }), "revision 3");

        const page = await ok("wiki_read", { slug: "test-page" });
        assertStringIncludes(page, "# Test page");
        assertStringIncludes(page, "revision 3");
        assertStringIncludes(page, "tags: topic, testing");
        assertStringIncludes(page, "sources (2 thoughts): #1, #2");
        assertStringIncludes(page, "[[hotel-sofia]] (missing)");
        assertStringIncludes(page, "Unique word zebracorn.");

        assertStringIncludes(await ok("wiki_read", { slug: "test-page", revision: 1 }), "Closure date is 2026-10-10 [#2]. See");
        const history = await ok("wiki_read", { slug: "test-page", history: true });
        assertMatch(history, /rev 3 .* by mcp/);
        assertMatch(history, /rev 1 .* by wiki-test: first/);
        assertStringIncludes(await ok("wiki_read", { slug: "test-pag" }), "Similar: [[test-page]]");
      });

      await t.step("wiki_search uses full text, names and meaning; degrades without embeddings", async () => {
        const hit = await ok("wiki_search", { query: "zebracorn" });
        assertStringIncludes(hit, "(full-text + semantic)");
        assertStringIncludes(hit, "[[test-page]] Test page");
        assertStringIncludes(hit, "**zebracorn**");
        assertMatch(await ok("wiki_search", { query: "test page" }), /matched: [^)]*name/);
        assertStringIncludes(await ok("wiki_search", { query: "nothingmatcheshere" }), "No wiki pages match");

        fake.failEmbeddings = true;
        try {
          assertStringIncludes(await ok("wiki_search", { query: "zebracorn" }), "full-text only");
          assertStringIncludes(await ok("wiki_write", { slug: "test-page", body: "More text.", append: true }), "revision 4");
        } finally {
          fake.failEmbeddings = false;
        }
      });

      await t.step("pending lists thoughts no page cites yet", async () => {
        const pending = await ok("wiki_read", { slug: "pending" });
        assertStringIncludes(pending, "1 thought(s) after the last compile (#0) are not cited by any page yet.");
        assertStringIncludes(pending, "## [#3] ");
        assertStringIncludes(pending, "Random note about nothing in particular.");
        assert(!pending.includes("[#1]") && !pending.includes("[#2]"), "cited thoughts should not be pending");
        assertMatch((await call("wiki_write", { slug: "pending", title: "x", body: "y" })).text, /built-in/);
      });

      await t.step("wiki_list filters and sorts", async () => {
        assertStringIncludes(await ok("wiki_list"), "- test-page: Test page. A page written by the test (topic, rev 4");
        assertStringIncludes(await ok("wiki_list", { tag: "testing", sort: "updated", days: 1 }), "1 wiki page(s)");
        assertStringIncludes(await ok("wiki_list", { tag: "client" }), "No wiki pages match");
      });

      await t.step("compile builds pages from thoughts and is idempotent", async () => {
        const out = await compile();
        assertStringIncludes(out, "3 new thought(s), #1-#3");
        assertStringIncludes(out, "created hotel-sofia");

        const page = await ok("wiki_read", { slug: "hotel-sofia" });
        assertStringIncludes(page, "# Hotel Sofia");
        assertStringIncludes(page, "by compile");
        assertStringIncludes(page, "tags: client, hotel");
        assertStringIncludes(page, "sources (2 thoughts): #1, #2");
        assertStringIncludes(page, "- Fact from thought 1 [#1]");
        assertStringIncludes(page, "linked from: [[test-page]]");
        assertStringIncludes(page, "See [[test-page]] and no such page.");
        assert(!page.includes("999999"), "invalid citation was not stripped");
        assert(!page.includes("Should be stripped"), "leading title heading was not stripped");

        const index = await ok("wiki_index");
        assertStringIncludes(index, "2 page(s). Last compile:");
        assertStringIncludes(index, "thoughts up to #3");
        assertStringIncludes(index, "No newer thoughts are waiting.");
        assertStringIncludes(index, "- [[hotel-sofia]] Hotel Sofia: Summary of hotel-sofia");

        const changes = await ok("wiki_read", { slug: "log" });
        assertMatch(changes, /compile #1 ok \(gpt-5\.4-mini\): thoughts #1-#3 \(3 read\)\. Created \[\[hotel-sofia\]\]\./);
        assertStringIncludes(changes, "Skipped as noise: #3.");
        assertMatch(changes, /\[\[test-page\]\] rev 1 by wiki-test: first/);

        assertStringIncludes(await ok("wiki_read", { slug: "pending" }), "0 thought(s) after the last compile (#3)");
        assertStringIncludes(await compile(), "No new thoughts since the last compile (high-water mark #3)");
      });

      await t.step("a failed page leaves its thoughts pending and the next compile recovers", async () => {
        await ok("capture_thought", { content: "Hotel Sofia added a sauna in October." });
        await ok("capture_thought", { content: "Broken subject note for the failure test." });
        fake.breakWriter = true;
        const partial = await compile("--concurrency", "1");
        assertStringIncludes(partial, "updated hotel-sofia");
        assertStringIncludes(partial, "FAILED broken-subject");
        assertStringIncludes(partial, "compile #2 partial");
        assertStringIncludes(await ok("wiki_index"), "thoughts up to #4");
        assertStringIncludes(await ok("wiki_read", { slug: "hotel-sofia" }), "Fact from thought 4 [#4]");

        fake.breakWriter = false;
        const recovered = await compile();
        assertStringIncludes(recovered, "1 new thought(s), #5-#5");
        assertStringIncludes(recovered, "created broken-subject");
        assertStringIncludes(await ok("wiki_index"), "thoughts up to #5");

        const status = await compile("--status");
        assertStringIncludes(status, "pages: 3");
        assertStringIncludes(status, "pending thoughts: 0");
      });

      await t.step("export writes markdown files", async () => {
        await compile("--export", "/tmp/wiki-export-test");
        const page = await Deno.readTextFile("/tmp/wiki-export-test/hotel-sofia.md");
        assertStringIncludes(page, 'title: "Hotel Sofia"');
        assertStringIncludes(page, "sources: [1, 2, 4]");
        assertStringIncludes(await Deno.readTextFile("/tmp/wiki-export-test/index.md"), "# Open Brain wiki");
        assertStringIncludes(await Deno.readTextFile("/tmp/wiki-export-test/log.md"), "compile #3 ok");
        await Deno.remove("/tmp/wiki-export-test", { recursive: true });
      });

      await t.step("drop_sources removes an uncited source only", async () => {
        const body = "Closure date is 2026-10-10 [#2][#1]. See [[hotel-sofia]].\n\n## Notes\nUnique word zebracorn.\n\nMore text.";
        assertStringIncludes(await ok("wiki_write", { slug: "test-page", body, drop_sources: [1] }), "No change");
        const fixed = body.replace("[#2][#1]", "[#2]");
        assertStringIncludes(await ok("wiki_write", { slug: "test-page", body: fixed, drop_sources: [1] }), "revision 5, 1 sources");
        assertStringIncludes(await ok("wiki_read", { slug: "test-page" }), "sources (1 thoughts): #2");
      });

      await t.step("rebuild rewrites pages from their sources and lint reports duplicates", async () => {
        for (const slug of ["dupe-a", "dupe-b"]) {
          await ok("wiki_write", { slug, title: "Duplicate page", body: "Same words on both pages." });
        }
        const preview = await compile("--rebuild", "hotel-sofia", "--dry-run");
        assertStringIncludes(preview, "===== hotel-sofia =====");
        assertStringIncludes(await ok("wiki_read", { slug: "hotel-sofia" }), "revision 2 |", "dry run must not write");

        fake.writePrompts = [];
        const out = await compile("--rebuild", "hotel-sofia");
        assertStringIncludes(out, "rebuilt hotel-sofia");
        assertMatch(fake.writePrompts[0], /=== \[#5\] captured [^\n]* \(related: filed under another page\) ===/);
        assertMatch(fake.writePrompts[0], /=== \[#1\] captured [^\n]*UTC ===/);
        assert(!/\[#1\] captured [^\n]*related/.test(fake.writePrompts[0]), "own sources are not marked related");
        assertStringIncludes(out, "Possible duplicates: [[dupe-a]] and [[dupe-b]] (1.00)");
        assertStringIncludes(out, "rebuild #4 ok: Rebuilt 1 page(s) from their sources.");
        assertStringIncludes(await ok("wiki_read", { slug: "hotel-sofia" }), "revision 3 |");
        assertMatch(await ok("wiki_read", { slug: "log" }), /compile #4 ok \(gpt-5\.4-mini\): no new thoughts\. Updated \[\[hotel-sofia\]\]\. Rebuilt 1 page/);
        assertStringIncludes(await ok("wiki_index"), "thoughts up to #5");
        assertMatch((await call("wiki_write", { slug: "x" })).text, /title and body are required/);
        const missing = await new Deno.Command(Deno.execPath(), {
          args: ["run", "--no-lock", "--allow-net", "--allow-env", "--allow-read", "wiki_compile.ts", "--rebuild", "no-such-page"],
          cwd: dir,
          env,
        }).output();
        assertEquals(missing.code, 1);
        assertStringIncludes(new TextDecoder().decode(missing.stderr), "no such page: no-such-page");
      });

      await t.step("unassigned thoughts stay pending; pages without embeddings get one later", async () => {
        fake.failEmbeddings = true;
        try {
          assertStringIncludes(await ok("wiki_write", { slug: "embed-later", title: "Embed later", body: "Text." }), "Created");
        } finally {
          fake.failEmbeddings = false;
        }
        await ok("capture_thought", { content: "Orphan note: Hotel Sofia changed its wifi password policy." });
        fake.dropOrphans = true;
        const partial = await compile();
        assertStringIncludes(partial, "re-embedded 1 page(s)");
        assertStringIncludes(partial, "Not assigned (retried next run): #6.");
        assertStringIncludes(partial, "compile #5 partial");
        assertStringIncludes(await ok("wiki_index"), "thoughts up to #5");
        assertStringIncludes(await ok("wiki_read", { slug: "pending" }), "## [#6] ");

        fake.dropOrphans = false;
        const recovered = await compile();
        assertStringIncludes(recovered, "updated hotel-sofia");
        assertStringIncludes(recovered, "compile #6 ok");
        assertStringIncludes(await ok("wiki_read", { slug: "pending" }), "0 thought(s) after the last compile (#6)");

        await ok("capture_thought", { content: "Orphan note two: Hotel Sofia never gets routed." });
        fake.dropOrphans = true;
        try {
          assertStringIncludes(await compile(), "compile #7 partial");
          const givenUp = await compile();
          assertStringIncludes(givenUp, "Not assigned twice, given up: #7.");
          assertStringIncludes(givenUp, "compile #8 ok");
        } finally {
          fake.dropOrphans = false;
        }
        assertStringIncludes(await ok("wiki_index"), "thoughts up to #7");
      });

      await t.step("wiki_read shows parent and sub-pages from slugs", async () => {
        await ok("wiki_write", { slug: "hotel-sofia-spa", title: "Hotel Sofia spa", body: "Spa plans." });
        await ok("wiki_write", { slug: "hotel-sofia-spa-sauna", title: "Sauna", body: "Sauna plans." });
        assertStringIncludes(await ok("wiki_read", { slug: "hotel-sofia" }), "parent: none | sub-pages: [[hotel-sofia-spa]]\n");
        assertStringIncludes(
          await ok("wiki_read", { slug: "hotel-sofia-spa" }),
          "parent: [[hotel-sofia]] | sub-pages: [[hotel-sofia-spa-sauna]]",
        );
        assert(!(await ok("wiki_read", { slug: "test-page" })).includes("sub-pages:"), "unrelated pages get no family line");
        fake.writePrompts = [];
        await compile("--rebuild", "hotel-sofia", "--dry-run");
        assertStringIncludes(fake.writePrompts[0], "Sub-pages of this page (slug | title | summary):\nhotel-sofia-spa | Hotel Sofia spa |");
        const subPages = fake.writePrompts[0].split("Sub-pages of this page")[1].split("\n\n")[0];
        assert(!subPages.includes("hotel-sofia-spa-sauna"), "only direct sub-pages are listed");
      });
    } finally {
      server.kill("SIGTERM");
      await server.status;
      await openai.shutdown();
    }
  },
});
