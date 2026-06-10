# Toolgate — Build Plan

*A dynamic tool-surface gateway for MCP, growing into a dynamic harness engine.*
*Drafted 2026-06-10.*

## The thing being built

An MCP gateway that sits between any agent (Claude Code, Cursor, custom harnesses) and N
downstream MCP servers. Instead of forwarding 100+ tool schemas into the model's context, it:

- exposes ~3 meta-tools: `search_tools`, `load_tool`, `call_tool`
- retrieves relevant tools per step, loads schemas lazily
- journals everything
- (later) tunes its own exposure policy against evals

**Honest caveat to design around:** Claude Code already does deferred tool loading internally,
and the MCP spec is moving in this direction. Differentiation is:
(a) bringing it to *every* MCP client, (b) eval/benchmark numbers proving it,
(c) the policy/autotuning layer nobody has. Ship fast — the window is real but not infinite.

## Stack decision

**TypeScript.** MCP SDK is first-class there, target clients (Claude Code, Cursor, VS Code)
live in that ecosystem, distribution is one `npx` command. Python port later if demand appears.
Eval harness can be Python (tau-bench and friends are Python) — that split is fine.

---

## Phase 0 — Prove the pain with numbers (week 1)

Before writing the gateway, write the measurement. This becomes the README headline and the
eval baseline.

- [ ] Script that connects to 10–15 popular MCP servers (GitHub, Slack, Postgres, Playwright,
      Sentry, Notion, filesystem…), dumps all tool schemas, counts tokens.
- [ ] Produce the table: "GitHub MCP alone: ~50 tools, ~XXk tokens. Typical 4-server setup:
      XXk tokens before the user types anything."
- [ ] Small experiment: same multi-step task with full tool surface vs. hand-pruned top-5
      tools — measure tool-choice accuracy and cost across ~20 runs.

**Exit criterion:** numbers compelling enough you'd retweet them. If the deltas are weak,
you've cheaply learned to pick a different wedge. **This is the kill-point.**

## Phase 1 — MVP gateway (weeks 2–4)

Five components:

1. **Registry** — connects downstream to N MCP servers (stdio + HTTP), collects
   tools/resources, handles server lifecycle and `listChanged` re-syncs.
2. **Indexer** — hybrid retrieval over tool name + description: BM25 (zero-dependency,
   works offline) plus optional embeddings. Start BM25-only; embeddings are a config flag later.
3. **Facade** — what the agent sees:
   - `search_tools(query)` → ranked names + one-line descriptions
   - `load_tool(names)` → full schemas
   - `call_tool(name, args)` → proxied result
   - plus a **passthrough mode** (forward everything unchanged) so adoption is zero-risk.
4. **Executor** — routes calls to the right downstream server, namespaces collisions,
   enforces per-call timeout.
5. **Config** — one `toolgate.json`: server list, mode (passthrough / search / auto),
   pinned always-loaded tools, k.

Scope discipline: **no** journaling, policy, or UI yet.

**Exit criterion:** run Claude Code through toolgate against 5 real servers for your own
daily work for a week without it getting in the way.

## Phase 2 — Eval harness (weeks 4–6)

The credibility phase, and the core skill being built.

- [ ] Thin eval runner: task spec (prompt + downstream servers + grader) → N runs → metrics.
      Graders programmatic (state checks, not LLM-judged, wherever possible).
- [ ] Adapt **tau-bench** (tool-use reliability is its whole point).
- [ ] Author ~20 custom multi-server tasks ("find the Sentry issue, locate the offending
      commit in GitHub, post a summary to Slack" style).
- [ ] Metrics: context tokens at first model call, end-to-end cost, tool-choice accuracy,
      task **pass^k** (k=4 — reliability, not luck), latency overhead added by the gateway.
- [ ] Run the matrix: {no gateway, passthrough, search mode} × {3 models}.
      Publish everything — methodology, harness, raw transcripts.

**Exit criterion:** results table showing meaningful token reduction with equal-or-better
pass^k. If search mode *hurts* accuracy, that's a finding too — fix retrieval before proceeding.

## Phase 3 — Launch (weeks 6–7)

- [ ] README with Phase 0 pain numbers and Phase 2 results up top; `npx toolgate` one-liner;
      configs for Claude Code, Cursor, VS Code.
- [ ] One blog post: "Your MCP servers are eating your context window" — the problem, the
      numbers, the fix. Post to HN / r/LocalLLaMA / X.
- [ ] License MIT/Apache-2.0. Be responsive to issues for the first two weeks — early
      adopters become contributors.

## Phase 4 — Journal + observability (weeks 7–10)

The AX-shaped layer; moat-building begins.

- [ ] Append-only JSONL journal per session: every search, load, call, result size,
      latency, error.
- [ ] `toolgate stats`: which tools are actually used vs. loaded, dead weight per server,
      failure rates.
- This data is what makes Phase 5 possible — you can't tune policy without traces.

## Phase 5 — Dynamic policy + autotuning (weeks 10–16, research-grade)

- [ ] **Exposure policy as data:** per-step rules — pin tools by task type, decay unused
      tools out of context, adjust k by remaining budget.
- [ ] **Autotuning loop:** treat harness config (k, retrieval weights, pinning rules,
      description rewrites) as parameters; optimize against the Phase 2 eval suite — plain
      grid/random search first, graduate to bandits. "DSPy for tool surfaces."
- [ ] **Tool facade generation:** auto-rewrite verbose tool descriptions into ACI-optimized
      ones, validated by eval deltas. This is a paper *and* a feature.

---

## Risks

| Risk | Mitigation |
|---|---|
| MCP spec standardizes tool search natively | Move fast; eval suite, journal data, and policy layer survive even if the facade commoditizes |
| Retrieval misses the needed tool → silent task failure | Passthrough fallback, pinned tools, `search_tools` returns a "browse all" escape hatch |
| Latency overhead annoys users | Budget <50ms p95 for the proxy hop; measure in Phase 2 |
| Crowded-space drift (becoming "another framework") | Hard rule: toolgate never owns the agent loop. Middleware only |

## Timeline summary

| Phase | Weeks | Deliverable |
|---|---|---|
| 0 — Measure | 1 | Pain numbers, kill/go decision |
| 1 — MVP gateway | 2–4 | Working proxy, daily-driver quality |
| 2 — Evals | 4–6 | Published benchmark matrix |
| 3 — Launch | 6–7 | npm release + blog post |
| 4 — Journal | 7–10 | Traces + usage stats |
| 5 — Dynamic policy | 10–16 | Autotuned exposure policy, possible paper |

~4 months part-time to a launched, benchmarked project. Phases 0–3 (~7 weeks) are the
shippable core.

## Background / reading

- SWE-agent paper — agent-computer interface (ACI) design
- Anthropic: *Building Effective Agents*, *Effective Context Engineering for AI Agents*,
  *How we built our multi-agent research system*
- tau-bench paper — multi-turn tool-use reliability, pass^k methodology
- Temporal / Restate docs — durable execution, journaling, replay (for Phase 4+)
- IBM OpenRAG (Docling + OpenSearch + Langflow) — what packaging-led adoption looks like;
  also what a *static* graph-defined harness looks like, i.e. the contrast point
