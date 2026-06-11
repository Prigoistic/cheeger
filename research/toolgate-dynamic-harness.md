# Toolgate — Dynamic MCP Tool-Surface Gateway

> Reconstructed from session 7dcc6f9c (June 2026) and session aa2c1fa7.
> This is a separate project from the cheeger spectral-segmentation work.
> When built, it should live in its own repository.

---

## The Problem

An MCP-connected agent has 100+ tool schemas loaded into its context window at all
times. Most are irrelevant for any given step. This wastes tokens, degrades decision
quality (the model attends to irrelevant options), and makes it harder to add new
tools without blowing the context budget.

**Measured pain (to verify at Phase 0):**
- Count tokens consumed by tool schemas in a typical deep-research or coding session
- Compare task completion with full schema dump vs. lazy-loaded schemas
- The hypothesis is that 60–80% of context budget goes to schemas that never get called

---

## The Idea

A gateway that sits between any agent (Claude Code, Cursor, custom harnesses) and N
downstream MCP servers. Instead of forwarding all tool schemas into context, it exposes
exactly **three meta-tools**:

```
search_tools(query)  → ranked list of relevant tool names + one-line descriptions
load_tool(name)      → full JSON schema for that tool, loaded on demand
call_tool(name, args) → execute the tool, return result
```

The model asks for what it needs when it needs it. Schemas load lazily. Only relevant
tools enter context.

---

## Architecture

```
Agent (Claude / Cursor / custom)
        │
        ▼
   [ Toolgate gateway ]
   ┌─────────────────────┐
   │  search_tools()     │  ← semantic index over all tool descriptions
   │  load_tool()        │  ← fetches schema from downstream server on demand
   │  call_tool()        │  ← proxies to the real MCP server
   │  journal            │  ← every tool call logged (for evals + replay)
   └─────────────────────┘
        │         │         │
        ▼         ▼         ▼
   [MCP srv A] [MCP srv B] [MCP srv C]   (N downstream servers)
```

**Key properties:**
- Universal: any MCP client works without modification
- Stateless from the agent's perspective: the gateway handles discovery
- Journaled: every search, load, and call is logged for evals and replay
- Later: autotuning — the exposure policy learns which tools to surface for which
  task types based on eval feedback

---

## Build Plan

### Phase 0 — Measure the pain (kill/go point)
Instrument a real Claude Code session. Count tokens consumed by tool schemas vs.
tool executions. If schemas are less than 30% of context usage, the problem is smaller
than expected and the project may not be worth building. If 50%+, proceed.

**Output:** a single number: "X% of context budget is schema overhead."

### Phase 1 — Minimal gateway (2 weeks)
- Python server exposing `search_tools` / `load_tool` / `call_tool` over MCP protocol
- Hard-coded semantic index (just keyword search to start)
- Connects to two real downstream MCP servers
- Proves the round-trip works and a real agent can drive it

**Test:** A Claude Code session completes a coding task using toolgate with 3 real MCP
servers. Schema tokens drop by >40% vs. direct connection.

### Phase 2 — Semantic retrieval (1 week)
- Replace keyword search with a proper embedding index (sentence-transformers or
  OpenAI embeddings)
- Tool descriptions + parameter names become the retrieval corpus
- `search_tools("read a file")` returns `filesystem.read_file` ahead of less relevant tools

### Phase 3 — Journaling + replay (1 week)
- Every call logged to a structured JSONL journal
- `replay(session_id)` re-runs a session against a new model or new tools
- This is the foundation for evals

### Phase 4 — Evals (2 weeks)
- Build eval harness over SWE-bench Lite or a custom task set
- Metric: pass@1, pass@3, tool-call efficiency (calls per task completion)
- Compare: full schema dump vs. toolgate with lazy loading
- This is the publishable result

### Phase 5 — Autotuning exposure policy (4+ weeks)
- Train a small policy (bandit or lightweight RL) on which tools to surface for which
  task types
- Feedback loop: eval outcomes → policy update → re-eval
- Differentiation: this is what makes toolgate more than a proxy

---

## Differentiation

Claude Code already defers tool loading internally and the MCP spec is moving this
direction. The differentiation for toolgate is:

1. **Universality** — works with any MCP client, not just Claude Code
2. **Published eval numbers** — a benchmark that proves the gain is real
3. **The autotuning layer** — dynamic exposure policy vs. static lazy loading

Without (2), toolgate is an engineering convenience, not a contribution.

---

## Adjacent Context

- **SWE-agent ACI paper** — founded the "agent-computer interface" framing: tool
  ergonomics (how you present tools to a model) moves benchmark scores more than
  prompting. Toolgate is an ACI-layer intervention.
- **Anthropic's multi-agent research system** — orchestrator/worker fan-out, subagents
  for context isolation. Toolgate is orthogonal: it operates at the tool-surface layer
  within a single agent.
- **OpenHands, smolagents, Aider** — reference implementations to study for how each
  handles tool schemas, context compaction, error recovery.
- **IBM OpenRAG** — uses Langflow (static graph-defined workflows). The contrast: a
  *dynamic* harness where the model drives control flow over a deterministic substrate.

---

## Key Design Caveat

"MCP client doesn't dump all schemas" is increasingly a first-party feature. Track the
MCP spec changelog. If v1.x adds native lazy loading, Phase 1 is obsoleted and the
value shifts entirely to (2) eval infrastructure and (3) autotuning policy.

---

*Session origins: 7dcc6f9c-6f49-4503-9751-31da3b8cd214 and adjacent sessions,
June 2026. Build in a separate repo — this document is only stored here for
persistence.*
