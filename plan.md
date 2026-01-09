# Implementation Plan — HDB Comparable Flats Agent (Agentic Search MVP)

## 0) MVP Goal (what you demo)

A chat-driven app that takes a natural-language request like:

> “Find resale comps for a 4-room in Sengkang, ~95 sqm, mid-floor, long remaining lease, last 12 months.”

…and returns:

* **Top comparable transactions** (ranked)
* **Price stats** (median / range) for the comp set
* **Chart:** **price distribution histogram** for the comp set (matplotlib)
* **Agent trace** showing iterative query refinement (plan → execute → observe → adjust)
* **Hybrid retrieval** trace line showing when semantic (vector) retrieval with optional lexical boost was used

**MVP scope decisions (agreed):**
* Keep the demo to: top comps + price stats + histogram + trace (skip “why these comps” breakdown and refine chips).
* Hard filters: `town`, `flat_type`, `months_back`; ask a clarifying question if `town` or `flat_type` is missing.
* Soft preferences (scored/relaxed): `floor_area_sqm` (tolerance), `storey_preference`, `min_remaining_lease`, `flat_model`, `price_budget_max`.
* If results are too few: Planner Agent chooses a relaxation adjustment; if still sparse, show available comps with a short “broaden search” note.
* Data scope: last 5 years of resale data across all towns; updates are manual (re-download CSV + rerun ingest).
* Model/config source of truth: `.env` loaded via `pydantic-settings` (single source of truth, no runtime config writes).
* Latency budget/caching decisions deferred for MVP.
* Max relax limit: 4 steps (one adjustment per loop, chosen by Planner Agent); if still <30 comps, stop and show available comps with a short “broaden search” note.
* Max tighten limit: 4 steps (one adjustment per loop, chosen by Planner Agent); if still >200 comps, stop and prompt for more constraints (e.g., town, sqm, price, lease).

Dataset: HDB resale flat prices (CSV with `month,town,flat_type,storey_range,floor_area_sqm,remaining_lease,resale_price,...`).

---

## 1) Architecture Overview

**Frontend:** Gradio
**Backend:** Python + PydanticAI
**DB access:** SQLAlchemy engine only (no ORM)
**LLM:** OpenRouter (config/env-driven)
**DB:** Postgres 17 (Docker) + `pg_textsearch` (BM25 text search extension) + `pgvector`

**Key design principle:** The LLM never “hallucinates comps” — it only:

1. extracts intent (`Target`) and decides action + adjustment steps (accept/clarify/relax/tighten)
2. summarizes results returned by the loop orchestrator

**Hybrid retrieval principle:** keep correctness via structured filters, improve robustness via semantic retrieval over “listing text”.

```text
┌──────────────────────────────────────────────────────────────────────────────────────────────┐
│                           HDB Comparable Flats Agent (Agentic Search MVP)                    │
└──────────────────────────────────────────────────────────────────────────────────────────────┘

   User
    │  natural-language query
    v
┌───────────────────────────────┐
│          Gradio UI            │
│  - Chat input + examples      │
│  - Strictness slider          │
│  - Results table + summary    │
│  - Histogram panel            │
│  - Agent trace panel          │
└───────────────┬───────────────┘
                │ request/response (session state)
                v
┌───────────────────────────────────────────────────────────────────────────────────────────────┐
│                                 Python Backend (app.py)                                       │
│                                                                                               │
│  ┌──────────────────────────────┐     ┌──────────────────────────────────────────────────┐    │
│  │     Config / Model Setup     │     │     Loop Orchestrator (agent/orchestrator.py)    │    │
│  │  - .env via pydantic-settings│     │      (orchestrator, not an LLM agent)            │    │
│  └──────────────┬───────────────┘     └───────────────┬──────────────────────────────────┘    │
│                 │                                     │                                       │
│                 │ LLM calls (Target + planner steps)  │ direct tool calls (SQL-only retrieval)│
│                 v                                     v                                       │
│       ┌───────────────────────┐              ┌───────────────────────────────────────┐        │
│       │   OpenRouter (LLM)    │              │   DB Access Layer (agent/tools.py)    │        │
│       │  - Target agent       │              │  1) count_candidates(filters)         │        │
│       │  - Planner agent      │              │  2) fetch_candidates(filters, limit)  │        │
│       │  - summarize results  │              │  3) price_stats(filters)              │        │
│       │  (never invent comps) │              │  4) explain_facets(filters)           │        │
│       └───────────────────────┘              │  5) hybrid_fetch_candidates(...) (opt)│        │
│                                              └───────────────┬───────────────────────┘        │
│                                                              │ parameterized SQL              │
└──────────────────────────────────────────────────────────────┼────────────────────────────────┘
                                                               v
┌──────────────────────────────────────────────────────────────────────────────────────────────┐
│                               Postgres (HDB Resale Store)                                    │
│  Table: hdb_resale                                                                           │
│  - raw cols: month,town,flat_type,block,street_name,storey_range,flat_model,...              │
│  - derived: month_date, storey_min/max/mid, remaining_lease_months                           │
│  Indexes: btree (town,flat_type,month_date,...), pg_textsearch BM25                          │
│                                                                                              │
│  pgvector                                                                                    │
│  - listing_text (concat fields)                                                              │
│  - listing_embedding VECTOR(d) + ANN index (HNSW)                                            │
│  - hybrid query: hard filters + ORDER BY listing_embedding <=> query_vec                     │
└──────────────────────────────────────────────────────────────────────────────────────────────┘
                ▲
                │ (Day 1 ingestion + embeddings + pgvector schema/index)
                │
┌──────────────────────────────────────────────────────────────────────────────────────────────┐
│                             Data / Jobs (db/*.py + schema.sql)                               │
│  ingest.py: CSV -> COPY/to_sql -> derived columns -> indexes + embeddings                    │
└──────────────────────────────────────────────────────────────────────────────────────────────┘


OUTPUT PIPELINE (what the demo shows)
------------------------------------
Agent loop finds candidate pool (30–200) -> fetch rows (<=500) -> deterministic rerank (agent/scoring.py)
-> stats (median/IQR/range) -> viz histogram (viz/plots.py, matplotlib) -> UI renders results + trace

TRACE: Each step logs
- filters applied, count observed, decision (accept/clarify/relax/tighten + adjustment)
- retrieval_mode: structured | hybrid (vector + optional lexical)
- (opt) query_text_used_for_embedding + topk
```

---

## 1.1) Code Architecture: Modular Agent Design

The agent package is split into seven focused modules for maintainability
(four LLM-facing modules + three supporting modules):

### Module Relationship Diagram

This diagram shows the four LLM-facing modules plus the DB tools module;
supporting modules are listed below.

```
┌─────────────────────────────────────────────────────────────────────┐
│                        agent/orchestrator.py                          │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │ run_iterative_search(query) → SearchResponse                   │ │
│  │                                                                 │ │
│  │  1. Call Target Agent to extract intent                        │ │
│  │     ↓                                                           │ │
│  │  2. Build filters from Target                                  │ │
│  │     ↓                                                           │ │
│  │  3. Count candidates (DB query)                                │ │
│  │     ↓                                                           │ │
│  │  4. Loop: Call Planner Agent to decide action                  │ │
│  │     ↓                                                           │ │
│  │  5. Apply adjustments (relax/tighten)                          │ │
│  │     ↓                                                           │ │
│  │  6. Rebuild filters, recount                                   │ │
│  │     ↓                                                           │ │
│  │  7. Repeat until accept/clarify or max iterations              │ │
│  │     ↓                                                           │ │
│  │  8. Fetch candidates and return SearchResponse                 │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                                                                       │
│  Imports:                                                             │
│  - agent/target.py → get_target_agent(), _target_to_filters()       │
│  - agent/planner.py → get_planner_agent(), apply_adjustments()      │
│  - agent/tools.py → count_candidates, fetch_candidates, price_stats │
│    explain_facets, hybrid_fetch_candidates                         │
└─────────────────────────────────────────────────────────────────────┘
                           │                    │
                           ↓                    ↓
        ┌──────────────────────────┐  ┌──────────────────────────┐
        │   agent/target.py        │  │   agent/planner.py       │
        ├──────────────────────────┤  ├──────────────────────────┤
        │ Agent Creation:          │  │ Agent Creation:          │
        │ • get_target_agent()     │  │ • get_planner_agent()    │
        │                          │  │                          │
        │ Filter Building:         │  │ Decision Logic:          │
        │ • _target_to_filters()   │  │ • _planner_payload()     │
        │ • _normalize_text()      │  │ • _filter_conflicts()    │
        │ • _normalize_flat_type() │  │                          │
        │ • _has_text()            │  │ Adjustments:             │
        │                          │  │ • _available_relax_*()   │
        │ Uses:                    │  │ • _available_tighten_*() │
        │ • TARGET_PROMPT          │  │ • _apply_relax_*()       │
        │   (from agent/prompts.py)│  │ • _apply_tighten_*()     │
        │                          │  │ • _next_relax_value()    │
        │                          │  │ • _next_tighten_value()  │
        │                          │  │                          │
        │                          │  │ Question Building:       │
        │                          │  │ • _clarify_*_note()      │
        │                          │  │ • _build_note()          │
        │                          │  │ • _format_adjustment_*() │
        │                          │  │                          │
        │                          │  │ Uses:                    │
        │                          │  │ • PLANNER_PROMPT         │
        │                          │  │   (from agent/prompts.py)│
        │                          │  │ • Constants (sequences,  │
        │                          │  │   thresholds, labels)    │
        └──────────────────────────┘  └──────────────────────────┘
                    │                              │
                    └──────────────┬───────────────┘
                                   ↓
                        ┌──────────────────────┐
                        │  agent/prompts.py    │
                        ├──────────────────────┤
                        │ • TARGET_PROMPT      │
                        │ • PLANNER_PROMPT     │
                        │                      │
                        │ Pure string          │
                        │ constants only       │
                        └──────────────────────┘
```

### Data Flow Through Modules

```
User Query (string)
    ↓
┌───────────────────┐
│  Target Agent     │  (agent/target.py)
│  (LLM call)       │
└───────────────────┘
    ↓
Target (Pydantic model)
    ↓
┌───────────────────┐
│ _target_to_filters│  (agent/target.py)
└───────────────────┘
    ↓
filters (dict)
    ↓
┌───────────────────┐
│  count_candidates │  (agent/tools.py)
│  price_stats      │
└───────────────────┘
    ↓
count, stats
    ↓
┌───────────────────────────────┐
│  Planner Agent                │  (agent/planner.py)
│  (LLM call with JSON payload) │
└───────────────────────────────┘
    ↓
PlannerDecision (action + adjustment)
    ↓
┌───────────────────┐
│ _apply_relax_     │  (agent/planner.py)
│ _apply_tighten_   │
└───────────────────┘
    ↓
Updated Target
    ↓
(Loop back to filters → count → planner)
    ↓
Fetch candidates (structured or hybrid) → deterministic rerank
    ↓
SearchResponse (final)
```

### Module Responsibilities

**agent/prompts.py**: LLM prompt strings
- `TARGET_PROMPT` - Few-shot examples for Target extraction
- `PLANNER_PROMPT` - Decision-making guidelines for Planner

**agent/target.py**: Target Agent and filter building
- `get_target_agent()` - Returns cached Agent[Target]
- `_target_to_filters()` - Converts Target to SQL filter dict
- Text normalization utilities

**agent/planner.py**: Planner Agent and adjustment logic
- `get_planner_agent()` - Returns cached Agent[PlannerDecision]
- `_planner_payload()` - Builds JSON context for planner
- `_filter_conflicts()` - Validates filter consistency
- Adjustment functions (_available_*, _apply_*)
- Clarifying question builders
- Constants (sequences, thresholds, labels)

**agent/orchestrator.py**: Main iterative search loop
- `run_iterative_search()` - Main entry point
- Loop coordination logic
- Trace building
- Type conversion utilities
- Result pagination

**agent/models.py**: Core data models
- Pydantic models for Target, PlannerDecision, QueryStep, SearchResponse

**agent/tools.py**: DB access tools
- `count_candidates()`, `fetch_candidates()`, `price_stats()`
- `explain_facets()`, `hybrid_fetch_candidates()`

**agent/scoring.py**: Deterministic reranking
- Score computation and top-k selection

---

## 2) Data & Storage Plan

### 2.0 Dockerized Postgres (PG17 + extensions)

Postgres runs in a **Docker container** that already includes **pg_textsearch** and **pgvector**.

* Base image: `postgres:17` (PostgreSQL 17 required)
* **pg_textsearch**: install from pre-built binaries (Timescale release tarball)
* **pgvector**: install per README (build from source against PG17, pin a version tag)
* Add `docker-compose.yml` for local dev (DB service + volume + port 5432)
* Enable extensions in schema:

  * `CREATE EXTENSION IF NOT EXISTS pg_textsearch;`
  * `CREATE EXTENSION IF NOT EXISTS vector;`

Example Dockerfile snippet for **pg_textsearch** prebuilt binaries:

```Dockerfile
ARG PG_MAJOR=17
ARG PG_TEXTSEARCH_VERSION=0.2.0
FROM postgres:${PG_MAJOR}
ARG PG_MAJOR=17
ARG PG_TEXTSEARCH_VERSION=0.2.0

RUN apt-get update \
  && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
  && rm -rf /var/lib/apt/lists/*

RUN set -eu; \
  tmp_dir="/tmp/pg_textsearch"; \
  mkdir -p "$tmp_dir"; \
  curl -fsSL \
    "https://github.com/timescale/pg_textsearch/releases/download/v${PG_TEXTSEARCH_VERSION}/pg_textsearch-pg${PG_MAJOR}-v${PG_TEXTSEARCH_VERSION}.tar.gz" \
    -o /tmp/pg_textsearch.tar.gz; \
  tar -xzf /tmp/pg_textsearch.tar.gz -C "$tmp_dir"; \
  install -m 0755 "$tmp_dir/pg${PG_MAJOR}/pg_textsearch.so" "/usr/lib/postgresql/${PG_MAJOR}/lib/pg_textsearch.so"; \
  install -m 0644 "$tmp_dir/pg${PG_MAJOR}/pg_textsearch.control" "/usr/share/postgresql/${PG_MAJOR}/extension/pg_textsearch.control"; \
  install -m 0644 "$tmp_dir/pg${PG_MAJOR}/pg_textsearch--"*.sql "/usr/share/postgresql/${PG_MAJOR}/extension/"; \
  rm -rf "$tmp_dir" /tmp/pg_textsearch.tar.gz
```

Add a subsequent Dockerfile step to install **pgvector** per the official repo (build from source for Postgres 17, pin the tag).

### 2.1 Ingest

* Download HDB CSV.  https://data.gov.sg/datasets/d_8b84c4ee58e3cfc0ece0d773c8ca6abc/view
* Load into Postgres with COPY (fast) or `pandas.to_sql()` (ok for MVP). Use SQLAlchemy engine only (no ORM).
* Filter to the last 5 years of data during ingest (by `month` or derived `month_date`).

**Table: `hdb_resale`**
Columns (from dataset):

* `month` (YYYY-MM)
* `town`, `flat_type`, `block`, `street_name`, `storey_range`, `flat_model` (text)
* `floor_area_sqm` (numeric)
* `lease_commence_date` (int)
* `remaining_lease` (text; parse into months/years)
* `resale_price` (int)

> Dataset column names align with this schema.

### 2.2 Normalization / Derived Columns

Add derived columns to make search + scoring easy:

* `month_date` (DATE = first day of month)
* `storey_min`, `storey_max` (ints parsed from `storey_range` like “10 TO 12”)
* `storey_mid` (avg)
* `remaining_lease_months` (int) parsed from `remaining_lease` (e.g., “82 years 3 months”)
  * Note: `remaining_lease` uses zero-padded months (e.g., "61 years 04 months") in the source CSV.
  * Note: `storey_range` uses zero-padded ranges like "01 TO 03" in the source CSV.

Tiny parsing examples (Python):

```python
import re

lease_years, lease_months = map(
    int,
    re.findall(r"\d+", "61 years 04 months"),
)
remaining_lease_months = lease_years * 12 + lease_months

storey_min, storey_max = map(int, "01 TO 03".split(" TO "))
```

### 2.3 Indexing (fast + demo-worthy)

* B-tree: `(town, flat_type)`, `month_date`, `floor_area_sqm`, `resale_price`, `remaining_lease_months`
* Lexical search: use pg_textsearch BM25 on `listing_text` for address-like queries
* Composite index for common filters:

  * `(town, flat_type, month_date)`
  * `(town, flat_type, floor_area_sqm)`

### 2.4 `pg_textsearch` usage

Note: pg_textsearch uses its own BM25 index and operators for lexical ranking.

Use it for:

* robust partial matching (e.g., “Sengkang” vs “SENGKANG”)
* street name queries (user enters “Compassvale”)
* **BM25 ranking** for lexical relevance when `street_hint` or free-text is present

**BM25 setup (recommended):**

* Create a pg_textsearch BM25 index on `listing_text` (use the extension's DDL/operator class).
* Query and rank with pg_textsearch operators (e.g., `ORDER BY listing_text <@> :q`).

### 2.5 **pgvector usage (schema in Day 1)**

Add vector retrieval for messy/free-text signals:

* user types “Compassvale area”, “near LRT”, “premium apartment”, partial addresses, typos
* semantic “street/model” matching when exact filters are missing or too strict

**New columns:**

* `listing_text` (TEXT): a deterministic concatenation, e.g.
  `"{town} {flat_type} {block} {street_name} {flat_model} storey {storey_range} {floor_area_sqm}sqm lease {remaining_lease}"`
* `listing_embedding` (VECTOR(d)): dimension depends on your embedding model

**Embedding model decision (agreed):**
* Use `jina-embeddings-v3` for embeddings; generate embeddings offline during ingest and at query-time for the user prompt.
* Store `embedding_model` + dimension in config (or small metadata) so ingest and query remain consistent.
* Soft reference only: `https://github.com/edangx100/adaptive_rag_vercel/blob/main/tools/chromadb_tool.py`

**New indexes (MVP baseline + optional):**

* **Required for MVP:** HNSW on `listing_embedding` for fast ANN search (enables hybrid retrieval)
---

## 3) Agentic Search Behavior (Core of MVP)

### 3.1 User Intent → Structured “Target” (Pydantic model)

```python
class Target(BaseModel):
    town: str | None
    flat_type: str | None
    street_hint: str | None
    floor_area_target: float | None
    floor_area_tolerance: float = 5.0
    storey_preference: Literal["low","mid","high"] | None
    min_remaining_lease_years: int | None
    months_back: int = 12
    price_budget_max: int | None
```

### 3.2 Query Plan representation (Pydantic)

```python
class QueryStep(BaseModel):
    name: str
    hard_filters: dict
    soft_preferences: dict
    action: Literal["search","relax","tighten","rerank","summarize"]
```

### 3.3 Agent Loop (plan → execute → observe → adjust)

Target counts:

* Ideal candidate set size: **30–200 rows** (enough comps, not too broad)

Loop logic (Planner Agent chooses action + adjustment):

1. **Initial search**: strict-ish hard filters (town + flat_type + time window)
2. Observe `count` (+ optional stats/variance)
3. **Planner decision** (Planner Agent): decide `relax` / `tighten` / `accept` / `clarify`
4. If `relax` or `tighten` → planner chooses *which* constraint to adjust
5. Apply that adjustment, rebuild filters, recount
6. Retrieve rows (limit 500), then **rerank** top 10–30 via a deterministic score
7. Summarize results + explain the chosen adjustment

**Relaxation options (planner chooses based on context):**

1. widen time window (12 → 18 → 24 months)
2. widen sqm tolerance (±5 → ±8 → ±12)
3. relax storey preference (mid → any)
4. lower remaining lease minimum
5. soften street hint (drop or widen)

**Tightening options (planner chooses based on context):**

1. narrow time window (24 → 12 → 6 months)
2. narrow sqm tolerance
3. raise remaining lease minimum
4. add/require street hint
5. cap price budget (if provided)

```text
+------------------------------+
| Loop Orchestrator (code)     |
+------------------------------+
  |
  |--calls--> +------------------------------+
  |           | Agent 1: Target Agent        |
  |           | output: Target               |
  |           +------------------------------+
  |                           |
  |                           v
  |                 build filters -> count (N) (+ stats)
  |                           |
  |                           v
  |--context--> +------------------------------+
                | Agent 2: Planner Agent       |
                | input: query + target +      |
                | filters + N + stats + history|
                | output: action + adjustment  |
                +------------------------------+
                           |
                           +--> accept  -> fetch + rerank + stats -> response
                           +--> clarify -> ask user (end or restart)
                           +--> relax/tighten (chosen adjustment)
                                   -> apply adjustment -> rebuild filters
                                   -> recount -> (loop to Planner Agent)
```


 ITERATIVE RETRIEVAL LOOP 
```text
┌──────────────────────────────────────────────────────────────────────┐
│              ITERATIVE RETRIEVAL LOOP (State Machine)                 │
│        Purpose: find a “right-sized” candidate pool (30–200)          │
└──────────────────────────────────────────────────────────────────────┘

                      (start with initial FilterState)
                                   │
                                   v
                      ┌──────────────────────────┐
                      │ [COUNT]                  │
         ────────────►│ N = count_candidates()   │◄───────────────────────────────┐
        │             └─────────────┬────────────┘                                │
        │                           │                                             │
        │                           v                                             │
        │          ┌────────────────────────────────────┐                         ┌──────────────────────────┐
        │          │ [DECIDE] by LLM planner            │────────────────────────►│ [CLARIFY] ask user      │
        │          │   if needs clarification → CLARIFY │                         │ (end or restart)        │
        │          │   else if N < 30 → RELAX           │                         └──────────────────────────┘
        │          │   else if N > 200 → TIGHTEN        │
        │          │   else → ACCEPT (EXIT LOOP)        │
        │          └───────┬───────────────┬────────────┘                         │
        │                  │               │                                      │
        │           N < 30  │               │  N > 200                            │
        │         (too few) │               │ (too many)                          │
        │                  │               │                                      │
        │                  v               v                                      │
        │ ┌────────────────────────┐   ┌──────────────────────────┐               │
        │ │ [RELAX] change ONE rule │  │ [TIGHTEN] change ONE rule│               │
        │ │  1) months 12→18→24     │  │  1) months 24→12→6       │               │
        │ │  2) sqm ±5→±8→±12       │  │  2) narrow sqm tolerance │               │
        │ │  3) storey pref → any   │  │  3) raise lease min      │               │
        │ │  4) lease min → lower   │  │  4) add/require street   │               │
        │ │  5) soften street hint  │  │  5) cap price budget     │               │
        │ └─────────────┬──────────┘   └─────────────┬────────────┘               │
        │               │                            │                            │
        │               └──────────────┬─────────────┘                            │
        │                              │                                          │
        │                              └──────────────────────────────────────────┘
        │                                 (back to COUNT)
        │
        └──────────────────────────────────────────────►  else: 30 ≤ N ≤ 200
                                                          (good enough)
                                                               │
                                                               v
                                                  ┌──────────────────────────┐
                                                  │ [ACCEPT] EXIT LOOP       │
                                                  │ keep current filters     │
                                                  └─────────────┬────────────┘
                                                                │
                                                                v
                              ┌─────────────────────────────────────────────────┐
                              │ NEXT (outside loop): FETCH + RERANK + STATS     │
                              │ rows = fetch_candidates(limit=500) OR           │
                              │        hybrid_fetch(limit=500); rerank top 10–30 │
                              └─────────────────────────────────────────────────┘
```

## 3.3.1 Two-Agent Loop Overview

This project uses **two LLM agents**: one to extract intent (`Target`), and one to decide each loop step (accept/clarify/relax/tighten + adjustment).

```text
User query
   |
   v
Loop Orchestrator (code)
   |--calls--> Agent 1: Target LLM -> Target
   |--builds filters -> count (N) (+ stats)
   |--calls--> Agent 2: Planner Agent -> action + adjustment
   |--if relax/tighten: apply chosen adjustment -> recount (loop)
   |--if accept: fetch + stats -> SearchResponse
   |--if clarify: ask user
```


### 3.4 ** When the agent uses hybrid retrieval**

Trigger hybrid mode when any of these is true:

* `street_hint` is present (address-like / landmark-like query)
* user asks “near/around” without precise street/block
* strict filters yield `count < 30` **and** the agent needs recall
* user provides “free text” constraints not mapped to columns

Behavior:

* Keep hard filters (town/flat_type/month_date) to preserve correctness
* Use vector similarity + optional lexical/BM25 scoring to rank the candidate pool before deterministic rerank
  * If `street_hint` exists, always enable BM25 (strong lexical signal)
  * Use **street_hint** as the embedding query, but use the **original user query** for BM25
  * If BM25 is used, fuse via RRF with configurable weights (defaults: vector 0.3, BM25 0.7)

ASCII diagram (trigger + outcome):

```
                 ┌────────────────────────────────────────────┐
                 │ Check hybrid triggers                      │
                 │ - street_hint                              │
                 │ - "near/around" without exact street/block │
                 │ - strict filters count < 30 + need recall  │
                 │ - free-text constraints not in columns     │
                 └───────────────────────┬────────────────────┘
                                         │ any true?
                              ┌──────────┴──────────┐
                              │                     │
                             yes                   no
                              │                     │
              ┌───────────────┴───────────────┐   ┌─┴────────────────────────────┐
              │ HYBRID MODE                   │   │ NOT TRIGGERED (STRUCTURED)   │
              │ - keep hard filters           │   │ - keep hard filters          │ 
              │ - vector + optional BM25 rank │   │ - no vector/BM25; use        │
              │ - then deterministic rerank   │   │   deterministic rerank only  │ 
              └───────────────────────────────┘   └──────────────────────────────┘
```

ASCII diagram (hybrid query split):

```
User query: "SENGKANG 4 room near Compassvale"
                   |
                   v
         Target extraction (LLM)
                   |
                   v
         street_hint = "Compassvale"
                   |
                   v
        Hybrid retrieval (trigger)
      +-----------------------------+
      | If street_hint exists:      |
      |   BM25 = ON                 |
      |   reason = "street_hint"    |
      +-----------------------------+
          |                     |
          |                     |
          v                     v
 Embedding query           BM25 query text
 = street_hint            = original user query
 ("Compassvale")          ("SENGKANG 4 room near Compassvale")
          |                     |
          v                     v
  Vector search             BM25 search
  (semantic)                (lexical)
          |                     |
          +----------+----------+
                     |
                     v
                  RRF fuse
                     |
                     v
               deterministic rerank
                     |
                     v
                  final results
```

---

### 3.5 Agent Memory & Conversation State (Multi-Turn Clarification)

**Problem:** When the agent asks a clarifying question (e.g., "Which town?"), users naturally reply with just the answer (e.g., "Bedok") rather than repeating the entire query. The agent must remember the original query context to combine it with the user's follow-up response.

**Example flow:**
```
User: "3-room, max 80 sqm, high floor, last 6 months"
Agent: "I need the town. Which town should I use?"
User: "Bedok"  ← just the answer, not a complete query
Agent: (internally combines context from message history)
```

**Implementation: PydanticAI Message History (CORRECT approach)**

PydanticAI provides built-in message history support for multi-turn conversations. Reference: https://ai.pydantic.dev/message-history/#using-messages-as-input-for-further-agent-runs

**Core pattern:**

```python
# First agent run - capture message history
result1 = agent.run_sync('Find 4-room in Bedok')
messages = result1.new_messages()  # or result1.all_messages()

# Subsequent runs - pass message history
result2 = agent.run_sync('Make it high floor', message_history=messages)
```

**Key API details:**
- `result.new_messages()`: Returns only messages from the current run
- `result.all_messages()`: Returns all messages including prior runs
- Pass messages to `message_history` parameter of `run_sync()`, `run()`, or `run_stream()`
- When `message_history` is non-empty, the agent assumes a system prompt already exists and won't add a new one

**When to use `new_messages()` vs `all_messages()`:**

Understanding the difference between these methods is critical for proper multi-turn conversations:

**Use `result.new_messages()`** when:
- **Building message chains incrementally** - Manually managing message history across multiple agent invocations
- **You passed message_history to the agent** - The result already contains prior context, so `new_messages()` gives you only the new additions
- **Example use case**: Chaining multiple agent runs where each run receives the accumulated history

```python
# Pattern: Incremental message accumulation
result1 = agent.run_sync('Find 4-room in Bedok')
messages = result1.new_messages()  # Get messages from run 1

result2 = agent.run_sync('Make it high floor', message_history=messages)
messages = result2.new_messages()  # Only new messages from run 2
# ❌ WRONG: messages now only has run 2, lost run 1 context!

# ✅ CORRECT: Accumulate properly
messages = []
result1 = agent.run_sync('Find 4-room in Bedok')
messages.extend(result1.new_messages())  # Add run 1 messages

result2 = agent.run_sync('Make it high floor', message_history=messages)
messages.extend(result2.new_messages())  # Add run 2 messages
# Now messages has both runs
```

**Use `result.all_messages()`** when:
- **You want the complete conversation history** - All messages from the current run AND any prior runs passed in
- **First agent invocation OR starting fresh** - When you didn't pass `message_history`, `all_messages()` equals `new_messages()`
- **Storing state between turns** - Simplest pattern for preserving full context in UI state (RECOMMENDED)
- **Example use case**: Orchestrator returning complete history to Gradio

```python
# Pattern: Complete history per turn (RECOMMENDED for UI)
def run_iterative_search(
    query: str,
    message_history: list[Message] | None = None,
) -> SearchResponse:
    target_result = agent.run_sync(query, message_history=message_history)

    # Use all_messages() to get complete history
    complete_history = target_result.all_messages()

    return SearchResponse(
        ...,
        messages=complete_history,  # Return everything for next turn
    )
```

**Key differences summarized:**

| Method | Returns | Use when | Typical pattern |
|--------|---------|----------|-----------------|
| `new_messages()` | Only messages from current `run_sync()` call | Building chains manually, need incremental updates | Manual accumulation with `.extend()` |
| `all_messages()` | Current run + any prior `message_history` passed in | Want complete context, storing state between turns | Simple state storage (recommended for UI) |

**Our implementation choice:**
We use `all_messages()` in the orchestrator because:
1. Simplicity - One method call gives us everything
2. Stateless turns - Each turn is independent, receives full history, returns full history
3. UI-friendly - Gradio state just stores and passes the complete message list
4. No accumulation logic needed - The agent handles merging prior + new messages internally

**Integration with Gradio:**

```python
# In app.py:
from pydantic_ai.messages import Message

conversation_state = gr.State(value={"messages": []})  # Store Message objects, not strings

def run_agent_search(query: str, strictness: float, conversation_state: dict):
    # Retrieve message history from previous turns
    previous_messages = conversation_state.get("messages", [])

    # Run agent with message history for context
    response = run_iterative_search(
        query,
        message_history=previous_messages,
        verbose=True
    )

    # Extract and store updated message history
    # Use all_messages() to accumulate full conversation
    updated_state = {"messages": response.all_messages()}

    return ..., updated_state

# Wire up state as both input and output
search_btn.click(
    run_agent_search,
    inputs=[query_input, strictness_slider, conversation_state],
    outputs=[status, count, stats, results, trace_display, histogram_plot, conversation_state],
)
```

**Orchestrator changes:**

```python
# In agent/orchestrator.py:
def run_iterative_search(
    query: str,
    *,
    message_history: list[Message] | None = None,
    verbose: bool = True
) -> SearchResponse:
    """Parse intent, then iterate on filters until the candidate pool is in range."""
    target_agent = get_target_agent()

    # Pass message history to Target Agent
    target_result = target_agent.run_sync(query, message_history=message_history)
    target = target_result.output

    # Store messages for response
    agent_messages = target_result.all_messages()

    # ... rest of orchestrator logic ...

    return SearchResponse(
        target=original_target,
        filters=filters,
        count=count,
        stats=stats,
        results=trimmed_rows,
        note=note,
        trace=trace,
        messages=agent_messages,  # Include for next turn
    )
```

**SearchResponse model update:**

```python
# In agent/models.py:
from pydantic_ai.messages import Message

class SearchResponse(BaseModel):
    target: Target = Field(..., description="Parsed user intent")
    filters: dict[str, Any] = Field(default_factory=dict, description="Filters used for search")
    count: int = Field(..., ge=0, description="Total matching rows")
    stats: Stats = Field(..., description="Summary price stats for the match set")
    results: list[ResultRow] = Field(default_factory=list, description="Comparable sales")
    note: str | None = Field(default=None, description="Optional note or warning")
    trace: list[TraceStep] = Field(default_factory=list, description="Agent loop trace steps")
    messages: list[Message] = Field(default_factory=list, description="Agent message history for multi-turn context")
```

**Benefits of PydanticAI message history:**
- **Native agent support**: Built into PydanticAI, designed for multi-turn conversations
- **Preserves full context**: Includes system prompts, user messages, model responses, and tool calls
- **Type-safe**: Message objects are properly typed and validated
- **No manual parsing**: Agent automatically handles message format and context
- **Scalable**: Supports complex multi-turn conversations beyond simple clarifications

**Comparison: WRONG vs RIGHT approach**

❌ **WRONG (string concatenation workaround):**
```python
# Manual string concatenation - brittle and loses context
if last_was_clarifying:
    query_to_use = f"{last_query}. {query}"  # Hack!
```

✅ **RIGHT (PydanticAI message history):**
```python
# Native message history - preserves full conversation context
result = agent.run_sync(query, message_history=previous_messages)
```

**Migration path:**
1. Update `SearchResponse` to include `messages` field
2. Update `run_iterative_search()` to accept and return message history
3. Replace Gradio string state with Message list
4. Remove manual string concatenation logic in `app.py`

---

### 3.6 Tracing to BrainTrust (PydanticAI)

Goal: capture the agent's spans and orchestration steps in BrainTrust via OpenTelemetry.

Reference: https://www.braintrust.dev/docs/integrations/sdk-integrations/pydantic-ai

**Minimum setup (from BrainTrust setup page):**

```python
import random
from braintrust.otel import BraintrustSpanProcessor
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from pydantic_ai import RunContext
from pydantic_ai.agent import Agent

# Set up tracing for the agent to automatically log to Braintrust
provider = TracerProvider()
trace.set_tracer_provider(provider)

provider.add_span_processor(BraintrustSpanProcessor())

Agent.instrument_all()

# Create your agent
# note i am not using openai in this project.
agent = Agent(
    "openai:gpt-4o",
    system_prompt=(
        "You are a dice game host. Roll the dice for the player and check if their guess matches. "
        "Always include the player's name in the response."
    ),
)

# Define tool calls for your agent (example code from BrainTrust docs)
@agent.tool_plain
def roll_dice() -> str:
    """Roll a six-sided die and return the result."""
    return str(random.randint(1, 6))

@agent.tool
def get_player_name(ctx: RunContext[str]) -> str:
    """Get the player's name."""
    return ctx.deps

# `deps` is a way to inject any extra data your tools might need during this run
dice_result = agent.run_sync("My guess is 4", deps="Anne")
print(dice_result)
```

**Project integration notes:**

* Initialize tracing once at app startup before creating agents.
* Keep BrainTrust keys in env/config (no hard-coding).
* Ensure the agent + tools in `agent/` are instrumented for trace visibility in the UI.

---

## 4) Retrieval, Stats, and Ranking

### 4.1 Tool Functions (called by the loop orchestrator)

Implement as PydanticAI “tools” (invoked directly by the loop orchestrator):

1. `count_candidates(filters) -> int`
2. `fetch_candidates(filters, limit) -> list[Row]`
3. `price_stats(filters) -> Stats` (median, p25/p75, min/max)
4. `explain_facets(filters) -> dict` (counts by storey_range / flat_model)

**One hybrid tool:**
5. `hybrid_fetch_candidates(filters, query_text, limit=200) -> list[Row]`

Minimal promise: it returns rows already *rough-ranked* by retrieval relevance; your deterministic scoring still produces the final top 10–30.

Execution note: tools run parameterized SQL via the SQLAlchemy engine (no ORM models).

### 4.2 SQL Patterns

**Count:**

* `SELECT COUNT(*) FROM hdb_resale WHERE ...`

**Fetch (with limit):**

* `SELECT ... FROM hdb_resale WHERE ... ORDER BY month_date DESC LIMIT 500`

**Stats:**

* `percentile_cont(0.5) WITHIN GROUP (ORDER BY resale_price)` etc.

**Hybrid fetch (filters + vector order):**

* `SELECT ... FROM hdb_resale WHERE <hard filters> ORDER BY listing_embedding <=> :query_vec LIMIT :limit`

Optional lexical boost (keep MVP simple; either-or):

* use `street_name ILIKE '%hint%'`
* or pg_textsearch BM25 over `listing_text` (operator syntax per the extension)

### 4.3 Deterministic Scoring (transparent + fast)

Score each row with weighted distance:

* `area_score = abs(floor_area_sqm - target_area) / tolerance`
* `lease_score = max(0, (min_lease_months - remaining_lease_months) / 12)`
* `storey_score = distance(mid/low/high)`
* `recency_score = months_since(month_date) / months_back`

Final:

* `score = 0.45*area_score + 0.25*lease_score + 0.15*storey_score + 0.15*recency_score`
  Lower is better.

---

## 5) Chart: Price Distribution Histogram (matplotlib)

### 5.1 What the chart shows

A histogram of `resale_price` for the **final comp set** (after filters, before/after ranking is optional). Overlay:

* vertical line at **median**
* optional lines at **p25/p75**

### 5.2 Implementation approach

* After `fetch_candidates()` (or after agent decides final filters), pass the comp `resale_price` list to a helper:

  * `plot_price_hist(prices: list[int], stats: Stats) -> matplotlib.figure.Figure`
* Render in Gradio via `gr.Plot()` or `gr.Image()` (figure to PNG).

### 5.3 MVP-level decisions

* Keep bins simple (`bins="auto"` or fixed like 20–30)
* Title includes key constraints: `{town} {flat_type} last {months_back} months, n={count}`
* Ensure chart updates whenever user refines query

---

## 6) Gradio UI Plan

### 6.1 Layout

Single page with 4 panels:

**A) Chat input**

* Textbox + example prompts
* “Strictness” slider (0–1) that adjusts tolerances and relax rules

**B) Results**

* Table: month, town, flat_type, street_name, storey_range, floor_area_sqm, remaining_lease, resale_price
* Summary card: median price, IQR, min/max, count of comps

**C) Chart**

* Matplotlib histogram for comp `resale_price` distribution

**D) Agent Trace (must-have)**

* Expandable list of steps:

  * Step name
  * SQL filters (human-readable)
  * count returned
  * decision (“too many, narrowed sqm band”)
* retrieval mode label: `structured` vs `hybrid` (vector + optional lexical)

---

## 7) Model Configuration (OpenRouter)

Settings source of truth: `.env` loaded by `pydantic-settings` (`BaseSettings` + `SettingsConfigDict(env_file=".env")`).
To keep `.env` authoritative, override `settings_customise_sources` to use dotenv as the primary source
(optionally keeping `init_settings` for tests).

Example `.env` keys:

* `OPENROUTER_API_KEY`
* `OPENROUTER_MODEL_NAME`
* `OPENROUTER_FALLBACK_MODELS` (comma-separated)
* `EMBEDDING_MODEL_NAME` (for ingest + query embeddings)

Implementation:

* A small `LLMClientFactory` that returns a PydanticAI client based on `BaseSettings`.

---

## 8) Deployment Plan — Fly.io (CLI run by Codex / Claude Code)

Codex / Claude Code will do the deployment for you using the Fly.io CLI (`fly` / `flyctl`). This section is the exact checklist/commands they should execute, aligned with Fly Docs. ([Fly][1])

### 8.1 Prerequisites

1. Install `flyctl` (per Fly docs / installer).
2. Authenticate:

   ```bash
   fly auth login
   ```

### 8.2 Make the Gradio server Fly-compatible

Fly routes traffic to the `internal_port` configured in `fly.toml`. Your app must:

* bind host to **`0.0.0.0`** (not `127.0.0.1`)
* listen on the same port as `internal_port` (commonly `8080`, or use `$PORT`)

Note: if `fly launch` can’t detect a framework/EXPOSE port, it defaults `internal_port` to **8080**; mismatches cause “app not listening on expected address/port” failures. ([Fly][2])

### 8.3 Initialize Fly app config (create `fly.toml`, no deploy yet)

From the repo root:

```bash
fly launch --no-deploy
```

`fly launch` sets you up with a `fly.toml` and defaults for a running Fly app. ([Fly][1])
`flyctl` uses `fly.toml` as the app config by default when commands run in the directory. ([Fly][3])

**Port alignment step (must-do):**

* Ensure `fly.toml` contains the correct service section (typically `http_service`)
* Set:

  * `http_service.internal_port = 8080` (or whatever your Gradio server listens on)

### 8.4 Provision Postgres on Fly (Dockerized PG17 + extensions)

Use a **custom Postgres Docker image** (PG 17 + `pg_textsearch` + `pgvector`) instead of managed Fly Postgres so the extensions are guaranteed.

* Create a separate Fly app for the DB (or run via Docker Compose locally)
* Build/deploy from your Postgres Dockerfile
* Attach a persistent volume for `/var/lib/postgresql/data`
* Set `DATABASE_URL` in the app to point at the internal Fly hostname (e.g. `<pg-app>.internal`)

### 8.5 Set secrets (OpenRouter + config)

Use Fly secrets for:

* `OPENROUTER_API_KEY`
* `DEFAULT_MODEL` / `MODEL_NAME`
* any additional config

Example:

```bash
fly secrets set OPENROUTER_API_KEY="..." MODEL_NAME="..."
```

### 8.6 Deploy

Deploy from repo root:

```bash
fly deploy
```

### 8.7 Verify and debug

* View logs:

  ```bash
  fly logs
  ```
* If the service is unreachable, re-check:

  * Gradio binds to `0.0.0.0`
  * app listens on the same port as `http_service.internal_port`
  * `internal_port` mismatch from launch defaults ([Fly][2])

### 8.8 DB schema initialization + ingest (MVP approach)

Two practical MVP options:

**Option A: one-off run via SSH console**

* SSH into the machine and run:

  * `psql` to execute `db/schema.sql`
  * `python db/ingest.py` to load/transform

**Option B: release command (optional)**

* Configure a release step in `fly.toml` to run migrations on deploy (useful, but optional for 3-day MVP). ([Fly][3])

---

## 9) Day-by-Day Build Plan (3 days)

### Day 1 — Data + DB + baseline search

* Build/run Dockerized Postgres 17 with `pg_textsearch` + `pgvector` extensions
* Add `listing_text` + `listing_embedding` columns (before ingest)
* Load CSV → Postgres (`hdb_resale`) and generate `listing_text` + embeddings during ingest (no backfill step)
* Create derived columns (storey + remaining lease months)
* Add indexes (btree + HNSW)
* Implement Python DB layer + basic SQL retrieval
* Build Gradio page with a **manual filter form** (town/flat_type/sqm) to verify data

Deliverable: You can query and show comps without the LLM.

### Day 2 — Agentic loop + trace

* Implement Pydantic models: `Target`, `QueryStep`
* Implement tools: `count_candidates`, `fetch_candidates`, `price_stats`
* Implement agent loop: refine based on counts
* Add trace panel in Gradio
* Set up BrainTrust tracing (OpenTelemetry + PydanticAI instrumentation)

Deliverable: Type a natural-language prompt → agent runs iterative retrieval → shows trace.

---

## ✅ Day 2.5 — **pgvector mini-checklist** (hybrid tool + trace updates)

### A) Implement **one** hybrid retrieval tool (45–90 min)

Add tool #5:

* `hybrid_fetch_candidates(filters, query_text, limit=200)`

**Minimal behavior (MVP):**

1. Embed `query_text` → `query_vec`
2. SQL:

   * apply hard filters: town/flat_type/month window (+ any numeric constraints you already use)
  * `ORDER BY listing_embedding <=> :query_vec`
   * `LIMIT :limit`
3. Return rows to agent for deterministic rerank.

**Nice-but-still-small:**

* If `street_hint` exists, also apply a weak lexical filter:

  * `listing_text ILIKE '%street_hint%'` or a pg_textsearch BM25 filter
  * If too restrictive, drop it (and log that decision in trace)

**Exit criteria:** tool returns plausible candidates even when street names are partial/typoed.

### B) Trace updates (30–45 min)

Add 2–3 fields to each trace step:

* `retrieval_mode`: `structured` | `hybrid`
* `query_text_used_for_embedding`: (shortened)
* `topk`: e.g. 200
* `filters_applied`: already present, keep it

Show in UI:

* A line like: **“Hybrid retrieval: vector rank within filtered pool (k=200, optional lexical boost)”**
* When it triggers: show the reason (“street hint present”, “too few comps → boosting recall”)

**Exit criteria:** Demo visibly shows *when* the agent switched retrieval mode and why.

### C) Two “pgvector demo prompts” (10 min)

Add to README + UI examples:

1. “Find comps for 4-room near Compassvale, ~95 sqm, last 12 months”
2. “Sengkang 4-room, premium apartment-ish, mid floor, long lease”
   *(the “ish” is deliberate to show semantic retrieval, then deterministic rerank)*

---

### Day 3 — Ranking + chart + polish

* Add deterministic ranking + explanations
* Add **matplotlib histogram** panel + median/IQR overlays
* Add README with screenshots + architecture diagram + example trace + chart screenshot
* Add 3 canned demo prompts + expected behavior

Deliverable: Portfolio-ready MVP with strong agentic search + visualization.

---

## 10) Testing & Evaluation (lightweight but credible)

Create 10 test prompts and log:

* final comp count (target range 30–200)
* whether it asked a clarifying question when key info missing (town/flat_type)
* latency (count + fetch + rerank + plot)
* trace completeness (steps, decisions)
* hybrid retrieval triggered? (Y/N) and did it improve `count` / match quality?

Include `pytest` for:

* parsing `remaining_lease`
* parsing `storey_range`
* scoring correctness
* SQL filter building safety (parameterized queries)
* plot function doesn’t crash on small n / empty sets (shows friendly message)
* hybrid tool doesn’t crash if listing_embedding missing / returns empty (graceful fallback to structured fetch)

---

## 11) Repo Structure (suggested)

```
hdb-agentic-search/
  app.py                 # Gradio UI
  agent/
    orchestrator.py      # iterative search loop
    target.py            # Target agent + filter building
    planner.py           # Planner agent + adjustments
    prompts.py           # LLM prompt strings
    models.py            # Pydantic models
    tools.py             # DB tools exposed to agent
    scoring.py           # deterministic scoring
  viz/
    plots.py             # matplotlib histogram helpers
  db/
    Dockerfile         # Postgres 17 + pg_textsearch + pgvector
    schema.sql
    ingest.py            # ingest + derived columns + embeddings
    queries.py
  docker-compose.yml
  .env
  .env.example
  README.md
  tests/
```

---

## Stretch Goals (only if time remains)

* Nearby-town relaxation when comps are too few (simple adjacency list JSON)
* Optional vector search: embed street_name + flat_model + (maybe) user notes
* Cache frequent queries (town+flat_type+months_back)

[1]: https://fly.io/docs/reference/fly-launch/?utm_source=chatgpt.com "Fly Launch overview · Fly Docs"
[2]: https://fly.io/docs/getting-started/troubleshooting/?utm_source=chatgpt.com "Troubleshoot your deployment"
[3]: https://fly.io/docs/reference/configuration/?utm_source=chatgpt.com "App configuration (fly.toml) · Fly Docs"

---

## MVP Questions and Answers (Agreed)

1) What’s the exact user outcome for the demo?
   Answer: Top comps + price stats + histogram + trace only; skip “why these comps” breakdown and refine chips.
2) What constraints are hard filters vs. soft preferences?
   Answer: Hard filters are `town`, `flat_type`, `months_back`; ask a clarifying question if `town` or `flat_type` is missing. Soft preferences are `floor_area_sqm` (tolerance), `storey_preference`, `min_remaining_lease`, `flat_model`, `price_budget_max`.
3) What is the “too few comps” behavior?
   Answer: Planner Agent chooses a relaxation adjustment with a max of 4 steps (one adjustment per loop); if still sparse, show available comps with a short “broaden search” note.
4) Which embedding model and how are embeddings handled?
   Answer: Use `jina-embeddings-v3`; generate embeddings offline during ingest and at query-time for user prompts; store model + dimension in `.env` (soft reference: https://github.com/edangx100/adaptive_rag_vercel/blob/main/tools/chromadb_tool.py).
5) What is the minimum data scope and update plan?
   Answer: Last 5 years across all towns; manual refresh by re-downloading CSV and rerunning ingest.
6) What latency budget/caching is required?
   Answer: Deferred for MVP.
7) What is the source of truth for model/config selection?
   Answer: `.env` via `pydantic-settings`; no runtime config writes.
