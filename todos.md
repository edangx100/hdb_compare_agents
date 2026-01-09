# HDB Comparable Flats Agent - Implementation Todos

This todo list breaks down the implementation into small, incremental steps. Each task produces something testable that moves the project forward.

**Working principle:** Complete each task fully and test it before moving to the next. Check off items as you go.

---

## Phase 1: Foundation - Database & Data Layer

### 1.1 Docker Setup for Postgres
- [x] Create `db/Dockerfile` for Postgres 17 with pg_textsearch and pgvector extensions
- [x] Create `docker-compose.yml` for local Postgres service (port 5432, volume mount)
- [x] Test: Run `docker-compose up -d` and verify Postgres is running with `psql --version`

### 1.2 Database Schema (No Embeddings Yet)
- [x] Create `db/schema.sql` with `hdb_resale` table (all CSV columns + derived columns)
- [x] Add derived columns: `month_date`, `storey_min`, `storey_max`, `storey_mid`, `remaining_lease_months`
- [x] Add basic B-tree indexes on `town`, `flat_type`, `month_date`, `floor_area_sqm`, `resale_price`
- [x] Test: Execute schema.sql and verify table structure with `\d hdb_resale`

### 1.3 Basic Data Ingestion (No Embeddings)
- [x] Create `db/ingest.py` to load CSV into Postgres using pandas/SQLAlchemy
- [x] Implement parsing for `remaining_lease` → `remaining_lease_months` (handle "61 years 04 months")
- [x] Implement parsing for `storey_range` → `storey_min`, `storey_max`, `storey_mid` (handle "01 TO 03")
- [x] Filter to last 5 years of data during ingestion
- [x] Test: Run ingestion, verify row count and spot-check derived columns

### 1.4 Basic Query Layer
- [x] Create `db/queries.py` with SQLAlchemy engine setup (from DATABASE_URL env)
- [x] Implement `fetch_flats(filters: dict, limit: int)` function with basic SQL
- [x] Implement `count_flats(filters: dict)` function
- [x] Test: Call functions from Python REPL with sample filters, verify results

### 1.5 Configuration Setup
- [x] Create `settings.py` using `pydantic-settings` with `SettingsConfigDict(env_file=".env")`
- [x] Override `settings_customise_sources` to make dotenv the primary source (keep `init_settings` only for tests)
- [x] Create `.env` as the single source of truth for configuration (database settings, OpenRouter API key)
- [x] Create `.env.example` with required environment variables
- [x] Create `requirements.txt` with initial dependencies (pandas, sqlalchemy, psycopg2-binary, pydantic-settings)
- [x] Test: Instantiate Settings and verify values load from `.env`

---

## Phase 2: Minimal UI - Manual Search Form

### 2.1 Basic Gradio App
- [x] Create `app.py` with minimal Gradio interface
- [x] Add manual filter form: dropdowns for town, flat_type; number inputs for sqm_min, sqm_max
- [x] Add date range filter (months_back slider)
- [x] Wire up filters to `db/queries.py` functions
- [x] Test: Launch app, select filters, see results in a Gradio DataFrame

### 2.2 Results Display
- [x] Display results table with columns: month, town, flat_type, street_name, storey_range, floor_area_sqm, remaining_lease, resale_price
- [x] Add results count display
- [x] Add basic price stats: min, max, median (calculate in Python from results)
- [x] Test: Query for "ANG MO KIO, 4 ROOM, last 12 months" and verify results make sense

---

## Phase 3: Embeddings & Vector Search Foundation

### 3.1 Add Vector Columns to Schema
- [x] Update `db/schema.sql` to add `listing_text` (TEXT) and `listing_embedding` (VECTOR) columns
- [x] Add pgvector extension creation: `CREATE EXTENSION IF NOT EXISTS vector;`
- [x] Create HNSW index on `listing_embedding`
- [x] Test: Run updated schema, verify columns exist with `\d hdb_resale`

### 3.4 Add pg_textsearch BM25 Index
- [x] Add pg_textsearch extension creation to `db/schema.sql`: `CREATE EXTENSION IF NOT EXISTS pg_textsearch;`
- [x] Create BM25 index on `listing_text` using pg_textsearch operator class
- [x] Test: Verify index exists and can be used for BM25 ranking queries
- [x] Test: Run sample BM25 query with `ORDER BY listing_text <@> 'Compassvale'`, verify results are ranked by lexical relevance

### 3.2 Embedding Generation During Ingestion
- [x] Add `jina-embeddings-v3` to requirements.txt (or appropriate client library)
- [x] Update `db/ingest.py` to generate `listing_text` from concatenated fields
- [x] Update `db/ingest.py` to generate embeddings for each row during ingestion
- [x] Store embedding dimension in `.env` (via Settings)
- [x] Test: Re-run ingestion, verify `listing_text` and `listing_embedding` are populated

### 3.3 Hybrid Retrieval Query Function
- [x] Add `hybrid_fetch_candidates(filters, query_text, limit)` to `db/queries.py`
- [x] Implement query-time embedding generation for user text
- [x] Implement SQL with hard filters + vector similarity ordering
- [x] Test: Call function with "Compassvale area 4-room", verify it returns relevant results

python - <<'PY'
from db.queries import hybrid_fetch_candidates
rows = hybrid_fetch_candidates(
    {"town": "SENGKANG", "flat_type": "4 ROOM", "months_back": 12},
    "Compassvale area 4-room",
    limit=10,
)
print(len(rows))
print(rows[:2])
PY


---

## Phase 4: Pydantic Models & Agent Foundation

### 4.1 Core Pydantic Models
- [x] Create `agent/models.py` with `Target` model (town, flat_type, floor_area_target, etc.)
- [x] Create `QueryStep` model (name, hard_filters, soft_preferences, action)
- [x] Create `Stats` model (median, p25, p75, min, max, count)
- [x] Test: Instantiate models with sample data, verify validation works

### 4.2 OpenRouter LLM Setup
- [x] Add PydanticAI to requirements.txt
- [x] Create `agent/llm_client.py` with OpenRouter client factory (using `pydantic-settings` + `.env`)
- [x] Add OpenRouter API key to .env.example
- [x] Test: Create client, run simple sync test call, verify it works

### 4.3 Tool Layer - Basic Retrieval (loop orchestrator uses these)
- [x] Create `agent/tools.py` with tool stub structure
- [x] Implement `count_candidates` tool (calls db/queries.py)
- [x] Implement `fetch_candidates` tool (calls db/queries.py)
- [x] Implement `price_stats` tool (SQL percentiles)
- [x] Test: Call each tool function directly, verify correct data types returned

---

## Phase 5: Agentic Loop - Iterative Refinement

### 5.1 Single Search Flow (agents + loop orchestrator)
- [x] Create `agent/planner.py` with PydanticAI agent setup
- [x] Define system prompts for Target + planner agents (LLM only)
- [x] Implement single-step search: extract Target → loop orchestrator calls tools → return results
- [x] Test: Send "Find 4-room in ANG MO KIO" and verify loop orchestrator calls tools correctly

### 5.2 Planner Agent Decisions (action + adjustment)
- [x] Update planner prompt to choose accept/clarify/relax/tighten + adjustment using context (query + target + filters + count + stats)
- [x] Add `adjustment` to PlannerDecision schema and validate output
- [x] Pass structured context (query/target/filters/count/stats/history) into Planner Agent
- [x] Enforce max 4 relaxation steps limit (one adjustment per loop)
- [x] Test: Query with very specific constraints, verify Planner Agent chooses a relax adjustment
      - PYTHONPATH=. python agent/planner.py "Find a 4-room in Sengkang, 120 sqm, high floor, minimum 95 years remaining lease, last 1 month"
      - PYTHONPATH=. python agent/planner.py "Town SENGKANG, 4 ROOM, 120 sqm, high floor, minimum 95 years remaining lease, last 1 month"

### 5.3 Apply Planner Adjustments (relax/tighten)
- [x] Implement tighten adjustment options (narrow time, narrow sqm, raise lease min, require street hint, cap price)
- [x] Let Planner Agent choose tighten adjustment when count is too high
- [x] Remove fixed relax/tighten priority order; apply Planner Agent's chosen adjustment
- [x] Test: Query with very broad constraints, verify Planner Agent tightens appropriately
      - PYTHONPATH=. python agent/planner.py "Find 4-room flats in ANG MO KIO from the last 5 years"

### 5.4 Clarifying Questions
- [x] Implement logic to ask clarifying questions when town or flat_type is missing
- [x] Allow Planner Agent to return "clarify" when constraints conflict or results are not useful
- [x] Test: Send vague query like "Find a flat, mid-floor", verify agent asks for town/flat_type
      - PYTHONPATH=. python agent/planner.py "Find a flat, mid-floor"
---

## Phase 6: Agent Trace & UI Integration

### 6.1 Trace Data Structure
- [x] Create trace logging structure in loop orchestrator (step name, filters, count, action + adjustment, retrieval_mode)
- [x] Store trace steps in agent run context
- [x] Return trace alongside results
- [x] Test: Run agent, print trace, verify all steps are captured

### 6.2 Update Gradio UI - Chat Interface
- [x] Replace manual filter form with chat textbox
- [x] Add example prompts (3-4 realistic queries)
- [x] Add "Strictness" slider (0-1) to adjust tolerances
- [x] Test: Enter natural language query, verify it reaches agent

### 6.3 Display Agent Trace in UI
- [x] Add expandable trace panel to Gradio UI
- [x] Display each step: name, filters, count, action + adjustment
- [x] Show retrieval_mode (structured vs hybrid)
- [x] Test: Run query, verify trace is visible and readable

---

## Phase 7: Deterministic Ranking & Scoring

### 7.1 Scoring Implementation
- [x] Create `agent/scoring.py` with score calculation function
- [x] Implement area_score, lease_score, storey_score, recency_score
- [x] Implement weighted final score: 0.45*area + 0.25*lease + 0.15*storey + 0.15*recency
- [x] Test: Score sample rows, verify lower scores for better matches

### 7.2 Integrate Ranking into Agent
- [x] Add rerank step after fetch_candidates
- [x] Return top 10-30 ranked results to user
- [x] Test: Query and verify results are ordered by score, not just by retrieval order

---

## Phase 8: Visualization - Price Histogram

### 8.1 Matplotlib Histogram Function
- [x] Create `viz/plots.py` with `plot_price_hist(prices, stats)` function
- [x] Generate histogram with auto bins
- [x] Add vertical line at median
- [x] Add p25/p75 lines
- [x] Add title with town, flat_type, months_back, count
- [x] Test: Call function with sample price list, verify figure renders

### 8.2 Integrate Chart into Gradio UI
- [x] Add gr.Plot() or gr.Image() component for histogram
- [x] Wire up histogram generation after agent returns results
- [x] Test: Run query, verify histogram displays with correct stats overlay

---

## Phase 9: Hybrid Retrieval Mode

### 9.1 Hybrid Tool Implementation
- [x] Add `hybrid_fetch_candidates` to agent/tools.py
- [x] Implement trigger logic (street_hint present OR count < 30)
- [x] Generate query embedding from user text
- [x] Execute vector similarity query with hard filters
- [x] Test: Query "4-room near Compassvale", verify hybrid mode triggers

### 9.2 BM25 Lexical Boost with pg_textsearch
- [x] Update `hybrid_fetch_candidates` to support optional BM25 ranking alongside vector similarity
- [x] Implement BM25 query using pg_textsearch operators with SQL ordering by `bm25_score`
- [x] Add combined ranking strategy: vector + BM25 (Reciprocal Rank Fusion with k=60)
- [x] Add trigger logic for BM25: when `street_hint` is present or partial address/landmark queries detected
- [x] Add query normalization and fallback logic for robustness
- [x] Test: Query with partial street names ("Compassvale" vs "COMPASSVALE"), verify lexical matching improves recall
- [x] Test: Query "near LRT" or "premium apartment", verify BM25 captures keywords not in structured fields

**Implementation notes:**
- BM25 query orders by `bm25_score` in SQL to avoid random LIMIT sampling; Python sort keeps ties stable
- Added fallback logic: if vector fails, use BM25-only; if BM25 fails, use vector-only
- Query normalization ensures consistent processing across both ranking methods
- RRF weights are configurable via `.env` (`RRF_VECTOR_WEIGHT`, `RRF_BM25_WEIGHT`)
- All tests passing with improved case-insensitivity

### 9.3 Update Trace for Hybrid Mode
- [x] Add `retrieval_mode`, `query_text_used_for_embedding`, `topk` to trace steps
- [x] Add `bm25_used` flag and `bm25_boost_reason` to trace steps
- [x] Display hybrid retrieval info in trace panel (vector + optional BM25)
- [x] Test: Run hybrid query, verify trace shows "hybrid" mode and reason
- [x] Test: Run BM25-enhanced query, verify trace shows "vector + BM25" and explains why

      - PYTHONPATH=. python agent/orchestrator.py "SENGKANG 4 room near Compassvale"
      - PYTHONPATH=. python db/test_hybrid_bm25.py

### 9.4 Demo Prompts for Hybrid
- [x] Add demo prompt: "Find comps for 4-room near Compassvale, ~95 sqm, last 12 months"
- [x] Add demo prompt: "Sengkang 4-room, premium apartment-ish, mid floor, long lease"
- [x] Test: Run both prompts, verify semantic retrieval works for partial/fuzzy inputs
- [x] Test: Verify BM25 boost activates for street-hint queries and improves relevance

      Test: Run both prompts, verify semantic retrieval works for partial/fuzzy inputs
      - PYTHONPATH=. python agent/orchestrator.py "Find comps for 4-room near Compassvale, ~95 sqm, last 12 months" 
      - PYTHONPATH=. python agent/orchestrator.py "Sengkang 4-room, premium apartment-ish, mid floor, long lease"
      Test: Verify BM25 boost activates for street-hint queries and improves relevance
      - PYTHONPATH=. python agent/orchestrator.py "Find comps for 4-room near Compassvale, ~95 sqm, last 12 months" 
      - PYTHONPATH=. python db/test_hybrid_bm25.py
---

## Phase 10: BrainTrust Tracing Integration

### 10.1 BrainTrust Setup
- [x] Add braintrust[otel] to requirements.txt
- [x] Add OpenTelemetry dependencies (included in braintrust[otel])
- [x] Add BRAINTRUST_API_KEY and BRAINTRUST_PARENT to .env.example
- [x] Update settings.py with BrainTrust configuration fields

### 10.2 Instrument Agent
- [x] Create agent/tracing.py module for BrainTrust initialization
- [x] Add BraintrustSpanProcessor setup in app.py startup
- [x] Add BraintrustSpanProcessor setup in orchestrator.py for CLI usage
- [x] Call Agent.instrument_all() before agent creation
- [x] Test: Verify graceful fallback when BRAINTRUST_API_KEY not set
- [x] Create BrainTrust account and get API key (user action)
- [x] Add actual BRAINTRUST_API_KEY to .env (user action)
- [x] Fix environment variable handling in tracing.py to ensure BraintrustSpanProcessor can access API key
- [x] Test: Run agent with API key, verify tracing initializes successfully

---

## Phase 11: Testing & Quality

### 11.1 Unit Tests - Parsing
- [x] Create `tests/test_parsing.py`
- [x] Add test for `remaining_lease` parsing (various formats)
- [x] Add test for `storey_range` parsing
- [x] Test: Run pytest, verify all parsing tests pass

### 11.2 Unit Tests - Scoring
- [x] Create `tests/test_scoring.py`
- [x] Add tests for score calculation edge cases
- [x] Add test for ranking order
- [x] Test: Run pytest, verify scoring logic is correct

### 11.3 Integration Tests
- [x] Create `tests/test_queries.py` for database query functions
- [x] Add test for count_candidates with various filters
- [x] Add test for fetch_candidates with limit
- [x] Test: Run pytest with test database, verify queries work

### 11.4 Agent Evaluation
- [x] Create 10 test prompts covering various scenarios
- [x] Run each prompt and log: comp count, latency, trace steps, hybrid mode usage
- [x] Document results in `test_results.md`
- [x] Test: Verify 80%+ prompts return 30-200 comps or explain why not

---

## Phase 12: Polish & Documentation

### 12.1 README with Architecture
- [x] Create comprehensive README.md
- [x] Add architecture diagram (ASCII or mermaid)
- [x] Add setup instructions (Docker, env vars, ingestion)
- [x] Add example queries and expected outputs
- [x] Include screenshots of UI, trace, and histogram

### 12.2 Error Handling
- [x] Add try-catch for database connection failures
- [x] Add graceful handling for missing embeddings
- [x] Add user-friendly error messages in Gradio UI
- [x] Add fallback when hybrid retrieval fails → structured only
- [x] Test: Disconnect DB, verify friendly error message

### 12.3 Final Demo Preparation
- [ ] Ensure all 3 canned demo prompts work smoothly
- [ ] Verify histogram updates correctly for each query
- [ ] Verify trace is clear and informative
- [ ] Test end-to-end flow from fresh Docker start

---

## Phase 12.5: Agent Memory & Multi-Turn Conversations

### 12.5.1 Implement PydanticAI Message History
- [x] Update SearchResponse model to include message history from agent runs
- [x] Modify run_iterative_search() to accept optional message_history parameter
- [x] Return agent message history alongside search results
- [x] Test: Run single query, verify message history is captured from agent runs

### 12.5.2 Replace Gradio String Concatenation with Message History
- [x] Remove manual string concatenation logic from app.py (lines 66-79)
- [x] Update Gradio conversation_state to store PydanticAI messages instead of strings
- [x] Pass message_history to run_iterative_search() on follow-up queries
- [x] Test: Ask clarifying question, then provide answer - verify context preserved

### 12.5.3 Multi-Turn Conversation Flow
- [x] Update run_agent_search() to extract messages from previous agent results
- [x] Implement proper message accumulation across turns (use result.all_messages())
- [x] Handle both Target Agent and Planner Agent message histories separately
- [x] Test: Multi-turn conversation - "Find 4-room in Bedok" → "Make it high floor" → "Within last 6 months"

### 12.5.4 Document Agent Memory Pattern
- [x] Add new section to CLAUDE.md explaining PydanticAI message history pattern
- [x] Document the difference between message_history vs manual string concatenation
- [x] Add code examples showing proper usage of result.new_messages() and result.all_messages()
- [x] Document when to use new_messages() vs all_messages()

---

## Phase 13: Deployment to Fly.io 

### 13.1 Fly.io Configuration
- [ ] Install flyctl CLI
- [ ] Run `fly auth login`
- [ ] Run `fly launch --no-deploy` to create fly.toml
- [ ] Set internal_port to 8080 in fly.toml
- [ ] Update app.py to bind Gradio to 0.0.0.0:8080

### 13.2 Database on Fly
- [ ] Build custom Postgres Docker image for Fly
- [ ] Deploy Postgres as separate Fly app
- [ ] Attach persistent volume
- [ ] Set DATABASE_URL secret to point to Fly Postgres

### 13.3 Deploy Application
- [ ] Set fly secrets: OPENROUTER_API_KEY, MODEL_NAME
- [ ] Run `fly deploy`
- [ ] Run schema.sql via SSH console
- [ ] Run ingest.py via SSH console
- [ ] Test: Access deployed app URL, verify it works end-to-end

### 13.4 Verify Production
- [ ] Run `fly logs` and check for errors
- [ ] Test all 3 demo prompts on production
- [ ] Verify histogram renders correctly
- [ ] Verify BrainTrust traces are captured

---

## Completion Checklist

- [ ] All core features working: chat → agent loop → trace → ranking → histogram
- [ ] Hybrid retrieval triggers correctly for fuzzy/partial inputs
- [ ] BM25 lexical boost (pg_textsearch) enhances hybrid retrieval for street hints and keyword queries
- [ ] Agent asks clarifying questions when needed
- [ ] Tests pass for parsing, scoring, and queries
- [ ] README is complete with screenshots and architecture
- [ ] (Optional) Application deployed to Fly.io and accessible

---

**Next Step:** Start with Task 1.1 (Docker Setup for Postgres)
