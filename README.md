# HDB Compare Flats Agent

An agentic search system for finding comparable HDB (Housing Development Board) resale transactions using natural language queries. The system uses a two-agent architecture with iterative refinement to find an optimal candidate pool of 30-200 comparable transactions.

**🎥 <a href="https://hdb-gradio-app-production.up.railway.app" target="_blank" rel="noopener noreferrer">Demo (over data from Jan 2024 to Jan 2026)</a>**

## Features

- **Natural Language Queries**: Ask questions like "Find 4-room flats in Sengkang, around 95 sqm, mid floor, last 12 months"
- **Intelligent Refinement**: Automatically relaxes or tightens search constraints to find the right number of comparables
- **Hybrid Retrieval**: Combines structured filters with semantic vector search and BM25 lexical matching
- **Deterministic Ranking**: Scores results by area match, lease remaining, storey preference, and recency
- **Visual Analytics**: Price histogram with median and percentile overlays
- **Transparent Agent Trace**: See every decision the agent makes during the search process
- **BrainTrust Integration**: Full observability with trace logging and LLM monitoring

## Screenshots

### Main Interface with Results

The Gradio UI provides a chat-based interface for natural language queries. Results are displayed in a sortable table with price statistics.

![UI with Results](docs/images/UI_with_dresults.jpg)

### Agent Trace Panel

See every decision the agent makes during the iterative search process - which constraints were relaxed or tightened, candidate counts at each step, and retrieval mode used.

![Trace Panel](docs/images/trace_panel.jpg)

### Price Distribution Histogram

Visual analytics showing price distribution with median (green), 25th percentile (yellow), and 75th percentile (yellow) overlays.

![Histogram](docs/images/histogram.jpg)

## Architecture

### Two-Agent Design Pattern

The system uses **two separate LLM agents** coordinated by a Python loop orchestrator:

```mermaid
graph TD
    A[User Query] --> B[Target Agent]
    B -->|Extracts Target| C[Loop Orchestrator]
    C -->|Builds Filters| D[Database Query]
    D -->|Count Results| E{Count in Range?}
    E -->|Yes: 30-200| F[Fetch & Rank]
    E -->|No| G[Planner Agent]
    G -->|Decide Action| H{Action?}
    H -->|Relax| I[Widen Constraints]
    H -->|Tighten| J[Narrow Constraints]
    H -->|Clarify| K[Ask User]
    H -->|Accept| F
    I --> C
    J --> C
    K --> L[User Response]
    L --> A
    F --> M[Return Results]
```

### Component Responsibilities

**1. Target Agent** (`agent/target.py`)
- Extracts structured intent from natural language
- Input: Raw user query string
- Output: `Target` Pydantic model with parsed constraints (town, flat_type, sqm, storey, lease, etc.)
- No tool calls, pure extraction

**2. Planner Agent** (`agent/planner.py`)
- Decides iterative refinement actions
- Input: JSON payload with query, target, filters, count, stats, history, available adjustments
- Output: `PlannerDecision` with action (accept/relax/tighten/clarify) + optional adjustment
- No tool calls, pure decision-making

**3. Loop Orchestrator** (`agent/orchestrator.py`)
- Coordinates agents and tools
- Calls Target Agent once to extract intent
- Iteratively calls Planner Agent to observe results and decide adjustments
- Directly invokes DB tools (count, fetch, stats)
- Applies adjustments and rebuilds filters between iterations
- LLMs never call tools directly - tools are Python functions only

### Iterative Retrieval Loop

Goal: Find 30-200 candidate transactions by relaxing or tightening constraints.

```
User Query → Target Agent → filters → COUNT
                                        ↓
                            ┌───────────┴─────────────┐
                            ↓                         │
                    Planner Agent observes            │
                    (count, stats, history)           │
                            ↓                         │
            ┌───────────────┼───────────────┐         │
            ↓               ↓               ↓         │
        CLARIFY         ACCEPT          RELAX/TIGHTEN │
     (ask user)      (done, fetch)    (adjust, loop)──┘
```

**Relax adjustments** (when count < 30):
- `widen_time_window`: 12→18→24 months
- `widen_sqm_tolerance`: ±5→±8→±12 sqm
- `drop_storey_preference`: remove low/mid/high constraint

**Tighten adjustments** (when count > 200):
- `narrow_time_window`: 24→12→6 months
- `narrow_sqm_tolerance`: ±12→±8→±5→±3 sqm
- `raise_min_lease_years`: 60→70→80→90 years
- `require_street_hint`: enforce street_name filter
- `cap_price_budget`: apply price_budget_max as hard filter

**Max iterations**: 4 relax steps, 4 tighten steps (configurable)

### Hybrid Retrieval System

The system supports three retrieval modes:

**1. Structured mode** (default):
- Pure SQL filters on town, flat_type, month_date, floor_area, storey, lease
- Used when constraints are specific and sufficient

**2. Vector similarity** (semantic search):
- pgvector with HNSW index
- 1024-dim embeddings from Jina AI (embeddings-v3)
- Triggered when: street_hint present or fuzzy location queries

**3. BM25 lexical matching** (keyword search):
- pg_textsearch for keyword-based ranking
- Boosts results matching street names, flat models, landmarks
- Triggered when: partial addresses, specific street hints

**Combined ranking**: Reciprocal Rank Fusion (RRF) merges vector and BM25 scores for optimal relevance.

### Tech Stack

- **Language**: Python 3.13
- **LLM Framework**: PydanticAI
- **Database**: PostgreSQL 17 with pgvector and [pg_textsearch](https://www.tigerdata.com/blog/introducing-pg_textsearch-true-bm25-ranking-hybrid-retrieval-postgres) extensions
- **LLM Provider**: OpenRouter (configurable model)
- **Embeddings**: Jina AI embeddings-v3 (1024 dimensions)
- **UI**: Gradio
- **Observability**: BrainTrust with OpenTelemetry
- **ORM**: SQLAlchemy (engine only, no ORM models)

## Setup

### Prerequisites

- Docker and Docker Compose
- Python 3.13+
- OpenRouter API key ([get one here](https://openrouter.ai/))
- Jina AI API key ([get one here](https://jina.ai/))
- (Optional) BrainTrust API key for tracing ([get one here](https://braintrust.dev/))

### Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd hdb_v0
```

2. **Create Python virtual environment**:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

3. **Configure environment variables**:
```bash
cp .env.example .env
```

**⚠️ MINIMUM REQUIRED CONFIGURATION**: You must update **3 API keys** in `.env` before running the application:

| Variable | Required | Get API Key | Purpose |
|----------|----------|-------------|---------|
| `OPENROUTER_API_KEY` | ✅ **YES** | https://openrouter.ai/keys | LLM inference (Target + Planner agents) |
| `JINA_API_KEY` | ✅ **YES** | https://jina.ai/embeddings/ | Text embeddings generation |
| `BRAINTRUST_API_KEY` | ❌ Optional | https://braintrust.dev/ | LLM tracing and observability |

Look for the `# UPDATE THIS` markers in `.env` to find these required fields.

All other variables have sensible defaults and can be left unchanged for initial setup.

4. **Start PostgreSQL**:
```bash
docker compose up -d
```

5. **Verify Postgres is running**:
```bash
docker compose exec postgres psql -U hdb -d hdb -c "SELECT version();"
```

6. **Initialize database schema**:
```bash
docker compose exec -T postgres psql -U hdb -d hdb < db/schema.sql
```

7. **Run data ingestion** (downloads CSV, parses, generates embeddings):
```bash
PYTHONPATH=. python db/ingest.py
```

This will:
- Download HDB resale data from data.gov.sg
- Filter to last 5 years (configurable via `INGEST_YEARS`)
- Parse derived columns (storey ranges, remaining lease)
- Generate embeddings for each listing
- Load data into PostgreSQL

8. **Verify data loaded**:
```bash
docker compose exec postgres psql -U hdb -d hdb -c "SELECT COUNT(*) FROM hdb_resale;"
docker compose exec postgres psql -U hdb -d hdb -c "SELECT COUNT(*) FILTER (WHERE listing_embedding IS NOT NULL) as with_embeddings FROM hdb_resale;"
```

### Running the Application

**Launch Gradio UI** (recommended):
```bash
python app.py
```

Then open http://localhost:7860 in your browser.

**Test agent via CLI**:
```bash
PYTHONPATH=. python agent/orchestrator.py "Find 4-room in Sengkang, 95 sqm, mid floor, last 12 months"
```

## Example Queries

### Basic Queries

```
Find 4-room flats in Ang Mo Kio
```
Expected: Agent asks for more details (time window, size preferences)

```
Find 4-room in Sengkang, around 95 sqm, last 12 months
```
Expected: Returns 30-200 comparables with area tolerance applied

```
5-room flat in Bishan, 120-130 sqm, high floor, min 80 years lease, last 18 months
```
Expected: May need to relax constraints if too specific

### Advanced Queries (Hybrid Mode)

```
Find 4-room near Compassvale, around 95 sqm, mid floor
```
Expected: Triggers vector similarity search for "Compassvale" street matching

```
Sengkang 4-room, premium apartment-ish, mid floor, long lease
```
Expected: Uses semantic search to find "premium" models like Improved, Premium Apartment

```
3-room in Punggol near LRT, recent sales
```
Expected: BM25 lexical boost for "LRT" keyword matching

### Edge Cases

```
Find a flat, mid floor
```
Expected: Agent asks for missing town and flat_type

```
4-room in Ang Mo Kio, 80-120 sqm, last 5 years
```
Expected: Agent tightens constraints (too broad, 5 years → 24 months)

```
4-room in Sengkang, 120 sqm, high floor, min 95 years lease, last 1 month
```
Expected: Agent relaxes constraints (too specific, widens time/sqm tolerances)

## Database Schema

### Core Table: `hdb_resale`

**Raw columns** (from CSV):
- `month`, `town`, `flat_type`, `block`, `street_name`, `storey_range`, `flat_model`
- `floor_area_sqm`, `lease_commence_date`, `remaining_lease`, `resale_price`

**Derived columns** (computed during ingestion):
- `month_date`: DATE parsed from "YYYY-MM" string
- `storey_min`, `storey_max`, `storey_mid`: Parsed from "01 TO 03" format
- `remaining_lease_months`: Parsed from "61 years 04 months" format
- `listing_text`: Concatenated text for embeddings
- `listing_embedding`: VECTOR(1024) from Jina embeddings v3

**Indexes**:
- B-tree: `town`, `flat_type`, `month_date`, `floor_area_sqm`, `resale_price`
- HNSW: `listing_embedding` for fast ANN vector search
- BM25: `listing_text` using pg_textsearch for lexical search

## Testing

### Run Unit Tests

```bash
pytest tests/
```

Tests cover:
- Parsing logic (storey ranges, remaining lease)
- Scoring functions (area, lease, storey, recency)
- Database query functions

### Run Integration Tests

```bash
# Test Target Agent extraction
python -c "from agent.target import get_target_agent; agent = get_target_agent(); print(agent.run_sync('Find 4-room in Bedok').output)"

# Test database queries
python -c "from db.queries import count_flats; print(count_flats({'town': 'ANG MO KIO', 'flat_type': '4 ROOM'}))"

# Test full orchestrator
PYTHONPATH=. python agent/orchestrator.py "Find 4-room in Sengkang, 95 sqm, last 12 months"
```

### Agent Evaluation

See `test_results.md` for evaluation results on 10 diverse test prompts covering various scenarios.

## Configuration

All configuration is managed via `.env` file (single source of truth).

### Required API Keys (⚠️ Must Update)

These **3 variables** must be configured with your own API keys (marked with `# UPDATE THIS` in `.env.example`):

```env
OPENROUTER_API_KEY=your_key_here          # Get from https://openrouter.ai/keys
JINA_API_KEY=your_key_here                # Get from https://jina.ai/embeddings/
BRAINTRUST_API_KEY=your_key_here          # Optional: Get from https://braintrust.dev/
```

### Variables with Defaults (No Action Needed)

These have sensible defaults and typically don't need changes:

```env
# Database (matches docker-compose.yml)
DATABASE_URL=postgresql+psycopg2://hdb:hdb@localhost:5432/hdb

# LLM Configuration
OPENROUTER_MODEL_NAME=minimax/minimax-m2.1
OPENROUTER_FALLBACK_MODELS=["z-ai/glm-4.7", "openai/gpt-4o-mini"]

# Embedding Configuration
JINA_BASE_URL=https://api.jina.ai/v1
EMBEDDING_MODEL_NAME=jinaai/jina-embeddings-v3
EMBEDDING_DIM=1024

# Hybrid Retrieval Weights
RRF_VECTOR_WEIGHT=0.3
RRF_BM25_WEIGHT=0.7

# Data Ingestion
INGEST_YEARS=2
INGEST_CSV_PATH=data/ResaleflatpricesbasedonregistrationdatefromJan2017onwards.csv

# Observability
BRAINTRUST_PARENT=project_name:hdb-comparable-flats
```

**📖 See `.env.example` for detailed comments explaining each variable.**

## Project Structure

```
hdb_v0/
├── agent/
│   ├── __init__.py
│   ├── llm_client.py      # OpenRouter LLM client setup
│   ├── models.py          # Pydantic models (Target, PlannerDecision, Stats, etc.)
│   ├── orchestrator.py    # Main loop coordinator
│   ├── planner.py         # Planner Agent (relax/tighten decisions)
│   ├── scoring.py         # Deterministic ranking logic
│   ├── target.py          # Target Agent (intent extraction)
│   ├── tools.py           # Database tool functions
│   └── tracing.py         # BrainTrust/OpenTelemetry setup
├── db/
│   ├── ingest.py          # CSV download and embedding generation
│   ├── queries.py         # SQL query functions
│   └── schema.sql         # Database schema (idempotent)
├── docker/
│   └── Dockerfile         # Postgres 17 with extensions
├── tests/
│   ├── test_parsing.py    # Parsing logic tests
│   ├── test_queries.py    # Database query tests
│   └── test_scoring.py    # Ranking logic tests
├── viz/
│   └── plots.py           # Matplotlib histogram generation
├── app.py                 # Gradio UI
├── docker-compose.yml     # Local Postgres service
├── requirements.txt       # Python dependencies
├── settings.py            # Configuration management (pydantic-settings)
├── .env.example           # Environment variable template
└── CLAUDE.md              # Developer guide for Claude Code
```

## Key Implementation Patterns

### Text Normalization

All town/flat_type/street values are normalized to uppercase with collapsed whitespace before DB queries:

```python
# "ang mo kio" → "ANG MO KIO"
# "4-room" → "4 ROOM"
```

### Filter Building

Converts `Target` model to SQL filter dict. Key behaviors:
- `floor_area_target` + `tolerance` → expands to `sqm_min`/`sqm_max` range
- `storey_preference` → maps to storey ranges (low: ≤4, mid: 5-10, high: ≥11)
- `min_remaining_lease_years` → converted to months (years × 12)
- Street hints and price budgets only enforced when adjustment flags are set

### Adjustment Sequences

Relax/tighten adjustments follow predefined sequences:

```python
RELAX_MONTHS_SEQUENCE = (12, 18, 24)
RELAX_SQM_TOLERANCE_SEQUENCE = (5.0, 8.0, 12.0)
TIGHTEN_MONTHS_SEQUENCE = (24, 12, 6)
TIGHTEN_SQM_TOLERANCE_SEQUENCE = (12.0, 8.0, 5.0, 3.0)
TIGHTEN_LEASE_YEARS_SEQUENCE = (60, 70, 80, 90)
```

Adjustments move to the next value in sequence; if already at limit, adjustment is unavailable.

### Clarifying Questions

The system asks clarifying questions in three cases:

1. **Missing required fields**: Town or flat_type not provided
2. **Filter conflicts**: Range filters inverted (e.g., sqm_min > sqm_max)
3. **Planner decides "clarify"**: Results too sparse/broad and no valid adjustments remain

## Performance Considerations

- Candidate pool capped at 500 rows for reranking
- UI results capped at 200 rows (adjustable)
- HNSW index provides fast ANN search (typically <50ms for vector queries)
- B-tree indexes on common filter columns for fast structured queries

## Observability

### BrainTrust Integration

When `BRAINTRUST_API_KEY` is configured, the system automatically:
- Traces all LLM calls with input/output/latency
- Logs agent decisions and adjustments
- Tracks token usage and costs
- Records database query performance

View traces at https://braintrust.dev/

### Agent Trace UI

The Gradio UI includes an expandable trace panel showing:
- Each iteration step
- Filters applied
- Candidate count at each step
- Planner decisions and adjustments
- Retrieval mode (structured/hybrid/vector+BM25)
- Query text used for embeddings

## Data Updates

Ingestion is **manual** - update by re-downloading CSV and running `db/ingest.py`.

To update data:
```bash
PYTHONPATH=. python db/ingest.py
```

**Important**: Changing `EMBEDDING_MODEL_NAME` or `EMBEDDING_DIM` requires re-running full ingestion to regenerate all embeddings.

## Troubleshooting

### Database connection failed
- Verify Postgres is running: `docker compose ps`
- Check logs: `docker compose logs postgres`
- Verify DATABASE_URL in `.env` matches docker-compose.yml

### No results found
- Check if data is loaded: `docker compose exec postgres psql -U hdb -d hdb -c "SELECT COUNT(*) FROM hdb_resale;"`
- Verify filters are not too restrictive (check trace panel)
- Try a broader query first

### Embeddings not working
- Verify JINA_API_KEY is set in `.env`
- Check embeddings are populated: `docker compose exec postgres psql -U hdb -d hdb -c "SELECT COUNT(*) FILTER (WHERE listing_embedding IS NOT NULL) FROM hdb_resale;"`
- Re-run ingestion if needed: `PYTHONPATH=. python db/ingest.py`

### LLM errors
- Verify OPENROUTER_API_KEY is valid
- Check OpenRouter dashboard for quota/rate limits
- Try a different model via OPENROUTER_MODEL_NAME

## Contributing

See `CLAUDE.md` for detailed developer documentation and Claude Code guidance.

## License

[Add license information here]

## Acknowledgments

- HDB resale data from [data.gov.sg](https://data.gov.sg/datasets/d_8b84c4ee58e3cfc0ece0d773c8ca6abc/view)
- Built with [PydanticAI](https://ai.pydantic.dev/)
- LLM inference via [OpenRouter](https://openrouter.ai/)
- Embeddings from [Jina AI](https://jina.ai/)
- Observability by [BrainTrust](https://braintrust.dev/)
