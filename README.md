# SignalFactory

SignalFactory is a signal-ingestion and scoring engine for LISA.

It pulls from heterogeneous sources, normalizes raw observations into a shared signal schema, scores them for trust/newness/relevance, and packages them into lane-ready evidence packets for downstream LISA consumption. It does not make final decisions on its own.

## What This Repo Is Now

SignalFactory is strongest as:

- configurable source ingestion
- normalized signal production
- operator-facing review tooling
- lane-based evidence packaging for LISA

SignalFactory is not positioned as:

- a universal prediction product
- a final decision engine
- a trustworthy single-model answer for unrelated domains

## Quick Start

```bash
cd d:\AAATraderCorner\TradeScout\SignalFactory\signalfactory

# 1. Install dependencies
pip install -r requirements.txt

# 2. Optional: add your FRED key for US macro signals
copy .env.example .env

# 3. Inspect available signal packs
python main.py --list-signal-packs

# 4. Run the dashboard
streamlit run dashboard/app.py

# 5. Or build one signal pack from the CLI
python main.py --signal-pack food_demand_weekly --save

# 6. Run migration tests
python -m unittest discover -s tests -p "test_*.py"
```

## 30-Second Architecture

SignalFactory now centers on the following pipeline:

1. RawEvent: unprocessed source observation with timestamp, payload, and source metadata.
2. NormalizedSignal: one standardized signal extracted from a raw event.
3. ScoredSignal: normalized signal plus truth, novelty, recency, relevance, corroboration, and contradiction scores.
4. LanePacket: grouped scored signals prepared for a LISA lane.

LISA is the downstream consumer. SignalFactory’s job is to prepare the signal packet cleanly and honestly.

## Config Model

The new primary config files are:

- `config/signal_specs.yaml`: defines lane-oriented signal packs
- `config/lanes.yaml`: defines lane scoring, decay, contradiction, and aggregation rules
- `config/sources.yaml`: defines connectors and source settings

Legacy `config/targets.yaml` is still supported during migration, but the repo now prefers `signal_specs.yaml`.

## Built-In Signal Packs

| Key | Label | Lane | Analyzer Output |
|---|---|---|---|
| `gdp_growth` | GDP Growth Signal | `macro` | regression support signal |
| `inflation_forecast` | Inflation Pressure Signal | `macro` | forecast support signal |
| `market_trend` | Market Trend Signal | `market` | directional classifier |
| `election_outcome` | Election Outcome Signal | `community` | political risk classifier |
| `business_viability` | Business Viability Signal | `business` | operational health score |
| `social_unrest_risk` | Social Unrest Signal | `risk` | instability classifier |
| `food_demand_weekly` | Food Demand Signal | `business` | demand forecast support |
| `construction_material_cost_direction` | Construction Material Cost Signal | `infrastructure` | cost direction classifier |
| `restaurant_failure_risk` | Restaurant Failure Risk Signal | `risk` | operational risk classifier |
| `construction_project_delay_risk` | Construction Delay Risk Signal | `infrastructure` | schedule risk classifier |

## Lane Model

Current lane configs include:

- `macro`
- `market`
- `business`
- `community`
- `sentiment`
- `opportunity`
- `risk`
- `infrastructure`

Each lane defines:

- allowed source types
- recency decay behavior
- scoring weights
- contradiction review thresholds
- packet aggregation rules

The same raw observation can score differently by lane. That is intentional.

## Current Migration State

This repo is in a staged rebuild. The modular skeleton is being preserved while the target-centric prediction framing is replaced with signal-centric packaging.

Already in place:

- signal core objects in `src/signals/`
- explicit raw event -> normalized signal -> scored signal -> lane packet engine in `src/pipeline/`
- analyzer naming layer in `src/analyzers/`
- lane config in `config/lanes.yaml`
- signal-pack config in `config/signal_specs.yaml`
- dashboard and CLI terminology shifted toward signal packs and LISA packaging

## Validation Fixtures

Signal pipeline fixtures and behavioral tests are in `tests/test_signal_pipeline.py`.

These tests validate:

- raw event extraction from source tables
- normalization into the shared signal schema
- scoring field ranges and confidence reason generation
- lane packet assembly with traceable signal payloads

Still being migrated:

- broad feature engineering into extractor/normalizer/scorer layers
- model registry into analyzer registry
- prediction-first exports into LISA packet-first exports

## Sources

| Source | Role in SignalFactory | Key required? |
|---|---|---|
| World Bank | macro and structural evidence | No |
| FRED | US macro and rate evidence | Yes |
| CSV / Excel / Parquet | operator-supplied evidence | No |
| SQLite cache | persisted local state | No |
| Web scraper | lightweight sentiment and event sourcing | No |

## Repo Layout

```text
signalfactory/
├── config/
│   ├── lanes.yaml
│   ├── signal_specs.yaml
│   ├── sources.yaml
│   └── targets.yaml              # legacy compatibility during migration
├── dashboard/
│   └── app.py
├── src/
│   ├── configuration.py
│   ├── data/
│   ├── features/                 # legacy feature pipeline, to be migrated
│   ├── models/                   # legacy analyzer implementations
│   ├── outputs/
│   └── signals/
│       ├── schema.py
│       ├── scorer.py
│       ├── packager.py
│       └── adapters.py
├── data/
└── main.py
```

## Operator Workflow

1. Choose a lane-oriented signal pack.
2. Ingest from configured sources or uploaded files.
3. Normalize observations into signals.
4. Score signals for truth, novelty, recency, relevance, corroboration, and contradiction risk.
5. Export lane packets for LISA review.

## Product Discipline

SignalFactory should provide:

- evidence packets
- traceable scoring reasons
- lane-aware prioritization
- explicit uncertainty

SignalFactory should avoid claiming:

- universal predictive confidence
- single-model truth across unrelated domains
- final autonomous decision authority
