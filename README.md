# SignalFoundry

A general-purpose predictive framework that pulls **economics, cultural, market, and social data** from multiple sources and runs ML/time-series models to predict anything you configure.

---

## Quick start

```bash
cd C:\Users\flavo\predictive-system

# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) Add your free FRED API key for US economic data
copy .env.example .env
# Edit .env and paste your key from https://fred.stlouisfed.org/docs/api/api_key.html

# 3. Run the interactive dashboard
streamlit run dashboard/app.py

# 4. Or use the CLI
python main.py --list-targets
python main.py --target gdp_growth --countries US,CN,DE --start 2000
python main.py --target inflation_forecast --horizon 24 --save
```

---

## Prediction targets (built-in)

| Key | Label | Type |
|-----|-------|------|
| `gdp_growth` | GDP Growth Rate | regression |
| `inflation_forecast` | Inflation Rate | forecast (ARIMA) |
| `market_trend` | Stock Market Direction | classification |
| `election_outcome` | Incumbent Win Probability | classification |
| `business_viability` | Business Viability Score | regression |
| `social_unrest_risk` | Social Unrest Risk Level | classification |
| `food_demand_weekly` | Food Demand Forecast (Weekly Units) | forecast (ARIMA) |
| `construction_material_cost_direction` | Construction Material Cost Direction | classification |
| `restaurant_failure_risk` | Restaurant Failure Risk | classification |
| `construction_project_delay_risk` | Construction Project Delay Risk | classification |

**Change the active target** by editing `config/targets.yaml` → `active_target`.

---

## Add a custom target

Open `config/targets.yaml` and add a new block:

```yaml
targets:
  my_custom_target:
    label: "My Custom Prediction"
    type: regression             # regression | classification | forecast
    output_unit: "score"
    features:
      economic: [gdp_growth, inflation, unemployment]
      social:   [population_growth]
      market:   [stock_index_return]
    countries: [US]
    date_range:
      start: "2000-01-01"
      end:   "today"
    model: gradient_boosting     # gradient_boosting | random_forest | ridge | ensemble
                                 # logistic_regression (classifiers only)
                                 # arima (forecast only)

active_target: my_custom_target
```

---

## Construction + Food Scaffolding (all 4 together)

This repository now includes a starter scaffold for all four requested use cases:

1. `food_demand_weekly`
2. `construction_material_cost_direction`
3. `restaurant_failure_risk`
4. `construction_project_delay_risk`

Starter data templates are in:

- `data/templates/food_demand_weekly_template.csv`
- `data/templates/construction_material_cost_direction_template.csv`
- `data/templates/restaurant_failure_risk_template.csv`
- `data/templates/construction_project_delay_risk_template.csv`

To use them:

1. Copy one or more template files into `data/uploads/`.
2. Fill them with your real data (keep column names unchanged).
3. Run dashboard and pick one of the 4 scaffolded targets.
4. Click **Run Prediction** for one target, or click **Run All 4 Scaffold Targets** to execute all four at once.

---

## Data sources

| Source | What it provides | Key required? |
|--------|-----------------|---------------|
| **World Bank** | GDP, inflation, unemployment, inequality, education, … | No |
| **FRED** | US interest rates, M2, CPI, oil price, VIX, … | Yes (free) |
| **CSV / Excel / Parquet** | Your own data | No |
| **SQLite** | Persisted tables from previous sessions | No |
| **Reddit scraper** | Basic news sentiment signals | No |

---

## Project structure

```
predictive-system/
├── config/
│   ├── targets.yaml        ← define what to predict
│   └── sources.yaml        ← configure data sources
├── src/
│   ├── data/               ← connectors (World Bank, FRED, CSV, cache, scraper)
│   ├── features/           ← feature engineering pipeline
│   ├── models/             ← regressor, classifier, forecaster, registry
│   └── outputs/            ← report builder
├── dashboard/
│   └── app.py              ← Streamlit UI
├── data/
│   ├── uploads/            ← drop your CSV/Excel files here
│   ├── cache/              ← SQLite cache (auto-managed)
│   ├── models/             ← saved model files
│   └── reports/            ← exported JSON/CSV reports
├── main.py                 ← CLI entry point
└── requirements.txt
```

---

## Feature engineering (automatic)

The pipeline applies these transforms to every raw indicator:

- **Lag features** (1, 2, 3 periods back)
- **Rolling mean & std** (3-period and 5-period windows)
- **% change** (period-over-period)
- **Interaction terms** (e.g. GDP × unemployment)
- **One-hot country encoding**
- Forward/backward fill for missing values

---

## Model types

| Config value | Algorithm | Best for |
|---|---|---|
| `gradient_boosting` | Scikit-learn GBM | tabular regression/classification |
| `random_forest` | Random Forest | robust, less tuning needed |
| `ridge` | Ridge regression | small datasets, linear relationships |
| `logistic_regression` | Logistic regression | binary/multiclass classification |
| `ensemble` | Voting ensemble (GBM + RF + Ridge) | best average accuracy |
| `arima` | ARIMA / SARIMAX | univariate time-series forecasting |
