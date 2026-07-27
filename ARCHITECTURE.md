# ALPHA LAB — Architecture & Build Specification

> **Read this file completely before writing a single line of code.**
> This document is the contract. It defines what to build, in what order, to what
> quality bar, and how to prove each piece works. When this document and your
> instinct disagree, this document wins. When this document is silent, write an
> ADR (`docs/decisions/NNNN-*.md`) recording the choice and move on.

---

## 0. How Claude Code should use this document

**Working agreement:**

1. **Do not generate the whole repo in one pass.** Build one phase at a time
   (§14). A phase is done when its *exit criteria* pass, not when the code
   "looks finished."
2. **Every phase ends with a green `just verify`.** No exceptions, no "I'll fix
   the types later."
3. **Maintain `PROGRESS.md`** at the repo root: a checklist mirroring §14 with
   one line per completed phase (what was built, what was deferred, what
   surprised you).
4. **Maintain `docs/decisions/`** as numbered ADRs. Any deviation from this spec
   requires an ADR explaining the trade-off.
5. **Write tests in the same commit as the code they test.** A module without
   its tests is not a module.
6. **Ask before inventing scope.** If a requirement here is ambiguous, state the
   two readings and pick the more conservative one, then note it in the ADR.
7. **Never fabricate results.** If a metric can't be computed, the pipeline
   raises. `NaN` silently propagating into a Sharpe ratio is a critical bug, not
   an inconvenience.

**Kickoff prompt to use in Claude Code:**

```
Read ARCHITECTURE.md end to end. Then:
1. Restate the system in ~15 bullets so I can confirm you understood it.
2. Produce the Phase 0 plan with file-level detail.
3. Implement Phase 0 only. Stop. Run `just verify` and show me the output.
```

---

## 1. What this is, and what it is not

**Alpha Lab** is a research platform for testing whether a short-horizon
predictive signal on financial time series is *real* — meaning it survives
purged validation, transaction costs, and multiple-testing correction.

The product's single job: **calibrate a researcher's trust in a claimed edge.**
Every screen, metric, and chart exists to answer "should I believe this?" — not
"how big is the number?"

### In scope

- Point-in-time data ingestion into an immutable local lake
- Causal feature engineering with an enforced no-lookahead contract
- Labelling: fixed-horizon, triple-barrier, and meta-labelling
- Purged / embargoed cross-validation and walk-forward evaluation
- Gradient-boosted + linear + sequence models behind one interface
- Hyperparameter search that optimises an out-of-fold objective, not an in-fold one
- An event-driven backtester with an explicit, auditable cost model
- Risk-adjusted metrics **including** deflated Sharpe and probability of backtest overfitting
- SHAP-based explanation
- A FastAPI service over an immutable artifact store
- A custom React research terminal (§13)
- Reproducible runs: same config + same data hash ⇒ same metrics, bit-for-bit

### Explicitly out of scope — do not build these

| Not building | Why |
|---|---|
| Streamlit | The UI is a first-class deliverable, not a widget dump. See §13. |
| Live order routing / broker connectivity | Research platform. No execution path exists, deliberately. |
| Multi-tenant auth, RBAC, SSO | Single-user local tool. One optional bearer token, that's it. |
| Kubernetes, Celery, Redis, Kafka | Single-node. Complexity that isn't earned is a negative signal. |
| A trading recommendation of any kind | The system reports evidence. It never says "buy." |
| Alpaca/IBKR paper trading | Same reason as live routing. |

Add a `DISCLAIMER.md` and surface a one-line version in the web app footer:
research software, not investment advice, past backtested performance is not
indicative of anything.

---

## 2. Deliberate deviations from a naïve stack

The original brief listed a stack. Several items are replaced. **Each swap is a
deliberate engineering judgement and each one is a talking point in an
interview** — document them as ADRs 0001–0007.

| Brief said | Build instead | Reason |
|---|---|---|
| `pandas-ta` / TA-Lib | **In-house `alphacore.features.primitives`** | `pandas-ta` is effectively unmaintained and fragile on NumPy 2 / Py3.12. More importantly: an indicator library you wrote and unit-tested against published reference values proves you understand causality and warm-up periods. Third-party indicators are a black box you cannot certify as leak-free. |
| VectorBT / Backtrader | **Own event-driven engine** (`alphacore.backtest`) | Every interesting question in a backtest is "what did you assume about fills and costs?" A vendored engine hides the answer. Own engine = auditable. *Cross-check* against a naive vectorised implementation in tests (§12.5) to catch accounting bugs. |
| TensorFlow / Keras | **PyTorch 2.x**, optional extra `[dl]` | Cleaner install story on 3.12, better tooling, and the sequence models here are small. Core must not import torch. |
| Black + Ruff | **Ruff only** (`ruff format` + `ruff check`) | Ruff formats identically to Black. Two formatters is a config-drift bug waiting to happen. |
| Streamlit | **FastAPI + React/TS** | §13. |
| `pip` + `requirements.txt` | **`uv` + `pyproject.toml` + `uv.lock`** | Reproducibility is a hard requirement here; a lockfile is non-negotiable. Still emit `requirements.txt` via `uv export` for graders who want it. |
| "accuracy leaderboard" | **Out-of-fold economic leaderboard** | Classification accuracy on price direction is close to meaningless. Rank by net-of-cost OOS Sharpe with a DSR haircut, show accuracy as a secondary column. |

---

## 3. Repository layout

```
alpha-lab/
├── ARCHITECTURE.md              ← this file
├── README.md                    ← 60-second pitch, screenshots, quickstart
├── PROGRESS.md                  ← build log (you maintain this)
├── DISCLAIMER.md
├── justfile                     ← task runner (see §16)
├── pyproject.toml               ← uv workspace root
├── uv.lock
├── .python-version              ← 3.12
├── .env.example
├── .gitignore                   ← data/, runs/, .venv/, node_modules/
│
├── packages/
│   ├── alphacore/               ← pure research library. No web, no CLI deps.
│   │   ├── pyproject.toml
│   │   └── src/alphacore/
│   │       ├── __init__.py
│   │       ├── runtime/         ← seeds, logging, timers, run manifest, hashing
│   │       │   ├── determinism.py
│   │       │   ├── logging.py
│   │       │   ├── manifest.py
│   │       │   └── artifacts.py
│   │       ├── config/          ← pydantic settings models + loader + hashing
│   │       │   ├── schema.py
│   │       │   ├── loader.py
│   │       │   └── defaults.yaml
│   │       ├── data/
│   │       │   ├── providers/   ← yahoo.py, binance.py, alphavantage.py, csv.py
│   │       │   ├── protocol.py  ← BarProvider Protocol
│   │       │   ├── lake.py      ← parquet write/read, partitioning
│   │       │   ├── catalog.py   ← duckdb catalog of datasets
│   │       │   ├── calendars.py ← session handling, expected-bar grids
│   │       │   └── quality.py   ← quality report generation
│   │       ├── schemas/         ← pandera dataframe schemas (the type system)
│   │       │   ├── bars.py
│   │       │   ├── features.py
│   │       │   └── labels.py
│   │       ├── features/
│   │       │   ├── primitives.py    ← the indicator maths (pure numpy)
│   │       │   ├── registry.py      ← FeatureSpec registry + causality metadata
│   │       │   ├── transforms.py    ← lags, rolling stats, z-scores, fracdiff
│   │       │   ├── pipeline.py      ← spec list → feature frame
│   │       │   └── selection.py     ← correlation pruning, MDA, clustered importance
│   │       ├── labels/
│   │       │   ├── horizon.py       ← fixed-horizon return / sign / quantile bins
│   │       │   ├── triple_barrier.py
│   │       │   ├── meta.py          ← meta-labelling
│   │       │   └── weights.py       ← uniqueness + time-decay sample weights
│   │       ├── cv/
│   │       │   ├── purged.py        ← PurgedKFold with embargo
│   │       │   ├── walkforward.py   ← anchored + rolling
│   │       │   ├── cpcv.py          ← combinatorial purged CV
│   │       │   └── splits.py        ← Split dataclass, serialisation
│   │       ├── models/
│   │       │   ├── protocol.py      ← AlphaModel Protocol
│   │       │   ├── linear.py        ← ElasticNet / LogisticRegression baseline
│   │       │   ├── trees.py         ← LightGBM, XGBoost, CatBoost adapters
│   │       │   ├── baselines.py     ← Constant, RandomSign, Momentum, Buy&Hold
│   │       │   ├── ensemble.py      ← rank-average / stacked
│   │       │   └── sequence.py      ← [dl] GRU/LSTM/TCN, imported lazily
│   │       ├── tuning/
│   │       │   ├── objective.py     ← OOF economic objective
│   │       │   └── search.py        ← optuna study wiring, pruners
│   │       ├── eval/
│   │       │   ├── classification.py
│   │       │   ├── calibration.py
│   │       │   ├── financial.py     ← sharpe/sortino/calmar/VaR/CVaR/...
│   │       │   ├── deflated.py      ← PSR, DSR, minTRL
│   │       │   └── pbo.py           ← CSCV probability of backtest overfitting
│   │       ├── signal/
│   │       │   ├── thresholds.py    ← prob → discrete signal, hysteresis
│   │       │   └── sizing.py        ← fixed-fraction, vol-target, Kelly (capped)
│   │       ├── risk/
│   │       │   ├── stops.py         ← stop-loss / take-profit / trailing
│   │       │   └── limits.py        ← daily loss, exposure, concurrency, DD kill
│   │       ├── backtest/
│   │       │   ├── engine.py        ← the event loop
│   │       │   ├── costs.py         ← commission, spread, impact, borrow
│   │       │   ├── portfolio.py     ← cash, positions, mark-to-market
│   │       │   ├── ledger.py        ← immutable event log → trades/equity
│   │       │   └── reference.py     ← naive vectorised engine, tests only
│   │       ├── explain/
│   │       │   └── shap_runner.py
│   │       ├── report/
│   │       │   ├── tearsheet.py     ← builds the canonical JSON artifact
│   │       │   ├── exporters.py     ← CSV/parquet exports
│   │       │   └── research_md.py   ← generated research write-up
│   │       └── pipelines/
│   │           ├── ingest.py
│   │           ├── build_features.py
│   │           ├── train.py
│   │           ├── backtest.py
│   │           └── full_run.py      ← the golden path (§15)
│   │
│   └── alphaservice/            ← FastAPI app + Typer CLI
│       ├── pyproject.toml
│       └── src/alphaservice/
│           ├── api/
│           │   ├── app.py
│           │   ├── deps.py
│           │   ├── routers/  ← runs.py, datasets.py, jobs.py, system.py
│           │   └── models.py ← pydantic response models (source of TS types)
│           ├── jobs/
│           │   ├── runner.py     ← ProcessPoolExecutor + SQLite state
│           │   ├── store.py
│           │   └── events.py     ← SSE stream of job logs
│           └── cli.py            ← `alpha ...` Typer entrypoint
│
├── web/                         ← Vite + React 19 + TS. See §13.
│   ├── package.json
│   ├── src/
│   │   ├── routes/
│   │   ├── components/
│   │   ├── charts/
│   │   ├── api/            ← generated client + TanStack Query hooks
│   │   ├── styles/tokens.css
│   │   └── lib/
│   └── e2e/                ← playwright
│
├── configs/
│   ├── default.yaml
│   ├── experiments/
│   │   ├── spy_daily_tb.yaml
│   │   ├── btc_1h_meta.yaml
│   │   └── smoke.yaml       ← tiny, runs in <60s, used by CI
│   └── schema.json          ← generated from pydantic, for editor autocomplete
│
├── data/                    ← gitignored lake
│   ├── raw/{provider}/{symbol}/{interval}/*.parquet
│   ├── curated/...
│   └── catalog.duckdb
├── runs/                    ← gitignored artifact store, one dir per run
├── samples/                 ← small committed parquet fixtures (<5 MB total)
├── docs/
│   ├── decisions/           ← ADRs
│   ├── diagrams/            ← mermaid sources + rendered svg
│   ├── data-dictionary.md   ← every feature: formula, window, causality
│   └── methodology.md       ← the stats: purging, DSR, PBO, cost model
├── tests/
│   ├── unit/
│   ├── property/            ← hypothesis
│   ├── leakage/             ← §12.3, the most important suite
│   ├── golden/              ← frozen expected outputs
│   ├── integration/
│   └── api/
├── docker/
│   ├── Dockerfile           ← multi-stage: web build → python runtime
│   └── compose.yaml
└── .github/workflows/ci.yml
```

**Dependency direction is strictly one-way:**

```
web  →  alphaservice  →  alphacore  →  (numpy, pandas, sklearn, ...)
```

`alphacore` must never import `fastapi`, `typer`, or anything web-shaped. Enforce
with an import-linter contract in CI.

---

## 4. Core invariants

These are the load-bearing rules. Violating any one silently invalidates every
number the system produces. Each has a test in `tests/leakage/`.

**I1 — Causality.** A feature value at timestamp `t` may depend only on data
with timestamp `≤ t`. Every `FeatureSpec` declares `lookback: int` and is
verified by the future-poison test (§12.3).

**I2 — Execution lag.** A signal derived from bar `t` may only be executed at
bar `t + execution_lag` (default 1), at that bar's **open**, never its close.
Same-bar execution is banned at the engine level — the engine physically cannot
see bar `t`'s close when placing the order for bar `t`.

**I3 — Fit only on train.** Scalers, imputers, encoders, feature selectors, and
any target-dependent statistic are fitted inside the fold, on train rows only.
Everything goes through an sklearn `Pipeline`; no pre-scaled global matrices.

**I4 — Purge and embargo.** Labels span `[t, t+h]`. Any training sample whose
label span overlaps the test window is *purged*. An additional `embargo_pct` of
bars after the test window is dropped from training. Both are mandatory, not
optional flags.

**I5 — No target leakage by construction.** The feature registry rejects any
spec whose computation graph touches the label column. Assert at pipeline build.

**I6 — Point-in-time universe.** Symbol lists are dated snapshots. Never build a
universe from "current members of index X."

**I7 — Consistent adjustment.** Choose split/dividend adjustment once per
dataset, record it in the manifest, and never mix adjusted and raw series in one
feature. Prefer *unadjusted* OHLC + a separate adjustment factor column so that
the backtester trades real historical prices.

**I8 — Determinism.** Given `(config_hash, data_hash, seed)`, every artifact is
byte-identical across runs. All RNGs seeded; `PYTHONHASHSEED=0`; no set
iteration in output paths; sort before writing; `n_jobs` fixed for tuning
reproducibility.

**I9 — Immutable runs.** `runs/{run_id}/` is written once and never mutated. Any
change means a new run. The API only reads.

**I10 — Costs are never zero by default.** A backtest config with zero
commission, zero spread, and zero slippage must emit a loud warning into the
tearsheet and be flagged in the UI as `UNCOSTED`.

---

## 5. Configuration

Single source of truth: **pydantic v2 models** in `alphacore/config/schema.py`.
YAML is deserialised into them; unknown keys are a hard error
(`model_config = ConfigDict(extra="forbid")`). `configs/schema.json` is generated
from the models so editors autocomplete the YAML.

**Config resolution order:** `defaults.yaml` → experiment YAML → env vars
(`ALPHA__BACKTEST__COMMISSION_BPS=1.5`) → CLI flags. The fully resolved config is
serialised into every run manifest, and `config_hash = blake2b(canonical_json)`.

```yaml
# configs/experiments/spy_daily_tb.yaml
meta:
  name: spy-daily-triple-barrier
  description: Daily SPY, triple-barrier labels, LGBM vs baselines.
  seed: 20260727

data:
  provider: yahoo            # yahoo | binance | alphavantage | csv
  symbols: [SPY]
  benchmark: SPY
  interval: 1d               # 1m 5m 15m 1h 4h 1d
  start: "2005-01-01"
  end: "2025-12-31"
  adjustment: unadjusted_plus_factor
  quality:
    max_missing_bar_pct: 2.0
    max_zero_volume_pct: 1.0
    fail_on_ohlc_violation: true

features:
  specs:                     # names resolve against the registry
    - {name: sma, params: {window: [10, 20, 50, 100]}}
    - {name: ema, params: {window: [12, 26]}}
    - {name: hma, params: {window: [20]}}
    - {name: rsi, params: {window: [14]}}
    - {name: macd, params: {fast: 12, slow: 26, signal: 9}}
    - {name: roc, params: {window: [5, 20]}}
    - {name: ppo, params: {fast: 12, slow: 26}}
    - {name: atr, params: {window: [14]}}
    - {name: realised_vol, params: {window: [10, 20, 60]}}
    - {name: bollinger, params: {window: 20, k: 2.0}}
    - {name: obv, params: {}}
    - {name: vwap_dist, params: {window: [20]}}
    - {name: cmf, params: {window: [20]}}
    - {name: volume_ratio, params: {window: [20]}}
    - {name: log_return, params: {window: [1, 5, 21]}}
    - {name: rolling_stat, params: {stat: [mean, std, median, min, max], window: [5, 20, 50]}}
    - {name: zscore, params: {window: [20, 100]}}
    - {name: fracdiff, params: {d: 0.4, thresh: 1e-4}}
  lags: [1, 2, 5, 10, 20]
  prune:
    method: clustered_correlation   # none | correlation | clustered_correlation
    threshold: 0.95
  winsorise_quantile: 0.001

labels:
  method: triple_barrier     # horizon_sign | horizon_return | horizon_quantile | triple_barrier
  horizon_bars: 10
  triple_barrier:
    profit_mult: 2.0         # × ATR
    loss_mult: 1.0
    vertical_bars: 10
    atr_window: 14
    min_return: 0.0
  meta_labelling: false
  sample_weights:
    uniqueness: true
    time_decay: 0.75         # weight of oldest observation; 1.0 = off

cv:
  scheme: walkforward        # purged_kfold | walkforward | cpcv
  walkforward:
    train_bars: 1260
    test_bars: 252
    anchored: false
    step_bars: 252
  embargo_pct: 0.01
  min_train_bars: 500

models:
  candidates: [baseline_buyhold, baseline_momentum, logistic, lightgbm, xgboost, catboost]
  calibrate: isotonic        # none | sigmoid | isotonic (fitted on inner split)
  tuning:
    enabled: true
    n_trials: 60
    timeout_s: 1800
    objective: oos_net_sharpe   # oos_net_sharpe | oos_logloss | oos_information_coefficient
    pruner: median

signal:
  long_threshold: 0.56
  short_threshold: 0.44
  allow_short: true
  hysteresis: 0.02           # must exceed threshold ± this to flip
  sizing:
    method: vol_target       # fixed_fraction | vol_target | kelly_capped
    target_annual_vol: 0.10
    max_leverage: 1.0
    kelly_cap: 0.25

risk:
  stop_loss_atr: 2.0
  take_profit_atr: 4.0
  trailing_stop_atr: null
  max_drawdown_kill: 0.25
  max_daily_loss: 0.03
  max_concurrent_positions: 5

backtest:
  initial_capital: 1_000_000
  execution_lag_bars: 1
  fill_price: next_open      # next_open | next_vwap
  costs:
    commission_bps: 0.5
    min_commission: 1.0
    half_spread_bps: 1.0
    impact_model: sqrt       # none | linear | sqrt
    impact_coefficient: 0.1
    max_participation: 0.05
    borrow_bps_annual: 50.0
    financing_bps_annual: 300.0
  cost_sensitivity_bps: [0, 1, 2, 5, 10, 20]

reporting:
  bootstrap_samples: 1000
  n_trials_for_dsr: auto     # auto = count of configs actually evaluated
  export: [tearsheet_json, metrics_csv, trades_csv, predictions_parquet, feature_importance_csv, research_md]
```

---

## 6. Data layer

### 6.1 Provider protocol

```python
class BarProvider(Protocol):
    name: str

    def fetch(
        self,
        symbol: str,
        interval: Interval,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame: ...  # must satisfy schemas.bars.RawBars
    def max_lookback(self, interval: Interval) -> timedelta | None: ...
```

Implement `yahoo` (yfinance), `binance` (public REST, no key, paginated 1000-bar
windows), `alphavantage` (key from env, rate-limited with backoff), `csv` (local
files, used for tests and offline demos). Every provider:

- retries with exponential backoff + jitter on 429/5xx, max 5 attempts
- caches raw responses to `data/raw/` before any transformation
- is skipped in CI (network tests marked `@pytest.mark.network`, off by default)

### 6.2 Canonical bar schema (`schemas/bars.py`, pandera)

| column | type | constraint |
|---|---|---|
| `ts` | `datetime64[ns, UTC]` | index, monotonic increasing, unique |
| `open` `high` `low` `close` | `float64` | `> 0`, finite |
| `volume` | `float64` | `>= 0` |
| `adj_factor` | `float64` | `> 0`, default 1.0 |
| `symbol` | `category` | non-null |

Validation checks beyond dtypes:
- `low <= min(open, close) <= max(open, close) <= high`
- no timestamp gaps beyond the expected session grid (calendar-aware for
  equities via `exchange_calendars`; 24/7 grid for crypto)
- monotonic timestamps, no duplicates (keep last on collision, log the drop)

### 6.3 Lake and catalog

- Parquet, ZSTD, partitioned `provider/symbol/interval/year=YYYY/`.
- `catalog.duckdb` holds one row per dataset: `dataset_id`, provider, symbol,
  interval, span, row count, `content_hash` (blake2b over sorted parquet bytes),
  ingest timestamp, provider version, quality summary.
- `content_hash` is what goes in the run manifest. This is how reproducibility is
  proven: a run pinned to a hash either finds that exact data or fails loudly.

### 6.4 Quality report

`quality.py` produces `DataQualityReport` (pydantic → JSON + a rendered table):
row count, span, missing-bar count and pct vs expected grid, duplicate count,
zero-volume bars, OHLC violations, extreme-return outliers (|z| > 10) with
timestamps, longest gap, first/last valid bar, per-year row counts. Thresholds
from config decide `PASS | WARN | FAIL`. A `FAIL` aborts the pipeline unless
`--allow-dirty-data` is passed, which stamps the run manifest with
`data_quality: FAIL` and marks every downstream artifact as untrusted.

---

## 7. Features

### 7.1 Registry

```python
@dataclass(frozen=True)
class FeatureSpec:
    name: str
    fn: Callable[[pd.DataFrame, dict], pd.DataFrame]
    params: dict
    lookback: int  # bars of warm-up required
    inputs: frozenset[str]  # source columns; must not contain label cols
    family: Literal[
        "trend", "momentum", "volatility", "volume", "returns", "statistical", "microstructure"
    ]
    description: str  # goes straight into docs/data-dictionary.md
```

`registry.register(...)` is a decorator. `pipeline.build()` resolves the config's
spec list, expands list-valued params into the cartesian product, computes each
feature, applies lags, prunes correlated columns, and returns a frame plus a
`FeatureManifest` (every column name → spec + params + lookback).

The first `max(lookback)` bars are dropped, not filled. Never backfill a feature.

### 7.2 Primitive implementations (write these yourself, test each)

**Trend:** SMA, EMA (with correct `adjust=False` recursion), WMA, HMA
(`WMA(2*WMA(n/2) - WMA(n), sqrt(n))`).
**Momentum:** RSI (Wilder smoothing), MACD line/signal/histogram, ROC, raw
momentum, PPO.
**Volatility:** ATR (Wilder), true range, realised vol (annualised from
log-returns, annualisation factor derived from `interval`, §11.1), Bollinger
upper/lower/%B/bandwidth, rolling std, Parkinson and Garman–Klass estimators.
**Volume:** OBV, rolling VWAP and distance-to-VWAP, Chaikin Money Flow, volume
ratio vs rolling mean, dollar volume.
**Returns:** log return over `n` bars, simple return, forward-looking versions
are **forbidden here** — they live in `labels/`.
**Statistical:** rolling mean/median/std/min/max/skew/kurt, rolling z-score,
percentile rank in window, fractional differentiation (fixed-width window, weight
cutoff `thresh`) — this last one matters: it makes price stationary while
retaining memory, and it's the kind of detail that separates this from a tutorial.

Every primitive: pure function, numpy-vectorised, `float64`, no in-place mutation
of the input, warm-up region is `NaN` (never 0), and a docstring containing the
formula.

**Reference tests:** for RSI, MACD, ATR, and Bollinger, commit a small CSV of
hand-verified expected values (from the original published definitions) in
`tests/golden/indicators/` and assert to `1e-9`.

### 7.3 Selection

- Winsorise at configured quantiles (fit on train only — so this lives in the
  sklearn pipeline, not the feature builder).
- Correlation pruning: build the correlation matrix on **train rows of fold 0
  only**, hierarchically cluster with distance `sqrt(0.5*(1-corr))`, keep one
  representative per cluster (highest univariate IC). Record the drop list.
- Report Mean Decrease Accuracy on out-of-fold data with the clustering applied
  (clustered MDA), because raw feature importance on correlated features is
  misleading.

---

## 8. Labels and sample weights

### 8.1 Methods

**`horizon_return`** — `r_{t→t+h} = log(close_{t+h}/close_t)`. Regression target.
**`horizon_sign`** — `sign(r_{t→t+h})` with a dead zone `|r| < min_return → drop`.
Binary.
**`horizon_quantile`** — 5-class bucketing of `r_{t→t+h}` by *train-fold*
quantiles (strong sell / sell / hold / buy / strong buy). Quantile edges must be
fitted per-fold, never globally — this is a classic leak.
**`triple_barrier`** — for each `t`, set upper barrier at
`close_t * (1 + profit_mult * σ_t)`, lower at `close_t * (1 - loss_mult * σ_t)`
where `σ_t` is ATR-based, plus a vertical barrier at `t + vertical_bars`. Label
by which barrier is touched first, evaluated on intrabar high/low. Emit
`label`, `t_touch` (**required** — this defines the purge span), `realised_return`.
**`meta_labelling`** — a primary rule (e.g. momentum crossover) produces side;
the ML model predicts only *whether to take the trade* (binary). Improves
precision and is a strong portfolio talking point. Requires the primary model's
signal as an input column.

### 8.2 Sample weights

Overlapping labels break the i.i.d. assumption every estimator makes. Implement:
- **Uniqueness weight:** for each label spanning `[t, t_touch]`, average
  concurrency `c_u` over its span; `w_i = mean(1/c_u)`. Normalise to mean 1.
- **Time decay:** linear decay in cumulative uniqueness, from `decay` (oldest) to
  1.0 (newest).
- Final weight is the product, passed to every estimator's `sample_weight`.

Every label frame carries `(t_start, t_end, label, weight)`. `t_end` is what the
CV purger consumes — nothing else is acceptable.

---

## 9. Cross-validation

```python
@dataclass(frozen=True)
class Split:
    fold: int
    train_idx: np.ndarray
    test_idx: np.ndarray
    train_span: tuple[Timestamp, Timestamp]
    test_span: tuple[Timestamp, Timestamp]
    purged: int  # count of samples removed by purge
    embargoed: int  # count removed by embargo
```

**`PurgedKFold`** — contiguous test blocks; drop train samples whose
`[t_start, t_end]` overlaps the test span; drop a further `embargo_pct * n` bars
after the test span.

**`WalkForward`** — anchored (expanding) or rolling (fixed `train_bars`), step
`step_bars`. This is the primary scheme and the one the UI visualises. Yields
folds in chronological order.

**`CPCV`** — `N` groups, `k` held out per combination, all `C(N,k)` combinations,
each with purge + embargo. Produces multiple backtest paths → the distribution of
Sharpe needed for PBO (§11.3).

Every split object is serialised into the run artifacts so the UI can draw the
exact bands, purge gaps, and embargo gaps that were used. **The purge/embargo
gaps must be visible in the UI** — that's the signature element (§13.2).

Random `train_test_split` and plain `KFold` are banned. Add a CI grep test that
fails if `KFold(` or `train_test_split` appears outside `tests/`.

---

## 10. Models

### 10.1 Interface

```python
class AlphaModel(Protocol):
    name: str
    task: Literal["binary", "multiclass", "regression"]

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        *,
        sample_weight: np.ndarray | None = None,
        eval_set: tuple | None = None,
    ) -> Self: ...
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray: ...  # classification
    def predict(self, X: pd.DataFrame) -> np.ndarray: ...
    def feature_importance(self) -> pd.Series | None: ...
    def save(self, path: Path) -> None: ...
    @classmethod
    def load(cls, path: Path) -> Self: ...
```

Adapters: `logistic`/`elasticnet` (always include — a linear baseline that beats
your GBM is important information), `lightgbm`, `xgboost`, `catboost`.

**Baselines are mandatory and go on the leaderboard:** buy-and-hold,
random-sign (seeded), always-long, simple momentum rule, and — the honest one —
the *same model trained on shuffled labels*. If the shuffled-label model's OOS
Sharpe distribution overlaps the real model's, say so prominently.

### 10.2 Calibration

Raw GBM probabilities are not probabilities. Fit isotonic or Platt calibration on
an **inner** split of the training fold (never the test fold), and report a
reliability diagram + Brier score. Thresholds in §config only make sense on
calibrated outputs.

### 10.3 Sequence models (optional extra `[dl]`)

Windowed tensors `(n_samples, seq_len, n_features)` built **inside** the fold,
with the same purge logic applied at the window level (a window that straddles
the boundary is dropped). Variants: single LSTM, stacked LSTM, bidirectional
(only valid if you're careful — bidirectional over a *past* window is fine,
bidirectional over a window that includes the future is a leak; document this),
GRU, and a small TCN. Dropout, early stopping on the inner validation split,
`ReduceLROnPlateau`, checkpointing, deterministic seeding, MPS/CUDA if available.
Torch is imported lazily inside `sequence.py`; `import alphacore` must work
without it.

### 10.4 Tuning

Optuna, TPE sampler, median pruner. **The objective is out-of-fold and
economic**: run the inner walk-forward, generate signals, run the backtest with
costs, return net Sharpe (or IC). Optimising in-fold log-loss is the single most
common way tutorials produce fantasy results.

Record `n_trials_total` across the entire study — that's the `N` for the deflated
Sharpe haircut (§11.2). Trial count must flow into the tearsheet automatically;
manually setting it low is self-deception.

---

## 11. Evaluation

### 11.1 Annualisation

Derive `periods_per_year` from the interval and the calendar:
`1d → 252` (equities) / `365` (crypto); `1h → 252*6.5` / `365*24`; `1m → 252*390`
/ `365*1440`. Never hardcode 252. Store it in the tearsheet — a Sharpe with an
undocumented annualisation factor is unfalsifiable.

### 11.2 Metric set

Return-based: annualised return (geometric), annualised volatility, Sharpe,
Sortino (downside deviation vs MAR=0), Calmar, Information Ratio vs benchmark,
max drawdown + duration + recovery, ulcer index, VaR and CVaR at 95/99
(historical and Cornish–Fisher), skew, kurtosis, best/worst period, monthly and
annual return tables, rolling 63/252-bar Sharpe.

Trade-based: trade count, turnover, win rate, loss rate, profit factor, average
win, average loss, payoff ratio, expectancy, largest winner, largest loser,
average holding period, max consecutive losses, exposure pct.

Prediction-based: accuracy, balanced accuracy, precision/recall/F1, ROC-AUC,
PR-AUC, log-loss, Brier, and — more useful than all of them — **information
coefficient** (Spearman of prediction vs forward return) with its t-stat, and IC
decay across horizons.

Honesty metrics (**these are what makes the project credible**):
- **PSR** — probability that true Sharpe exceeds a benchmark, adjusting for skew
  and kurtosis:
  `PSR(SR*) = Φ( (SR − SR*)·√(n−1) / √(1 − γ₃·SR + ((γ₄−1)/4)·SR²) )`
- **DSR** — PSR evaluated at the expected maximum Sharpe from `N` trials:
  `SR* = √Var[SR_n] · [ (1−γ)·Φ⁻¹(1 − 1/N) + γ·Φ⁻¹(1 − 1/(N·e)) ]`, γ = 0.5772
- **Minimum track record length** — how many bars you'd need for the Sharpe to be
  significant at 95%
- **PBO** via CSCV (§11.3)
- **Break-even cost**: the round-trip cost in bps at which net Sharpe hits zero.
  Report it next to the headline Sharpe, always.
- **Bootstrap CI** on Sharpe (stationary block bootstrap, block length via
  Politis–White) — report the interval, not just the point estimate.

### 11.3 PBO (CSCV)

Split the OOS return series into `S` (default 16) contiguous chunks. For each of
the `C(S, S/2)` splits: pick the config with best in-sample Sharpe, find its rank
among all configs out-of-sample, map rank to logit. PBO = fraction of splits
where the IS-best config lands below the OOS median. `PBO > 0.5` means your
selection process is worse than random — the UI must say so in plain language.

### 11.4 The tearsheet artifact

One canonical JSON per run — `runs/{run_id}/tearsheet.json` — is the *only*
thing the API and UI read for results. Schema (abbreviated; full pydantic model
in `report/tearsheet.py`):

```jsonc
{
  "schema_version": "1.0",
  "run_id": "2026-07-27T14-22-01Z_spy-daily-tb_a3f91c",
  "created_at": "...",
  "manifest": {
    "config_hash": "...", "data_hash": "...", "code_version": "git sha",
    "seed": 20260727, "python": "3.12.4", "packages": {"lightgbm": "4.5.0"},
    "duration_s": 412.7, "host": "..."
  },
  "dataset": { "symbols": ["SPY"], "interval": "1d", "span": ["2005-01-03","2025-12-31"],
               "n_bars": 5283, "quality": "PASS", "quality_report_ref": "quality.json" },
  "features": { "n_built": 214, "n_after_prune": 96, "manifest_ref": "features.json" },
  "labels": { "method": "triple_barrier", "class_balance": {"1": 0.41, "-1": 0.38, "0": 0.21},
              "mean_uniqueness": 0.31 },
  "cv": { "scheme": "walkforward", "folds": [ {"fold":0, "train_span":[...], "test_span":[...],
          "purged": 9, "embargoed": 52, "metrics": {...}} ] },
  "leaderboard": [
    { "model": "lightgbm", "oos_net_sharpe": 0.71, "sharpe_ci": [0.12, 1.28],
      "dsr": 0.34, "psr": 0.81, "break_even_bps": 6.4, "max_drawdown": -0.19,
      "ic": 0.031, "ic_tstat": 2.1, "auc": 0.534, "brier": 0.248, "n_trials": 60,
      "rank": 1, "flags": ["LOW_DSR"] }
  ],
  "selected_model": "lightgbm",
  "backtest": { "metrics": {...}, "cost_sensitivity": [{"bps":0,"sharpe":1.02}, ...],
                "series_refs": {"equity":"equity.parquet","drawdown":"...","trades":"trades.parquet"} },
  "overfitting": { "pbo": 0.41, "n_configs_evaluated": 312, "shuffled_label_sharpe_p95": 0.55 },
  "explain": { "shap_summary_ref": "shap_summary.parquet", "top_features": [...] },
  "warnings": ["UNCOSTED", "LOW_DSR", "SHORT_SAMPLE"],
  "disclaimer": "Research output. Not investment advice."
}
```

Large series live beside it as parquet and are streamed/downsampled by the API,
never inlined.

---

## 12. Backtesting

### 12.1 Engine

Event-driven loop over bars. Per bar `t`:

1. **Mark to market** using bar `t` open (positions carried from `t-1`).
2. **Apply risk exits** — stops/targets evaluated against `t`'s high/low with a
   conservative tie-break: if both the stop and target are touched in the same
   bar, assume the **stop** fills first. Never assume the favourable one.
3. **Execute pending orders** created at `t - execution_lag` at `t.open` (plus
   costs).
4. **Read signal for `t`** using only information available at `t.close`.
5. **Size the position** (vol target / fixed fraction / capped Kelly).
6. **Apply limits** — daily loss, exposure, concurrency, drawdown kill switch.
7. **Queue orders for `t + execution_lag`.**
8. **Append events to the ledger.**

The engine's state object never holds a reference to future bars. Enforce by
passing a bar-by-bar iterator, not the whole frame.

### 12.2 Cost model (`costs.py`)

```
cost_per_share = commission_bps/1e4 * price
               + half_spread_bps/1e4 * price
               + impact(price, qty, adv)

impact_sqrt   = impact_coefficient * price * sigma_daily * sqrt(qty / adv)
impact_linear = impact_coefficient * price * (qty / adv)
```

Plus `max(min_commission, ...)` per trade, borrow cost accrued daily on short
notional, and financing on leveraged long notional. Orders exceeding
`max_participation * bar_volume` are **partially filled** and the remainder is
carried or cancelled per config — silently filling 100% of a bar's volume at the
open is a fantasy and the engine must not permit it.

Emit `cost_sensitivity`: rerun the ledger's fills at each bps level in
`cost_sensitivity_bps` and record the resulting Sharpe curve. The UI plots this;
a strategy whose Sharpe collapses between 0 and 2 bps is not a strategy.

### 12.3 Leakage test suite (`tests/leakage/`) — highest priority tests

1. **`test_future_poison`** — build features on a series; then replace every row
   after index `k` with `NaN`/garbage; rebuild; assert rows `0..k` are identical
   to `1e-12`. Run over the entire feature registry, parametrised.
2. **`test_label_span_purged`** — for every split from every CV scheme, assert no
   train sample's `[t_start, t_end]` intersects the test span, and that the
   embargo gap is present.
3. **`test_no_same_bar_execution`** — synthetic series where the close of bar `t`
   is a perfect predictor; assert the backtester's Sharpe is ~0. If it isn't, the
   engine can see the future.
4. **`test_pipeline_fit_on_train_only`** — spy on the scaler's `fit`; assert it's
   never called with test indices.
5. **`test_shuffled_labels_have_no_edge`** — full pipeline on shuffled labels;
   assert OOS Sharpe CI contains zero. This is the integration-level canary.
6. **`test_quantile_edges_per_fold`** — assert label bin edges differ across folds
   (i.e. they were fitted per-fold, not globally).
7. **`test_perfect_foresight_sanity`** — a model given the *actual* forward return
   as a feature must produce an absurdly high Sharpe. If it doesn't, the wiring
   is broken. (The inverse canary — proves the test in #3 is meaningful.)

### 12.4 Determinism test

Run `configs/experiments/smoke.yaml` twice in the same session and in a fresh
subprocess; assert `tearsheet.json` is byte-identical after stripping
`created_at`, `duration_s`, and `host`.

### 12.5 Engine cross-check

`backtest/reference.py` implements a deliberately dumb vectorised backtester:
positions shifted by `execution_lag`, returns multiplied, flat costs. On a
config with `impact_model: none`, `max_participation: 1.0`, and no stops, the
event-driven engine's equity curve must match the reference to `1e-8`. Any
divergence is an accounting bug in the engine.

---

## 13. The web application

This replaces Streamlit and is a **first-class deliverable**. Follow the
`frontend-design` skill: plan the token system, critique the plan against
defaults, then build.

### 13.1 Brief

- **Subject:** a quantitative researcher's evidence file for one hypothesis.
- **Audience:** the researcher, and a sceptical reviewer reading over their
  shoulder.
- **Single job:** make it fast to decide whether an edge is real, and impossible
  to accidentally overlook the reason it isn't.
- **Metaphor:** a lab notebook and its margin, not a trading terminal.

**Explicitly avoid** the near-black-plus-acid-green trading terminal — it is both
the AI-generated default and, more importantly, the *wrong* mood. Terminals are
built for speed of action. This tool is built for slowness of judgement. The
visual language should feel closer to a printed statistical report or an
instrument calibration sheet than to a Bloomberg screen.

### 13.2 Design tokens (implement in `web/src/styles/tokens.css`)

```
Palette — "plotter on cold stock"
  --paper       #EFF0EA   page ground, a cold oat that is not cream
  --paper-sunk  #E4E6DE   recessed panels, table stripes
  --rule        #C8CBC0   hairlines, grid
  --ink         #191B18   primary text
  --ink-muted   #6B7069   secondary text, captions
  --prussian    #1E3A5F   in-sample / train bands / primary structure
  --plum        #6C2B54   out-of-sample / test bands / the model's own line
  --sulphur     #C29A1F   purge + embargo hatching, warnings
  --oxblood     #7A2418   losses, drawdown fill, failures
  --sage        #3F6B57   gains (never a saturated green)
  (dark mode: invert to --ink #101210 ground with the same four accents
   lifted +12% lightness — accents keep their identity, ground flips)

Type
  Display : "Bricolage Grotesque" variable — headings only, wdth axis used
            to set section titles slightly condensed. Restraint: max 3 uses/page.
  Body    : "Public Sans" 400/500 — prose, labels, controls.
  Data    : "IBM Plex Mono" 400/500 — every number, without exception.
            font-variant-numeric: tabular-nums on all numeric cells.
            Decimal alignment in tables via a shared decimal column width.

Scale (1.25 ratio, 16px base): 12 / 14 / 16 / 20 / 25 / 31 / 39
Spacing: 4px base, steps 4 8 12 16 24 32 48 64
Radius: 2px only. Shadows: none — use hairline rules and paper-sunk fills.
```

**Signature element — the Fold Strip.** A full-width horizontal timeline of the
walk-forward split: prussian bands for train, plum for test, sulphur diagonal
hatching for purge and embargo gaps, with the price series drawn as a hairline
behind it. It is simultaneously (a) the hero of the run page, (b) the global
fold selector — click a band to filter every chart below to that fold, and (c)
the honest disclosure of exactly how the model was validated. Nothing else on
the page is allowed to be this loud.

**Second signature — the Provenance Stamp.** Every chart and table carries a
one-line monospace footer at 12px in `--ink-muted`:
`run a3f91c · data 7e2b · costs 1.5bps+1.0sp · net of costs · lag 1 bar`.
It looks like a plate stamp on a lab photograph. It means no chart can be
screenshotted out of context and misread. This is the detail a reviewer will
remember.

**Third device — the margin.** A persistent right gutter (280px, collapses below
1280px) holding auto-generated margin notes tied to what's on screen:
`PBO 0.62 — the selection process performs worse than chance` next to the
leaderboard; `break-even 6.4 bps — your assumed cost is 1.5` next to the equity
curve. Notes are generated from the tearsheet's `warnings` array plus rules in
`web/src/lib/critique.ts`. The app argues with itself, in the margin, in the
researcher's own vocabulary.

**Motion:** one orchestrated page-load — the Fold Strip draws left to right over
600ms, charts fade in after. Hover states are instant. Nothing else animates.
`prefers-reduced-motion` disables the draw.

**Colour is never the only encoder:** long/short use ▲/▼ glyphs, purge/embargo
use hatching, pass/warn/fail use text labels.

### 13.3 Stack

Vite · React 19 · TypeScript strict · TanStack Router (file-based) · TanStack
Query · Tailwind v4 with the tokens above as `@theme` · Radix primitives for
menus/dialogs/tooltips (unstyled, styled by us) · `lightweight-charts` for OHLC ·
`uPlot` for equity/drawdown/rolling series (canvas, handles 100k points) ·
hand-written SVG for the Fold Strip, monthly heatmap, SHAP beeswarm, calibration
curve · `@tanstack/react-virtual` for the trades table.

No component library with an opinion (no MUI/Chakra/AntD). No `recharts` — it
will not survive 100k points and its defaults are recognisable.

### 13.4 Routes

| Route | Content |
|---|---|
| `/` | Run index — the notebook. Sortable table: run id, config name, symbol, span, selected model, net Sharpe, DSR, PBO, warnings. Filter by symbol/date/flag. |
| `/runs/$id` | Overview. Fold Strip hero → headline metrics with CIs → equity + underwater → margin critique. |
| `/runs/$id/data` | Quality report, coverage calendar heatmap, gap list, OHLC violations. |
| `/runs/$id/features` | Feature manifest table (name, family, formula, lookback), clustered correlation matrix, IC by feature, pruning drop list. |
| `/runs/$id/folds` | Per-fold metrics table + small-multiple equity curves, purge/embargo counts. |
| `/runs/$id/models` | Leaderboard, ROC/PR, confusion matrix, calibration/reliability diagram, learning curves, shuffled-label comparison. |
| `/runs/$id/backtest` | Candlestick with trade markers, trades table (virtualised), monthly heatmap, rolling Sharpe, cost-sensitivity curve. |
| `/runs/$id/explain` | SHAP beeswarm, dependence plots, single-prediction waterfall with a date picker. |
| `/compare?a=&b=` | Two runs side by side: config diff (as a real text diff), metric deltas with CIs, overlaid equity curves. |
| `/jobs/$id` | Live log console (SSE), phase progress, cancel. |
| `/config` | Rendered resolved config with a diff against defaults. Read-only; editing happens in YAML. |

### 13.5 Frontend quality bar

- Initial JS ≤ 200 KB gzipped; route-level code splitting.
- Equity chart with 100k points pans at 60fps (server pre-downsamples with LTTB
  to the requested pixel width; full resolution only on export).
- Every interactive element keyboard-reachable with a visible focus ring in
  `--prussian`.
- Loading states are skeletons matched to final layout — no spinners, no layout
  shift.
- Empty state on `/` is an invitation with the exact CLI command to create the
  first run, copyable.
- Error states name the cause and the fix: "Run a3f91c has no backtest artifact.
  It failed during the backtest phase — see the job log." Never "Something went
  wrong."
- Responsive to 768px (the margin collapses into inline callouts).
- Vitest for `lib/` logic, Playwright smoke test covering the golden path.

---

## 14. API and job runner

### 14.1 Endpoints (`/api/v1`)

```
GET    /health                          → version, git sha, artifact store path
GET    /system/config/schema            → JSON schema for the config

GET    /datasets                        → list from duckdb catalog
GET    /datasets/{dataset_id}           → detail + quality report

GET    /runs?symbol=&status=&limit=&cursor=      → paginated index
GET    /runs/{run_id}                            → tearsheet.json
GET    /runs/{run_id}/config                     → resolved config + diff vs defaults
GET    /runs/{run_id}/folds                      → split definitions + per-fold metrics
GET    /runs/{run_id}/series/{name}?width=1200   → LTTB-downsampled series
                                                   (equity|drawdown|rolling_sharpe|price)
GET    /runs/{run_id}/trades?limit=&cursor=      → paginated trades
GET    /runs/{run_id}/predictions?fold=          → paginated predictions
GET    /runs/{run_id}/leaderboard
GET    /runs/{run_id}/shap/summary
GET    /runs/{run_id}/shap/explain?ts=           → single-prediction contributions
GET    /runs/{run_id}/export/{artifact}          → file download
GET    /runs/compare?a=&b=                       → metric deltas + config diff

POST   /jobs           {type: ingest|features|train|backtest|full_run, config_ref}
GET    /jobs/{job_id}
GET    /jobs/{job_id}/events                     → SSE log stream
POST   /jobs/{job_id}/cancel
```

Conventions: cursor pagination, `ETag`/`If-None-Match` on immutable run
resources with `Cache-Control: immutable`, RFC 9457 problem+json errors,
OpenAPI 3.1 auto-generated, TS client generated from it into
`web/src/api/generated/` via `openapi-typescript` in the justfile.

### 14.2 Job runner

`ProcessPoolExecutor` (max 1–2 workers) + SQLite for job state and a rolling log
table. No Celery, no Redis — ADR 0006 explains why single-node in-process is the
right call for a single-user research tool, and what you'd change at scale
(that answer is the interesting part).

Jobs report structured progress: `phase`, `step`, `total`, `message`. The SSE
stream carries both progress events and log lines. Jobs are cancellable and
crash-safe: a killed job's run directory is marked `status: failed` with the
traceback, never left half-written and presented as valid.

---

## 15. Build roadmap

Each phase: **deliverables → exit criteria → verify command.** Do not start
phase N+1 until phase N's exit criteria pass.

### Phase 0 — Foundations
Deliverables: uv workspace, both packages installable, ruff + mypy strict on
`alphacore`, pytest with coverage, justfile, pre-commit, CI skeleton, loguru
setup with JSON sink to `runs/{run_id}/logs.jsonl`, determinism helpers, run
manifest + content hashing, artifact store writer, import-linter contract.
Exit: `just verify` green on an empty test suite; `alpha --help` works; a
throwaway run dir is created with a valid manifest.

### Phase 1 — Data layer
Deliverables: `BarProvider` protocol, `csv` + `yahoo` providers, pandera bar
schema, calendar grids, parquet lake, duckdb catalog, quality report, `alpha data
pull` CLI, committed sample fixtures.
Exit: pulling SPY daily produces a validated parquet + catalog row + quality
report; schema violations raise with a useful message; offline tests pass with no
network.

### Phase 2 — Features
Deliverables: every primitive in §7.2, registry, transforms, pipeline, pruning,
`docs/data-dictionary.md` generated from the registry.
Exit: golden indicator tests pass to `1e-9`; `test_future_poison` passes over the
whole registry; data dictionary regenerates identically (CI check).

### Phase 3 — Labels and weights
Deliverables: all four labelling methods, `t_end` on every label, uniqueness and
time-decay weights, class-balance reporting.
Exit: triple-barrier verified against a hand-computed 20-bar fixture; uniqueness
weights sum correctly on a synthetic overlapping set; per-fold quantile test
passes.

### Phase 4 — Cross-validation
Deliverables: PurgedKFold, WalkForward, CPCV, split serialisation, the CI grep
ban on `train_test_split`/`KFold`.
Exit: `test_label_span_purged` passes for all three schemes across randomised
horizons (hypothesis); splits round-trip through JSON.

### Phase 5 — Models and leaderboard
Deliverables: `AlphaModel` protocol, linear + three GBM adapters, all baselines
including shuffled-label, calibration, OOF prediction assembly, leaderboard
builder.
Exit: `test_shuffled_labels_have_no_edge` passes; every model trains on the smoke
config in under 60s; leaderboard JSON validates against the tearsheet schema.

### Phase 6 — Signals and risk
Deliverables: threshold + hysteresis logic, three sizing methods, stops, limits,
kill switch.
Exit: property test — sizing never exceeds `max_leverage`; hysteresis strictly
reduces flip count vs plain thresholding on random walks.

### Phase 7 — Backtest engine
Deliverables: event loop, cost model, portfolio accounting, ledger, reference
engine, cost-sensitivity sweep.
Exit: `test_no_same_bar_execution` passes; `test_perfect_foresight_sanity` passes;
engine matches `reference.py` to `1e-8` under the frictionless config; cash +
position value reconciles to equity every bar (assertion inside the loop under
`--strict`).

### Phase 8 — Metrics and tearsheet
Deliverables: full metric set, PSR/DSR/minTRL, PBO via CSCV, block bootstrap CIs,
break-even cost, tearsheet builder + JSON schema, CSV/parquet exporters,
generated research markdown.
Exit: metrics validated against hand-computed fixtures; DSR reproduces the
published worked example; tearsheet validates against its schema; determinism
test passes.

### Phase 9 — Explainability
Deliverables: SHAP for tree models on a sampled OOF set, summary parquet,
dependence data, single-prediction contributions.
Exit: SHAP values sum to `f(x) − E[f(x)]` within tolerance; runtime on the smoke
config under 30s.

### Phase 10 — Pipelines and CLI
Deliverables: `alpha data pull|inspect`, `alpha features build`, `alpha train`,
`alpha backtest`, `alpha run` (full golden path), `alpha report`, `alpha runs
list|show|compare`. Rich console output.
Exit: `alpha run -c configs/experiments/smoke.yaml` completes end to end and
writes a complete run directory. This is the integration gate for everything
before it.

### Phase 11 — API service
Deliverables: every endpoint in §14.1, job runner, SSE, OpenAPI, generated TS
client.
Exit: schemathesis contract tests pass; a run created via `POST /jobs` is fully
readable through the API; SSE streams logs in real time.

### Phase 12 — Web app
**Do the design pass first**: write `docs/design-plan.md` with the token system,
ASCII wireframes for `/` and `/runs/$id`, and the self-critique required by the
frontend-design skill. Then build shell + routing + tokens, then the Fold Strip,
then charts, then the margin critique engine, then polish.
Exit: golden-path Playwright test passes; Lighthouse ≥ 95 on
performance/accessibility/best-practices; bundle budget met; keyboard-only walk
through every route works; screenshots committed to `docs/`.

### Phase 13 — Sequence models (optional extra)
Deliverables: torch adapters behind the lazy import, windowing with fold-aware
purging, training loop with early stopping and checkpoints, ensemble with GBM.
Exit: `import alphacore` works with torch uninstalled; sequence model appears on
the leaderboard; window-purge test passes.

### Phase 14 — Packaging and docs
Deliverables: multi-stage Dockerfile + compose, GitHub Actions CI (lint, type,
test, coverage gate 85% on `alphacore`, web typecheck + build + Playwright,
docker build), README with screenshots and a 5-minute quickstart,
`docs/methodology.md`, mermaid architecture diagrams rendered to SVG, three
example experiment configs, the generated research report for one real run.
Exit: `docker compose up` serves the app with the sample dataset on a clean
machine; CI green; a stranger can go from `git clone` to a rendered tearsheet in
under 10 minutes following only the README.

---

## 16. Golden path

```bash
just setup                                   # uv sync, npm ci, pre-commit install
alpha data pull -c configs/experiments/spy_daily_tb.yaml
alpha data inspect --dataset yahoo:SPY:1d    # prints the quality report
alpha run -c configs/experiments/spy_daily_tb.yaml
#   → ingest ✓  features ✓  labels ✓  cv ✓  train ✓  tune ✓
#     backtest ✓  metrics ✓  explain ✓  report ✓
#   → runs/2026-07-27T14-22-01Z_spy-daily-tb_a3f91c/
just serve                                   # api :8000 + web :5173
alpha runs compare a3f91c b7e204
```

Run directory contract:

```
runs/{run_id}/
├── manifest.json          # config, hashes, versions, timings, status
├── config.resolved.yaml
├── tearsheet.json         # ← the canonical result artifact
├── quality.json
├── features.json          # feature manifest
├── folds.json             # split definitions
├── equity.parquet  drawdown.parquet  rolling_sharpe.parquet
├── trades.parquet  predictions.parquet  ledger.parquet
├── shap_summary.parquet
├── leaderboard.csv  metrics.csv  feature_importance.csv
├── research.md
├── models/{model_name}/model.{txt,json,pt} + calibrator.pkl
└── logs.jsonl
```

---

## 17. Tooling

```makefile
# justfile
setup:      uv sync --all-extras && cd web && npm ci && pre-commit install
fmt:        uv run ruff format . && cd web && npm run format
lint:       uv run ruff check . && uv run import-linter && cd web && npm run lint
types:      uv run mypy packages/alphacore/src packages/alphaservice/src && cd web && npm run typecheck
test:       uv run pytest -q --cov=alphacore --cov-fail-under=85
test-leak:  uv run pytest tests/leakage -q -x
smoke:      uv run alpha run -c configs/experiments/smoke.yaml
e2e:        cd web && npx playwright test
verify:     just fmt lint types test test-leak smoke
serve:      uv run uvicorn alphaservice.api.app:app --reload & cd web && npm run dev
docs:       uv run python -m alphacore.features.registry --emit-docs > docs/data-dictionary.md
```

`just verify` is the gate. CI runs `just verify` plus `e2e` and the docker build.

---

## 18. Definition of done

A phase is done when all of these are true:

- [ ] `just verify` is green
- [ ] Public functions have type hints and docstrings with an `Args`/`Returns`/
      `Raises` section; every financial formula appears in the docstring
- [ ] New behaviour has unit tests; new invariants have property or leakage tests
- [ ] No `# type: ignore` without an adjacent comment justifying it
- [ ] No `print()` — loguru only
- [ ] No magic numbers — they live in config or named module constants
- [ ] Errors are typed exceptions from `alphacore.errors`, not bare `ValueError`
      with a string
- [ ] `PROGRESS.md` updated; any deviation has an ADR
- [ ] Public API of the module is exported from its package `__init__`

---

## 19. Anti-patterns that will sink this project

If any of these appear in the final output, the project reads as a tutorial
regardless of how much code was written:

- **Accuracy above ~0.55 on next-bar direction.** Almost certainly leakage. Go
  find it. Real short-horizon edges live at 0.51–0.54.
- **Sharpe above ~2.5 on daily equity data after costs.** Same conclusion.
- **A backtest with zero costs presented as a result.**
- **Same-bar execution** — signal from `close[t]`, fill at `close[t]`.
- **Scaling the full dataset before splitting.**
- **`train_test_split(shuffle=True)`** on time series.
- **Reporting only the best model** out of forty tried, with no DSR haircut and
  no trial count.
- **An equity curve with no benchmark and no drawdown panel.**
- **Retraining on the test set** to "use all the data."
- **Global quantile-based labels.**
- **Dropping NaNs with `df.dropna()`** without knowing which warm-up region you
  just deleted.
- **Charts with no axis labels, no units, and no cost/lag provenance.**
- **A README that shows code but never shows a result.**

The single most impressive thing this repository can contain is a run where the
model **does not** find an edge, reported honestly, with the diagnostics that
prove the pipeline would have detected one if it existed. Build that case first;
it is the strongest possible evidence that the platform works.

---

## 20. README requirements (Phase 14)

The README is what gets read. In order:

1. One sentence on what this is, one on what it deliberately refuses to do.
2. A screenshot of the run overview page with the Fold Strip visible.
3. The honest headline result from a real run — including the DSR and PBO, even
   if they're unflattering.
4. 5-minute quickstart (docker compose path and local path).
5. Architecture diagram (rendered SVG from `docs/diagrams/`).
6. "How this avoids the usual backtest failures" — a table mapping each
   invariant in §4 to the test that enforces it. This section is the reason a
   reviewer will keep reading.
7. Links to `docs/methodology.md` and the ADR index.
8. Disclaimer.
