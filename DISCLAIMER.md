# Disclaimer

**Alpha Lab is research software. It is not investment advice.**

This system exists to measure how much confidence a claimed predictive signal
deserves. It is deliberately built so that it can — and often should — conclude
that a signal deserves none.

- **No recommendation is produced.** The system reports evidence: out-of-sample
  metrics, confidence intervals, and the statistical corrections that make those
  numbers honest. It never says "buy", "sell", or "this strategy works".
- **Backtested performance is not evidence of future performance.** It is not
  even strong evidence of past performance. A backtest is a simulation whose
  result depends entirely on assumptions about fills, costs, borrow, and
  survivorship — all of which this system makes explicit precisely because they
  are usually hidden.
- **There is no execution path.** No broker connectivity, no order routing, no
  paper trading. This is a deliberate design constraint, not a missing feature.
- **The statistical corrections are not a guarantee.** Deflated Sharpe ratio and
  probability of backtest overfitting reduce the chance of being fooled by a
  multiple-testing artifact. They do not eliminate it.
- **The data may be wrong.** Vendor price history contains errors, revisions,
  and survivorship bias. The quality report surfaces what it can detect. It
  cannot detect everything.

Nothing here is a recommendation to buy or sell any security or financial
instrument. Do not use this software as the basis of a financial decision.
Trading involves risk of loss.
