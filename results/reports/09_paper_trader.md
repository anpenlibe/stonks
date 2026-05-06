# Sprint 09 — Paper Trader

## What got built

`scripts/09_paper_trader.py` — converts the RELIANCE one-step-ahead log-return predictions reconstructed in Sprint 08 (ARIMA / MLP / LSTM) into BUY / HOLD / SELL signals at the directive's ±0.5% threshold, then simulates a portfolio from ₹1,00,000 over the 562-day test window (2024-01-01 → 2026-04-10) against a buy-and-hold benchmark. Long-only, fractional shares, no transaction costs (all per directive §09).

The prediction-reconstruction code from `08_benchmark.py` is reused verbatim via the same `importlib.util.spec_from_file_location` trick — `bench_mod.reliance_arima_predictions / reliance_mlp_predictions / reliance_lstm_predictions` are imported and called directly. Zero re-derivation of model code, and the predictions land byte-identical to Sprint 08.

Trades execute at that day's **Open** price, joined from `data/raw/RELIANCE.csv` using the same `skiprows=3 + header=None + names=...` pattern that `align_prices.py` already uses on the yfinance dump.

## Outputs

- `results/trades/arima_trade_log.csv` — 562 rows, schema `date, signal, executed, price, shares, cash, portfolio_value`. **0 executed signals.**
- `results/trades/mlp_trade_log.csv` — 562 rows. **1 executed signal** (one BUY on 2024-02-05 at ₹1,460.75; never SELLs).
- `results/trades/lstm_trade_log.csv` — 562 rows. **0 executed signals.**
- `results/trades/buy_and_hold_trade_log.csv` — 562 rows, single BUY on day 1 at ₹1,290.28 then mark-to-market.
- `results/metrics/paper_trading_summary.csv` — 4 model rows × 9 columns (model, total/annualised return %, Sharpe, max drawdown %, n_trades, completed round-trips, win rate %, final equity).
- `results/figures/equity_curves.png` — four lines (ARIMA, MLP, LSTM, BuyAndHold), shared x-axis on test dates, ₹1,00,000 baseline marked.

## Headline numbers

| model | total return % | annualised % | Sharpe | max DD % | n trades | final equity |
|---|---:|---:|---:|---:|---:|---:|
| BuyAndHold | **+3.62** | +1.61 | −0.089 | −29.43 | 1 | ₹1,03,621.32 |
| ARIMA | 0.00 | 0.00 | NaN | 0.00 | 0 | ₹1,00,000.00 |
| LSTM | 0.00 | 0.00 | NaN | 0.00 | 0 | ₹1,00,000.00 |
| MLP | **−8.47** | −3.89 | −0.351 | −29.43 | 1 | ₹91,528.32 |

Buy-and-hold wins. By a lot. Even the buy-and-hold Sharpe is faintly negative because the 6.5% Indian T-bill risk-free rate is well above the strategy's 1.6% annualised return — bonds beat RELIANCE-only buy-and-hold over this 27-month stretch, never mind the models. Max drawdown for both BuyAndHold and MLP is the same **−29.4%**, which makes sense: both end up long-only through the same trough; MLP just enters later and at a higher price (₹1,460.75 vs B&H's ₹1,290.28), guaranteeing a worse final outcome.

## Why ARIMA and LSTM never traded — the threshold-vs-prediction-σ mismatch

The signal threshold is ±0.5% (50 basis points of log return). The actual predicted-log-return distributions on the test window:

| model | min pred | max pred | std(pred) | days &gt; +0.5% | days &lt; −0.5% |
|---|---:|---:|---:|---:|---:|
| ARIMA(0,0,0) | +0.00090 | +0.00090 | 0.00000 | 0 | 0 |
| MLP | −0.00403 | +0.00619 | 0.00127 | 6 | 0 |
| LSTM | +0.00050 | +0.00086 | 0.00005 | 0 | 0 |

For comparison, the **realised** RELIANCE log-return std on the test window is **0.01373** — two orders of magnitude wider than the LSTM's predicted std (5e-5) and 10× wider than the MLP's (1.3e-3). The Sprint 04/06/07 finding ("models converge to a near-constant prediction near the in-sample mean") is what's biting here: ARIMA(0,0,0) literally is the in-sample drift constant (+0.0009 every day), and the LSTM has collapsed to a similarly tight band around it. Daily log-return shocks have σ ≈ 1.4% but the models, having found no exploitable signal, predict a tight ribbon around 0.1%.

So the threshold of ±0.5% — picked by the directive as "a reasonable starting point, would be optimised in a real system" — is almost an order of magnitude further from zero than the LSTM ever predicts. The MLP, with its slightly wider predicted band, fires BUY 6 times (on dates where it predicts > +0.5%) and SELL 0 times — and after its first BUY consumes all the cash, the remaining 5 BUY signals are no-ops (already long).

## Reading the result

This isn't a bug in the simulator — it's **the same noise-floor finding as Sprints 06/07/08, expressed in a P&L idiom rather than RMSE/DA**:

- Sprint 04: RELIANCE auto_arima winner = **(0,0,0)**.
- Sprint 06: RELIANCE MLP best val MSE at **epoch 4**, then plateau.
- Sprint 07: RELIANCE LSTM best val MSE at **epoch 1**.
- Sprint 08: DM test ARIMA vs LSTM, p = **0.327** → fail to reject equal accuracy.
- Sprint 09: at any honest threshold larger than the prediction-band width, **two of three models never trade and the third loses to buy-and-hold by ~12 percentage points**.

The directive flags the threshold as arbitrary and notes that beating buy-and-hold consistently is hard. That framing absorbs this result cleanly — the headline isn't "the models fail at trading", it's "the models predict a flat ribbon and the threshold filter is doing the right thing by rejecting those predictions as un-actionable". A model that fired hundreds of trades on near-zero predicted edges would be the worse outcome.

## Sanity checks

- ✓ All four trade logs have **562 rows** (one per test date).
- ✓ BuyAndHold final equity = ₹1,03,621.32 = ₹1,00,000 × 1,337.00 / 1,290.28 — matches the spot-check assertion in the script.
- ✓ MLP single executed BUY = 2024-02-05 at ₹1,460.75; subsequent BUY signals are recorded with `executed=False` because the position is already maxed out.
- ✓ Re-ran `python scripts/09_paper_trader.py` twice — `md5sum` of all four trade-log CSVs and `paper_trading_summary.csv` byte-identical across runs (CPU-pinned + seeded; same convention as Sprints 06/07/08).
- ✓ Sharpe NaN for ARIMA / LSTM is intentional (zero return variance → undefined Sharpe). The script emits a `[warn]` line and writes NaN, rather than dividing by zero.
- ✓ Win rate NaN across all models because there are zero **completed** round-trips (no SELLs ever fired). Documented in the column.

## Dead ends / things tried

- **Tighter threshold to force the LSTM to trade.** Briefly considered dropping the threshold to ±0.05% so the LSTM's [+0.0005, +0.0009] predicted band would fire BUY constantly. Dropped because (a) the directive specifies ±0.5%, and (b) at ±0.05% the LSTM would *always* predict above the BUY threshold and never SELL → identical behaviour to buy-and-hold but with worse fill on day 21 (first prediction date). Not interesting.
- **Allowing short positions on SELL.** Considered, dropped — directive says "On a SELL signal: liquidate all shares", not "go short". Long-only is the textbook simplification and matches the directive literally. Going short would also need a borrowing-cost model that's out of scope.
- **Integer share counts (real-exchange behaviour).** With ₹1,00,000 capital and RELIANCE around ₹1,300, integer shares would round capital allocation to ~76 shares = ₹98,800, leaving ₹1,200 idle. That ~1% bookkeeping leakage swamps the actual 0.04% Sharpe differences we're measuring. Fractional shares is the right call for a model-comparison harness; flagged as the assumption in code.
- **Including the multivariate LSTM as a fifth model.** Sprint 07's multivariate variant has Δ DA = +0.002 and the same near-constant prediction shape; it would also never cross the threshold. No new signal, more chart clutter.
- **Transaction costs.** Directive says no costs. With NSE retail brokerage at ~0.03% per side + STT, a 1-trade strategy like MLP would lose another ~0.06% of its already-negative return — wouldn't change the qualitative ranking.

## Queued for the next sprint

`scripts/10_sentiment.py` — VADER sentiment on `yfinance.Ticker("RELIANCE.NS").news` headlines, 5-day rolling smoothing, scatter against next-day log returns, and (data permitting) a sentiment-augmented LSTM that adds the rolling sentiment score as a second input feature. Honest expectation: yfinance returns ≈ 30–90 days of headlines so the correlation analysis will be small-sample, and the sentiment-augmented LSTM may not have enough overlap to train cleanly. The report will cover the limitation up front per directive §10.
