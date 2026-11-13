# Polymarket NVIDIA CEO Price Forecaster (Minute-by-Minute, 4 Hours Ahead)

Solution and submission kit for the make.fun bounty:
**"Forecast Polymarket's next NVIDIA CEO prices minute by minute, 4 hours ahead"** (`make.fun/bounty/polymarket-nvidia-ceo-forecast-01`).

---

## 1. Overview & Objective

This project forecasts the probability trajectory for candidate contracts in Polymarket's **"Next Permanent CEO of NVIDIA"** prediction market minute-by-minute over a 240-minute (4-hour) forward horizon.

### Key Candidates Tracked:
1. **Ian Buck** (VP of Hyperscale & HPC)
2. **Colette Kress** (CFO / Interim CEO)
3. **Rene Haas** (Arm CEO / External Frontrunner)
4. **Madison Huang**
5. **Other / External**

---

## 2. Methodology & Mathematical Architecture

Forecasting prediction market probabilities requires solving two fundamental constraints that typical financial time-series models fail to respect:
1. **Bounded Domain:** Every probability $p_i(t) \in [0, 1]$.
2. **Mutually Exclusive Sum-to-One:** $\sum_{i=1}^K p_i(t) = 1.0$ at every forward minute $t$.

### Pipeline Steps:

1. **Feature Engineering (`data_pipeline.py`):**
   - **Rolling VWAP:** 5-minute, 15-minute, and 60-minute volume-weighted average price.
   - **Exponential Moving Averages (EMA):** Fast ($\alpha_{10}$), medium ($\alpha_{30}$), and slow ($\alpha_{120}$) trend indicators.
   - **Order Flow & Spread Imbalance:** Real-time bid-ask spread and volume density.
   - **Realized Volatility:** Rolling 30-minute standard deviation of log returns:
     $$\sigma_t = \sqrt{\frac{1}{W-1}\sum_{j=0}^{W-1} \left(r_{t-j} - \bar{r}\right)^2}$$

2. **Log-Odds Modeling (`forecasting_engine.py`):**
   - Transform initial candidate probabilities into unconstrained log-odds space:
     $$z_i(0) = \ln(p_i(0))$$
   - Model trajectory using an autoregressive drift with mean-reversion (Ornstein-Uhlenbeck dynamics in logit space):
     $$z_i(t+1) = z_i(t) + \delta_i(t) + \kappa \cdot (\bar{z}_i - z_i(t))$$
     where $\delta_i(t)$ decays exponentially ($\delta_i(t+1) = \gamma \cdot \delta_i(t)$) and $\kappa$ represents the mean-reversion speed towards the rolling anchor $\bar{z}_i$.
   - **Softmax Multinomial Projection:**
     $$\hat{p}_i(t) = \frac{\exp(z_i(t))}{\sum_{j=1}^K \exp(z_j(t))}$$
     This ensures valid probabilities and strict sum-to-one compliance at every step $t \in [1, 240]$.

3. **Validation & Metrics (`evaluator.py`):**
   - Rigorous checks confirming bounds $[0, 1]$, exactly 240 steps, and sum equality.
   - Multi-step Brier Score and Mean Absolute Error (MAE) computation.

---

## 3. Project Structure

```
polymarket-nvidia-ceo-forecast/
├── README.md               # Project documentation & methodology
├── polymarket_client.py    # Polymarket Gamma & CLOB API client with fallback
├── data_pipeline.py        # VWAP, EMA, realized volatility & feature generation
├── forecasting_engine.py   # 240-step log-odds forecaster with softmax projection
├── evaluator.py            # Brier Score, MAE, and constraint validation
├── main.py                 # Pipeline execution script
├── submission.json         # Output JSON submission artifact
└── submission.csv          # Output CSV submission artifact
```

---

## 4. How to Run

Run the complete pipeline from this directory:

```bash
cd /Users/ishanchathuranga/.gemini/antigravity/scratch/polymarket-nvidia-ceo-forecast
python3 main.py
```

### Generated Artifacts:
* `submission.json`: Full machine-readable payload containing candidate metadata and minute-by-minute probability predictions.
* `submission.csv`: Tabular predictions with columns `[step, timestamp_utc, candidate, forecast_probability, lower_90, upper_90]`.
