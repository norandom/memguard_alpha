# Recall Guard (MemGuard-Alpha)

**A DSPy-based pipeline to mitigate look-ahead bias in financial quantitative agents using Membership Inference Attack (MIA) penalties and DSPy Math Reasoning.**

---

## 📌 The Problem: Look-Ahead Bias
Financial LLMs are often trained on web-scale data containing post-hoc market analyses and explicit descriptions of historical asset performance. When evaluating an LLM on historical financial data (like Apple's 2021 earnings), the model will often **regurgitate the answer from memory** rather than reasoning about the actual data. 

In a quantitative trading system, this causes extreme, false confidence, which leads to catastrophic overfitting in backtests.

## 🚀 The Solution
This project solves look-ahead bias through a two-layered DSPy pipeline:

1. **Input Abstraction (DSPy Math Reasoning):** Uses `dspy.ChainOfThought` to intercept the raw financial context and mathematically extract pure metrics (e.g., "Revenue up 54% YoY") while stripping all identifiable entity names ("AAPL") and dates ("2021"). By breaking the temporal anchor, the model is forced to reason on abstract factors rather than cheat from memory.
2. **Output Filtering (MIA Scorer):** Calculates the continuous token-level log-probabilities directly from the NVIDIA API. If the LLM generates tokens with unnaturally high probability (Loss < 0.5, Min-K% > -0.5), it detects the statistical signature of memorization and applies a strict penalty to the model's confidence.

---

## 🛠️ Setup & Installation

This project is managed by `uv`.

1. **Environment Setup:**
   Make sure you have python 3.14. Install dependencies:
   ```bash
   uv sync
   ```

2. **API Keys:**
   This pipeline relies on NVIDIA's inference endpoints to extract raw token `logprobs`. 
   Create a `.env` file in the root directory (or in `papers/.env`) and add your API key:
   ```env
   NVIDIA_API_KEY="your_actual_key_here"
   ```

3. **Dataset:**
   A sample dataset replicating the structure of `Look-Ahead-Bench` is located at `data/lookahead_bench_sample.jsonl`.

---

## 🔬 How to Verify & Run (Step-by-Step)

You can run the full pipeline to observe the "Scaling Paradox" and the effects of MemGuard.

### Running a Specific Model
To test a single model, run:
```bash
uv run python main.py --model nvidia/nemotron-3-super-120b-a12b
```

### Running the Entire Ensemble
To evaluate all models and observe how different model sizes handle memorization, use the `--all` flag:
```bash
uv run python main.py --all
```

---

## 📊 Understanding the Results

When you run the pipeline, you will see a comparison between **Raw Avg Confidence** and **MemGuard Avg Conf**.

### Scenario A: The Scaling Paradox (Without Input Abstraction)
*If you were to disable the DSPy Input Masker*, here is what happens mathematically:
- **120B Model (Heavy Memorization):** The massive model recognizes "AAPL 2021" and perfectly regurgitates the answer. Because it memorized the text, its token probabilities are unnaturally high (Loss < 0.5). MemGuard catches this "cheating" and slashes its 85% confidence to **42.5%**.
- **9B Model (Standard Uncertainty):** The smaller model lacks the capacity to memorize specific dates. It guesses. Its token probabilities reflect standard uncertainty (Loss > 0.5). MemGuard realizes it is not cheating, and leaves its confidence at **85%**.

### Scenario B: DSPy Math Reasoning (Current Master)
Because we have successfully integrated **DSPy ChainOfThought Input Abstraction**, the inputs are scrubbed before they reach the predictor. 
- The 120B model only sees: *"Asset X with a 54% YoY revenue increase"*. 
- Because the model can no longer "cheat" by looking up AAPL's history, its mathematical certainty normalizes. 
- MemGuard correctly sees that the model is no longer regurgitating memorized data.
- **Result:** You will see the `MemGuard Avg Conf` perfectly match the `Raw Avg Confidence`, proving that the temporal anchor has been successfully broken without destroying the underlying financial signal!
