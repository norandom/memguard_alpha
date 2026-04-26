import os
import argparse
from src.dataset.lookahead_loader import LookaheadLoader
from src.models.nvidia_lm import NvidiaLM
from src.pipeline.predict_module import RecallGuardPredictor
from src.evaluate.metrics import Evaluator
from dotenv import load_dotenv

load_dotenv()
load_dotenv("papers/.env")

def main():
    parser = argparse.ArgumentParser(description="Recall Guard Pipeline")
    parser.add_argument("--data", type=str, default="data/lookahead_bench_sample.jsonl", help="Path to Look-Ahead-Bench dataset")
    parser.add_argument("--model", type=str, default="nvidia/nemotron-3-super-120b-a12b")
    parser.add_argument("--all", action="store_true", help="Run through all specified models")
    args = parser.parse_args()

    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        print("Error: NVIDIA_API_KEY environment variable is not set. Please set it before running.")
        return

    # 1. Load Data
    if not os.path.exists(args.data):
        print(f"Error: Dataset '{args.data}' not found.")
        print("Please download a Look-Ahead-Bench JSONL dataset to this location or specify --data.")
        return
        
    loader = LookaheadLoader(args.data)
    trainset, devset = loader.load_split(train_ratio=0.8)
    print(f"Loaded {len(trainset)} train examples and {len(devset)} dev examples.")

    if args.all:
        models_to_run = [
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
            "nvidia/nemotron-3-super-120b-a12b",
            "nvidia/llama-3.3-nemotron-super-49b-v1.5",
            "nvidia/nvidia-nemotron-nano-9b-v2"
        ]
    else:
        models_to_run = [args.model]

    for model_name in models_to_run:
        print(f"\n{'='*40}")
        print(f"Initializing Recall Guard with model: {model_name}")
        print(f"{'='*40}")

        import dspy
        # Configure global DSPy LM for ChainOfThought so it can parse math reasoning
        # The 'openai/' prefix tells litellm to use the OpenAI API spec, and we point it to NVIDIA.
        dspy_lm = dspy.LM(f"openai/{model_name}", api_base="https://integrate.api.nvidia.com/v1", api_key=api_key)
        dspy.configure(lm=dspy_lm)

        # 2. Initialize Models
        lm = NvidiaLM(api_key=api_key, model=model_name)
        predictor = RecallGuardPredictor(nvidia_lm=lm)
        evaluator = Evaluator()

        # 3. Run Pipeline on Devset
        predictions = []
        print("Running predictions and applying MemGuard MIA penalties...")
        for idx, ex in enumerate(devset):
            print(f"  Processing Example {idx+1}/{len(devset)} (Ticker: {ex.ticker})")
            try:
                pred = predictor(ticker=ex.ticker, date=ex.date, context=ex.context)
                predictions.append({
                    "target_direction": ex.target_direction,
                    "predicted_direction": pred.direction,
                    "raw_confidence": pred.raw_confidence,
                    "penalized_confidence": pred.penalized_confidence
                })
            except Exception as e:
                print(f"  Error processing {ex.ticker}: {e}")

        # 4. Evaluate Results
        raw_acc, raw_conf, mem_acc, mem_conf = evaluator.evaluate(predictions)
        
        print("\n=== RECALL GUARD RESULTS ===")
        print(f"Model:                  {model_name}")
        print(f"Raw Accuracy:           {raw_acc:.2%}")
        print(f"Raw Avg Confidence:     {raw_conf:.4f}")
        print("-" * 30)
        print(f"MemGuard Accuracy:      {mem_acc:.2%}")
        print(f"MemGuard Avg Conf:      {mem_conf:.4f}")
        print("============================\n")

if __name__ == "__main__":
    main()
