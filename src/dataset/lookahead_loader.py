import json
import random
import dspy

class LookaheadLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path
        
    def load_split(self, train_ratio=0.8, seed=42):
        with open(self.file_path, "r") as f:
            data = [json.loads(line) for line in f]
            
        examples = [
            dspy.Example(
                ticker=d.get("ticker", ""),
                date=d.get("date", ""),
                context=d.get("context", ""),
                target_direction=d.get("target_direction", 0)
            ).with_inputs("ticker", "date", "context")
            for d in data
        ]
        
        random.seed(seed)
        random.shuffle(examples)
        
        split_idx = int(len(examples) * train_ratio)
        return examples[:split_idx], examples[split_idx:]
