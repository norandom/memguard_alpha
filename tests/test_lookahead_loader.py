import pytest
import json
import dspy
from src.dataset.lookahead_loader import LookaheadLoader

def test_lookahead_loader_loads_data(tmp_path):
    mock_data = [
        {"ticker": "AAPL", "date": "2021-01-15", "context": "AAPL shares rose on January 15, 2021...", "target_direction": 1},
        {"ticker": "MSFT", "date": "2021-01-15", "context": "MSFT reports earnings...", "target_direction": -1}
    ]
    
    data_file = tmp_path / "mock_lookahead.jsonl"
    with open(data_file, "w") as f:
        for d in mock_data:
            f.write(json.dumps(d) + "\n")
            
    loader = LookaheadLoader(str(data_file))
    trainset, devset = loader.load_split(train_ratio=0.5)
    
    assert len(trainset) == 1
    assert len(devset) == 1
    
    # Assert they are DSPy Examples
    assert isinstance(trainset[0], dspy.Example)
    assert hasattr(trainset[0], "context")
    assert hasattr(trainset[0], "target_direction")
