import pytest
from src.evaluate.metrics import Evaluator

def test_evaluator():
    evaluator = Evaluator()
    predictions = [
        {"target_direction": 1, "predicted_direction": 1, "raw_confidence": 0.9, "penalized_confidence": 0.45},
        {"target_direction": -1, "predicted_direction": 1, "raw_confidence": 0.8, "penalized_confidence": 0.4},
        {"target_direction": 1, "predicted_direction": 1, "raw_confidence": 0.85, "penalized_confidence": 0.85}
    ]
    
    raw_acc, raw_conf, mem_acc, mem_conf = evaluator.evaluate(predictions)
    
    assert abs(raw_acc - (2/3)) < 1e-5
    assert abs(raw_conf - 0.85) < 1e-5
    assert mem_conf < raw_conf
