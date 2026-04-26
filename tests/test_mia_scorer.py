import pytest
from src.pipeline.mia_scorer import MIAScorer

def test_mia_scorer():
    logprobs = [
        {"token": "The", "logprob": -0.1},
        {"token": "stock", "logprob": -0.2},
        {"token": "went", "logprob": -0.5},
        {"token": "up", "logprob": -0.1},
        {"token": ".", "logprob": -0.1}
    ]
    
    scorer = MIAScorer()
    loss = scorer.calculate_loss(logprobs)
    
    assert abs(loss - 0.2) < 1e-5
    
    mink = scorer.calculate_mink(logprobs, k=0.4)
    assert abs(mink - (-0.35)) < 1e-5

    penalized = scorer.apply_penalty(confidence=0.9, loss_score=0.2, mink_score=-0.35)
    assert 0.0 <= penalized <= 1.0
