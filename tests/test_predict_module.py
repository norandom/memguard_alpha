import pytest
from src.pipeline.predict_module import RecallGuardPredictor

def test_predict_module(mocker):
    mock_lm = mocker.Mock()
    mock_lm.generate_with_logprobs.return_value = (
        "Direction: 1\nConfidence: 0.9",
        [{"token": "Bull", "logprob": -0.1}]
    )
    
    predictor = RecallGuardPredictor(nvidia_lm=mock_lm)
    prediction = predictor(ticker="AAPL", date="2021-01-15", context="News...")
    
    assert prediction.direction == 1
    assert prediction.raw_confidence == 0.9
    assert prediction.loss_score == 0.1
    assert abs(prediction.penalized_confidence - 0.225) < 1e-5
