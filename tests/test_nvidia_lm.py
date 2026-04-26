import pytest
from src.models.nvidia_lm import NvidiaLM

def test_nvidia_lm_requests_logprobs(mocker):
    mock_post = mocker.patch("requests.post")
    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [
            {
                "message": {"content": "Bullish"},
                "logprobs": {
                    "content": [
                        {"token": "Bull", "logprob": -0.1},
                        {"token": "ish", "logprob": -0.2}
                    ]
                }
            }
        ]
    }
    mock_post.return_value = mock_response

    lm = NvidiaLM(api_key="test_key", model="nvidia/nemotron-3-super-120b-a12b")
    response, logprobs = lm.generate_with_logprobs("Predict the stock direction.")
    
    called_json = mock_post.call_args.kwargs["json"]
    assert called_json.get("logprobs", False) is True or called_json.get("include_logprobs", False) is True
    
    assert response == "Bullish"
    assert len(logprobs) == 2
    assert logprobs[0]["token"] == "Bull"
    assert logprobs[0]["logprob"] == -0.1
