import dataclasses

import pytest
import requests

from src.core.nvidia_lm import CompletionResult, NvidiaLM, TokenLogprob


def _build_mock_response(mocker, content="Bullish", with_logprobs=True):
    mock_response = mocker.Mock()
    mock_response.status_code = 200
    payload = {
        "choices": [
            {
                "message": {"content": content},
            }
        ]
    }
    if with_logprobs:
        payload["choices"][0]["logprobs"] = {
            "content": [
                {
                    "token": "Bull",
                    "logprob": -0.1,
                    "top_logprobs": [
                        {"token": "Bull", "logprob": -0.1},
                        {"token": "Bear", "logprob": -1.5},
                    ],
                },
                {
                    "token": "ish",
                    "logprob": -0.2,
                    "top_logprobs": [
                        {"token": "ish", "logprob": -0.2},
                        {"token": "y", "logprob": -2.0},
                    ],
                },
            ]
        }
    mock_response.json.return_value = payload
    return mock_response


def test_request_body_contains_temperature_and_top_logprobs(mocker):
    mock_post = mocker.patch("requests.post")
    mock_post.return_value = _build_mock_response(mocker)

    lm = NvidiaLM(api_key="test_key", model="nvidia/nemotron-3-super-120b-a12b")
    lm.generate("Predict the stock direction.")

    called_json = mock_post.call_args.kwargs["json"]
    assert called_json["temperature"] == 0.0
    assert called_json["logprobs"] is True
    assert called_json["top_logprobs"] == 20


def test_response_parsed_into_completion_result(mocker):
    mock_post = mocker.patch("requests.post")
    mock_post.return_value = _build_mock_response(mocker)

    lm = NvidiaLM(api_key="test_key", model="nvidia/nemotron-3-super-120b-a12b")
    result = lm.generate("Predict the stock direction.")

    assert isinstance(result, CompletionResult)
    assert result.content == "Bullish"
    assert len(result.logprobs) == 2
    first = result.logprobs[0]
    assert isinstance(first, TokenLogprob)
    assert first.token == "Bull"
    assert first.logprob == -0.1
    assert len(first.top_logprobs) == 2
    assert first.top_logprobs[0]["token"] == "Bull"


def test_completion_result_is_frozen_dataclass():
    assert dataclasses.is_dataclass(CompletionResult)
    params = CompletionResult.__dataclass_params__
    assert params.frozen is True


def test_token_logprob_is_frozen_dataclass():
    assert dataclasses.is_dataclass(TokenLogprob)
    params = TokenLogprob.__dataclass_params__
    assert params.frozen is True


def test_temperature_override_passed_through(mocker):
    mock_post = mocker.patch("requests.post")
    mock_post.return_value = _build_mock_response(mocker)

    lm = NvidiaLM(api_key="test_key", model="nvidia/nemotron-3-super-120b-a12b")
    lm.generate("Predict the stock direction.", temperature=0.7)

    called_json = mock_post.call_args.kwargs["json"]
    assert called_json["temperature"] == 0.7


def test_timeout_raises_timeout_error(mocker):
    mock_post = mocker.patch("requests.post")
    mock_post.side_effect = requests.exceptions.Timeout("boom")

    lm = NvidiaLM(api_key="test_key", model="nvidia/nemotron-3-super-120b-a12b")
    with pytest.raises(TimeoutError):
        lm.generate("Predict.")


def test_default_timeout_is_fifteen_seconds(mocker):
    mock_post = mocker.patch("requests.post")
    mock_post.return_value = _build_mock_response(mocker)

    lm = NvidiaLM(api_key="test_key", model="nvidia/nemotron-3-super-120b-a12b")
    lm.generate("Predict.")

    assert mock_post.call_args.kwargs["timeout"] == 15.0


def test_missing_top_logprobs_raises_clear_error(mocker):
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
                    ]
                },
            }
        ]
    }
    mock_post.return_value = mock_response

    lm = NvidiaLM(api_key="test_key", model="nvidia/nemotron-3-super-120b-a12b")
    with pytest.raises(RuntimeError, match=r"top_logprobs"):
        lm.generate("Predict.")


def test_missing_logprobs_section_raises_clear_error(mocker):
    mock_post = mocker.patch("requests.post")
    mock_post.return_value = _build_mock_response(mocker, with_logprobs=False)

    lm = NvidiaLM(api_key="test_key", model="nvidia/nemotron-3-super-120b-a12b")
    with pytest.raises(RuntimeError):
        lm.generate("Predict.")
