# Experimental divergence decoding

Use divergence decoding when your own generation runtime exposes logits before
each token is selected. The mathematical layer computes:

```text
adjusted_logits = main_logits + alpha * (older_logits - newer_logits)
```

`combine_logits` accepts three finite real vectors of shape `(vocabulary_size,)`
and a finite nonnegative `alpha`. Each position must refer to the same token ID
in all three vectors. It returns a new float64 array and preserves its inputs.
At `alpha=0`, it returns an independent copy of the validated main logits.

`decode_greedy` calls three model callbacks with the same immutable full prefix
at each step, applies the equation, and selects the highest adjusted logit.
Ties select the lowest token ID. It returns generated token IDs, including a
selected terminal EOS, and a finish reason of `eos` or `length`.

## Runnable example

Run this example with the default recall_guard installation. The callbacks are
synthetic and require no model downloads or credentials. Their outputs verify
the equation and token-loop behavior only; they provide no evidence of reduced
memorization.

```python
from dataclasses import replace

import numpy as np
from numpy.typing import NDArray

from recall_guard.core.divergence import combine_logits
from recall_guard.core.divergence_decode import ModelSource, decode_greedy


# Every logit vector has shape (3,); its axis indexes token IDs 0, 1, 2.
main_logits = np.array([2.0, 0.0, 0.0])
older_logits = np.array([0.0, 3.0, 0.0])
newer_logits = np.zeros(3)
adjusted_logits = combine_logits(
    main_logits, older_logits, newer_logits, alpha=1.0
)
np.testing.assert_array_equal(adjusted_logits, [2.0, 3.0, 0.0])


def main_next_logits(prefix_token_ids: tuple[int, ...]) -> NDArray[np.float64]:
    if prefix_token_ids[-1] == 1:
        return np.array([0.0, 0.0, 3.0])
    return main_logits.copy()


def older_next_logits(prefix_token_ids: tuple[int, ...]) -> NDArray[np.float64]:
    if len(prefix_token_ids) == 1:
        return older_logits.copy()
    return np.zeros(3)


def newer_next_logits(prefix_token_ids: tuple[int, ...]) -> NDArray[np.float64]:
    return np.zeros(3)


main_source = ModelSource(
    next_logits=main_next_logits,
    vocabulary_id="synthetic-vocabulary-v1",
    vocabulary_size=3,
    eos_token_ids=frozenset({2}),
    context_limit=8,
)
older_source = replace(main_source, next_logits=older_next_logits)
newer_source = replace(main_source, next_logits=newer_next_logits)

result = decode_greedy(
    [0],
    main_source=main_source,
    older_source=older_source,
    newer_source=newer_source,
    alpha=1.0,
    max_new_tokens=3,
)
assert result.token_ids == (1, 2)
assert result.finish_reason == "eos"
print(result.token_ids, result.finish_reason)
```

Expected output: `(1, 2) eos`. Token `0` is the input prompt, token `1` is the
first generated token, and token `2` is EOS.

## Caller responsibilities and limits

Each callback accepts the entire prompt-plus-generated prefix as a tuple of
token IDs and returns the next-token logits. Callbacks must use compatible
token meanings and prompt tokenization. Supply a shared `vocabulary_id` that
identifies that mapping, matching vocabulary sizes and EOS sets, and accurate
per-model context limits. The helper checks those declarations; it does not
independently verify tokenizer compatibility or training-cutoff history.
Choose appropriate older/newer auxiliary checkpoints yourself.

The prompt length plus requested output budget must fit every declared context
limit. All three callbacks run sequentially, including at `alpha=0`. If a
callback fails or returns invalid logits, decoding raises `DecodeError` with
the model `role`, zero-based `step`, and original exception as its cause.
There is no successful partial result or automatic unsteered fallback.

The current NVIDIA client receives completed responses and cannot insert this
operation into the provider's generation loop. You need caller-controlled
logits access to use the utility with actual models.

This utility is experimental. It returns token IDs rather than a calibrated
contamination score. Existing MIA/MCS calibration must not be reused unchanged
for adjusted generation. Mathematical correctness alone does not establish
that divergence decoding reduces memorization for any model or dataset.
