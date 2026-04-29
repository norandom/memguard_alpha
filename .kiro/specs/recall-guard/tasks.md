# Implementation Plan

## Tasks

- [ ] 1. Set up the `src` directory and skeleton files.
  - _Requirements: 1_
- [ ] 2. Implement Look-Ahead-Bench Dataset Loader
  - Create `src/dataset/lookahead_loader.py` to parse historical data into DSPy Examples.
  - _Requirements: 1_
- [ ] 3. Implement Custom DSPy LM for NVIDIA API
  - Create `src/models/nvidia_lm.py`.
  - Ensure the API requests `include_logprobs=true`.
  - Extract the text and logprobs from the response.
  - _Requirements: 2_
- [ ] 4. Implement MIA Scorer
  - Create `src/pipeline/mia_scorer.py`.
  - Implement `calculate_loss(logprobs)`.
  - Implement `calculate_mink(logprobs, k=0.2)`.
  - Implement `apply_penalty(confidence, loss_score, mink_score)`.
  - _Requirements: 4, 5_
- [ ] 5. Implement DSPy Predict Module and Signatures
  - Create `src/pipeline/signature.py` and `src/pipeline/predict_module.py`.
  - Run the NVIDIA LM and pass results through the MIA Scorer.
  - _Requirements: 3, 5_
- [ ] 6. Build the Evaluation Script
  - Create `src/evaluate/metrics.py`.
  - Compute the accuracy and confidence for "Raw" (ignoring penalty).
  - Compute the accuracy and confidence for "MemGuard" (applying penalty).
  - Compare the gap.
  - _Requirements: 3, 5_
