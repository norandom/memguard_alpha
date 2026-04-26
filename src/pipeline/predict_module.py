import dspy
from src.pipeline.signature import FinancialPrediction
from src.pipeline.mia_scorer import MIAScorer
from src.models.nvidia_lm import NvidiaLM
from src.pipeline.math_reasoning import InputMasker

class RecallGuardPredictor(dspy.Module):
    def __init__(self, nvidia_lm: NvidiaLM):
        super().__init__()
        self.nvidia_lm = nvidia_lm
        self.mia_scorer = MIAScorer()
        self.input_masker = InputMasker()

    def forward(self, ticker: str, date: str, context: str):
        # 1. Abstract the raw inputs
        masked_result = self.input_masker(context=context)
        abstracted_context = masked_result.abstracted_context
        
        # 2. Predict with masked inputs
        prompt = f"Given an anonymous Asset X in the Current Period with the following mathematical context:\n{abstracted_context}\n\nPredict the stock direction.\nFormat:\nDirection: [1/-1/0]\nConfidence: [0.0-1.0]"
        content, logprobs = self.nvidia_lm.generate_with_logprobs(prompt)
        
        direction = 0
        confidence = 0.5
        for line in content.split("\n"):
            if "Direction:" in line:
                try:
                    direction = int(line.split("Direction:")[1].strip())
                except:
                    pass
            if "Confidence:" in line:
                try:
                    confidence = float(line.split("Confidence:")[1].strip())
                except:
                    pass
        
        loss_score = self.mia_scorer.calculate_loss(logprobs)
        mink_score = self.mia_scorer.calculate_mink(logprobs)
        
        penalized_confidence = self.mia_scorer.apply_penalty(confidence, loss_score, mink_score)
        
        return dspy.Prediction(
            direction=direction,
            raw_confidence=confidence,
            penalized_confidence=penalized_confidence,
            loss_score=loss_score,
            mink_score=mink_score
        )
