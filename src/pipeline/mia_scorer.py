import math

class MIAScorer:
    def calculate_loss(self, logprobs: list) -> float:
        if not logprobs:
            return 0.0
        
        sum_logprob = sum(item.get("logprob", 0.0) for item in logprobs)
        avg_logprob = sum_logprob / len(logprobs)
        return -avg_logprob

    def calculate_mink(self, logprobs: list, k: float = 0.2) -> float:
        if not logprobs:
            return 0.0
            
        probs = [item.get("logprob", 0.0) for item in logprobs]
        probs.sort()
        
        k_count = max(1, int(len(probs) * k))
        lowest_probs = probs[:k_count]
        
        return sum(lowest_probs) / len(lowest_probs)

    def apply_penalty(self, confidence: float, loss_score: float, mink_score: float) -> float:
        penalty_factor = 1.0
        
        if loss_score < 0.5:
            penalty_factor *= 0.5
            
        if mink_score > -0.5:
            penalty_factor *= 0.5
            
        return confidence * penalty_factor
