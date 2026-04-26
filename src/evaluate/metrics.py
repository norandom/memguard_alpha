class Evaluator:
    def evaluate(self, predictions: list):
        if not predictions:
            return 0.0, 0.0, 0.0, 0.0
            
        correct_count = 0
        raw_conf_sum = 0.0
        mem_conf_sum = 0.0
        
        for p in predictions:
            target = p.get("target_direction", 0)
            pred = p.get("predicted_direction", 0)
            
            if target == pred:
                correct_count += 1
                
            raw_conf_sum += p.get("raw_confidence", 0.0)
            mem_conf_sum += p.get("penalized_confidence", 0.0)
            
        total = len(predictions)
        accuracy = correct_count / total
        
        return accuracy, raw_conf_sum / total, accuracy, mem_conf_sum / total
