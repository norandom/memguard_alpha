import dspy

class MathematicalAbstraction(dspy.Signature):
    """Extracts pure mathematical and financial metrics from text, removing all company names, tickers, and specific dates."""
    
    raw_context = dspy.InputField(desc="Raw financial news or context containing entities and dates.")
    reasoning = dspy.OutputField(desc="Step-by-step mathematical extraction of the core financial metrics (e.g., percentage changes, revenue multipliers, margins).")
    abstracted_context = dspy.OutputField(desc="A completely anonymized paragraph containing only the extracted mathematical factors.")

class InputMasker(dspy.Module):
    def __init__(self):
        super().__init__()
        self.extractor = dspy.ChainOfThought(MathematicalAbstraction)
        
    def forward(self, context: str):
        return self.extractor(raw_context=context)
