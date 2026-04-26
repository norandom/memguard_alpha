import dspy

class FinancialPrediction(dspy.Signature):
    """Predict the stock direction based on historical context."""
    ticker = dspy.InputField(desc="The stock ticker symbol")
    date = dspy.InputField(desc="The date of the prediction")
    context = dspy.InputField(desc="The financial context including news and price action")
    
    direction = dspy.OutputField(desc="Direction prediction: 1 for Bullish, -1 for Bearish, 0 for Neutral")
    confidence = dspy.OutputField(desc="Confidence score between 0.0 and 1.0")
