import requests

class NvidiaLM:
    def __init__(self, api_key: str, model: str = "nvidia/nemotron-3-super-120b-a12b"):
        self.api_key = api_key
        self.model = model
        self.api_base = "https://integrate.api.nvidia.com/v1/chat/completions"

    def generate_with_logprobs(self, prompt: str):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "logprobs": True,
            "top_logprobs": 20
        }
        
        response = requests.post(self.api_base, headers=headers, json=payload)
        response.raise_for_status()
        
        data = response.json()
        choice = data["choices"][0]
        content = choice["message"]["content"]
        logprobs = choice.get("logprobs", {}).get("content", [])
        
        return content, logprobs

    def __call__(self, prompt, **kwargs):
        content, _ = self.generate_with_logprobs(prompt)
        return [content]
