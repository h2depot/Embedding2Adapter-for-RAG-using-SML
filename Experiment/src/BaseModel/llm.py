import ollama

from ..utils.config import get_generating_model, get_generating_options, get_ollama_host


class LLM:
    def __init__(self):
        self.model_name = get_generating_model()
        self.client = ollama.Client(host=get_ollama_host())
        self.generation_options = get_generating_options()
        print("LLM initialized")

    def generate(self, prompt: str):
        response = self.client.generate(
            model=self.model_name,
            prompt=prompt,
            options=self.generation_options,
        )
        return response["response"]
