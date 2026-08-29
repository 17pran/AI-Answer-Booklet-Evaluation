import ollama


class LLMClient:

    def __init__(self, model="qwen3:8b"):

        self.model = model

        print(
            f"Using LLM: {self.model}"
        )

    def generate(self, prompt):

        response = ollama.chat(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return response["message"]["content"]
