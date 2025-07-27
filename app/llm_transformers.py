import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class TransformersLLM:
    def __init__(
        self,
        model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        device_map: str = "auto",
        torch_dtype: torch.dtype = torch.float16,
        max_new_tokens: int = 512,
        temperature: float = 0.2,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=os.getenv("HUGGINGFACE_TOKEN"))
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            torch_dtype=torch_dtype,
            token=os.getenv("HUGGINGFACE_TOKEN"),
        )
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    def __call__(self, prompt: str) -> str:
        """Generate a response for the given prompt."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            do_sample=True,
        )
        gen_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
        return self.tokenizer.decode(gen_tokens, skip_special_tokens=True)
