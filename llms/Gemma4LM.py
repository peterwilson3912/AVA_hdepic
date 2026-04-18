import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from llms.BaseModel import BaseLanguageModel


class Gemma4LM(BaseLanguageModel):
    def __init__(self, model_type="google/gemma-4-26B-A4B-it", tp=1):
        """
        Initialize the Gemma4LM model using HuggingFace transformers (text-only).

        Args:
            model_type (str): HuggingFace model ID.
            tp (int): Unused; device_map="auto" handles multi-GPU placement.
        """
        self.processor = AutoProcessor.from_pretrained(model_type)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_type,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="eager",
        )
        self.model.eval()

    def _generate(self, texts, max_new_tokens, temperature):
        inputs = self.processor(
            text=texts,
            return_tensors="pt",
            padding=True,
        ).to(self.model.device)

        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=0.95,
                top_k=64,
                do_sample=True,
            )

        return [
            self.processor.decode(out[input_len:], skip_special_tokens=True)
            for out in outputs
        ]

    def generate_response(self, inputs, max_new_tokens=512, temperature=1.0):
        """
        Generate a response based on the inputs.

        Args:
            inputs (dict): Input data containing text.
            {
                "text": str,
            }

        Returns:
            str: Generated response.
        """
        assert "text" in inputs, "Please provide a text prompt."
        messages = [{"role": "user", "content": inputs["text"]}]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        return self._generate([text], max_new_tokens, temperature)[0]

    def batch_generate_response(self, batch_inputs, max_batch_size=8, max_new_tokens=512, temperature=1.0):
        all_responses = []
        for i in range(0, len(batch_inputs), max_batch_size):
            batch = batch_inputs[i:i + max_batch_size]
            texts = [
                self.processor.apply_chat_template(
                    [{"role": "user", "content": inp["text"]}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                for inp in batch
            ]
            all_responses.extend(self._generate(texts, max_new_tokens, temperature))
        return all_responses
