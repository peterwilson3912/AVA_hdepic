import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from llms.BaseModel import BaseVideoModel


class Gemma4VL(BaseVideoModel):
    def __init__(self, model_type="google/gemma-4-26B-A4B-it", tp=1):
        """
        Initialize the Gemma4VL model using HuggingFace transformers.

        Args:
            model_type (str): HuggingFace model ID.
            tp (int): Number of GPUs for the model (starting from GPU 0).
        """
        # restrict model to first `tp` GPUs, leaving remaining GPUs free for other models
        max_memory = {i: "45GiB" for i in range(tp)}
        for i in range(tp, torch.cuda.device_count()):
            max_memory[i] = "0GiB"

        self.processor = AutoProcessor.from_pretrained(model_type)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_type,
            dtype=torch.bfloat16,
            device_map="auto",
            max_memory=max_memory,
        )
        self.model.eval()

    def _build_messages(self, inputs):
        if "video" in inputs:
            content = [{"type": "image", "image": img} for img in inputs["video"]]
            content.append({"type": "text", "text": inputs["text"]})
        else:
            content = [{"type": "text", "text": inputs["text"]}]
        return [{"role": "user", "content": content}]

    def _generate_one(self, messages, max_new_tokens, temperature):
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        images = [
            item["image"]
            for msg in messages
            for item in msg["content"]
            if item["type"] == "image"
        ]
        inputs = self.processor(
            text=text,
            images=images if images else None,
            return_tensors="pt",
        ).to(self.model.device)

        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=0.95,
                top_k=64,
                do_sample=True,
            )

        return self.processor.decode(output[0][input_len:], skip_special_tokens=True)

    def _generate(self, messages_list, max_new_tokens, temperature):
        return [self._generate_one(messages, max_new_tokens, temperature) for messages in messages_list]

    def generate_response(self, inputs, max_new_tokens=512, temperature=1.0):
        """
        Generate a response based on the inputs.

        Args:
            inputs (dict): Input data containing text and optionally video frames.
            {
                "text": str,
                "video": list[Image.Image]  (optional)
            }

        Returns:
            str: Generated response.
        """
        assert "text" in inputs, "Please provide a text prompt."
        messages = self._build_messages(inputs)
        return self._generate([messages], max_new_tokens, temperature)[0]

    def batch_generate_response(self, batch_inputs, max_batch_size=8, max_new_tokens=512, temperature=1.0):
        all_responses = []
        for i in range(0, len(batch_inputs), max_batch_size):
            batch = batch_inputs[i:i + max_batch_size]
            messages_list = [self._build_messages(inp) for inp in batch]
            all_responses.extend(self._generate(messages_list, max_new_tokens, temperature))
        return all_responses
