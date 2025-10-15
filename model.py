#model.py
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
import torch

class Gemma3ImageDescriber():
    """
    Load and configure gemma3 model
    """
    def __init__(self, model_id="google/gemma-3-4b-it", device="cuda:0"):
        self.model_id = model_id
        self.device = device
        
        # Load processor and model
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map=device
        ).cuda().eval()
    
    def describe_frame(self, image_path, prompt=None, max_new_tokens=16):
        if prompt is None:
            prompt = "Describe the image precisely. "
        
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are an open vocabulary detection agent. Output within 10 words. Do not provide additional explanations"}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self.device, dtype=torch.bfloat16)
        
        input_len = inputs["input_ids"].shape[-1]
        
        with torch.inference_mode():
            generation = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False
            )
            generation = generation[0][input_len:]
        
        decoded = self.processor.decode(generation, skip_special_tokens=True)
        print(decoded)
        return decoded

