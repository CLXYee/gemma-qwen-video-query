#model.py
import torch
from transformers import AutoProcessor, Gemma3ForConditionalGeneration, AutoModelForImageTextToText

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

#
class QwenImageDescriber():
    def __init__(self, model_id="Qwen/Qwen2.5-VL-7B-Instruct", device="cuda:0"):
        self.model_id = model_id
        self.device = device
        
        # Load processor and model
        #self.model = AutoModelForImageTextToText.from_pretrained(
        self.model=AutoModelForImageTextToText.from_pretrained(
            self.model_id, torch_dtype=torch.bfloat16, device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.processor.tokenizer.padding_side = "left"

        pad_id = self.processor.image_token_id or self.processor.video_token_id
        eos_id = self.processor.video_token_id

        self.model.generation_config.image_token_id = pad_id
        self.model.generation_config.video_token_id = eos_id

    def describe_frame(self, image_path, prompt=None, max_new_tokens=16):
        system_prompt = "You are an open vocabulary detection agent. Output within 10 words. Do not provide additional explanations. "
        if prompt is None:
            prompt = "Describe the image precisely"
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
                    },
                    {"type": "text", "text": system_prompt+prompt},
                ],
            }
        ]

        # Preparation for inference
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        )
        inputs = inputs.to(self.model.device)

        # Inference: Generation of the output
        generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].split("addCriterion")[1]
        
        print(output_text)
        return output_text