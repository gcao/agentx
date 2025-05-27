import torch
from transformers import LlavaForConditionalGeneration, AutoProcessor

class MultiModalModel:
    def __init__(self, model_name="llava-hf/llava-1.5-7b-hf"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            load_in_8bit=True  # For memory efficiency
        )
        self.processor = AutoProcessor.from_pretrained(model_name)
    
    def generate(self, text, image=None, max_tokens=256):
        if image:
            inputs = self.processor(text, image, return_tensors="pt").to(self.device)
        else:
            inputs = self.processor(text, return_tensors="pt").to(self.device)
        
        outputs = self.model.generate(**inputs, max_new_tokens=max_tokens)
        return self.processor.decode(outputs[0], skip_special_tokens=True)