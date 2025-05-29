import torch
from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig

class MultiModalModel:
    def __init__(self, model_name="models/qwen-vl-chat-7b"):
        self.device = torch.device("cpu")  # Force CPU usage
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map="cpu",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    
    def generate(self, text, image=None, max_tokens=256):
        if image:
            inputs = self.processor(text, image, return_tensors="pt").to(self.device)
        else:
            inputs = self.processor(text, return_tensors="pt").to(self.device)
        
        outputs = self.model.generate(**inputs, max_new_tokens=max_tokens)
        return self.processor.decode(outputs[0], skip_special_tokens=True)
