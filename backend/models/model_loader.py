import torch
from io import BytesIO
import base64
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

class MultiModalModel:
    def __init__(self, model_name="models/qwen_vl_chat_7b"):
        self.device = torch.device("cpu")  # Force CPU usage
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map="cpu",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        import sys
        from pathlib import Path
        sys.path.append(str(Path(__file__).parent.parent.parent))
        from models.qwen_vl_chat_7b.visual import VisionTransformer
        self.processor = VisionTransformer(
            image_size=224,
            patch_size=14,
            width=1664,
            layers=48,
            heads=16,
            mlp_ratio=4.9231,
            output_dim=4096
        )
    
    def generate(self, text, image=None, max_tokens=256):
        # Tokenize text
        text_inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        
        if image:
            # Convert image to base64 string
            from io import BytesIO
            import base64
            buffered = BytesIO()
            image.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
            
            # Create input with image tokens
            input_text = f"<image>{img_str}</image>\n{text}"
            inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
        else:
            inputs = text_inputs
        
        outputs = self.model.generate(**inputs, max_new_tokens=max_tokens)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def chat(self, tokenizer, query, history=None, images=None, max_length=8192, **kwargs):
        """Wrapper for the model's chat interface with length handling"""
        try:
            if images and len(images) > 0:
                # Process image
                buffered = BytesIO()
                images[0].save(buffered, format="JPEG")
                img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
                
                # Create input with image tokens
                input_text = f"<image>{img_str}</image>\n{query}"
                inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True, max_length=max_length).to(self.device)
                
                max_tokens = max(1, min(256, max_length - inputs.input_ids.shape[1]))
                outputs = self.model.generate(**inputs, max_new_tokens=max_tokens)
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                return response, history or []
            
            else:
                # Text-only chat with truncation
                inputs = self.tokenizer(query, return_tensors="pt", truncation=True, max_length=max_length).to(self.device)
                outputs = self.model.generate(**inputs, max_new_tokens=min(256, max_length - inputs.input_ids.shape[1]))
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                return response, history or []
                
        except Exception as e:
            print(f"Chat error: {str(e)}")
            return "Sorry, I encountered an error processing your request.", history or []
