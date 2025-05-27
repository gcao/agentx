import gradio as gr
import requests
import base64

class ChatInterface:
    def __init__(self, api_url="http://localhost:8000"):
        self.api_url = api_url
        self.conversation_history = []
    
    def chat(self, message, image=None):
        # Prepare message
        msg = {"role": "user", "content": message}
        
        # Handle image upload
        if image is not None:
            with open(image.name, "rb") as f:
                img_data = base64.b64encode(f.read()).decode()
                msg["images"] = [f"data:image/jpeg;base64,{img_data}"]
        
        self.conversation_history.append(msg)
        
        # Send to API
        response = requests.post(
            f"{self.api_url}/v1/chat/completions",
            json={
                "model": "conscious-agent",
                "messages": self.conversation_history[-10:],  # Last 10 messages
                "temperature": 0.7
            }
        )
        
        if response.status_code == 200:
            result = response.json()
            assistant_msg = result["choices"][0]["message"]["content"]
            self.conversation_history.append({
                "role": "assistant",
                "content": assistant_msg
            })
            return assistant_msg
        else:
            return "Error: Could not get response"
    
    def launch(self):
        interface = gr.Interface(
            fn=self.chat,
            inputs=[
                gr.Textbox(placeholder="Enter your message..."),
                gr.File(label="Upload Image (optional)")
            ],
            outputs=gr.Textbox(),
            title="Conscious AI Agent",
            description="Chat with your AI agent that can see and learn"
        )
        interface.launch()