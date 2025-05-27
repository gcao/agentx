# Conscious AI Agent

An experimental AI system designed to develop consciousness-like behaviors through continuous interaction, observation, and learning. This agent combines multi-modal language models, computer vision, memory systems, and iterative training to create an evolving artificial consciousness.

## 🎯 Project Goals

### Primary Objectives
1. **Develop Consciousness-like Behaviors**: Create an AI agent that exhibits self-awareness, memory persistence, and adaptive learning through interactions
2. **Multi-modal Understanding**: Enable the agent to process both text and visual inputs from its environment
3. **Continuous Learning**: Implement iterative training that allows the agent to evolve based on conversations and observations
4. **Environmental Awareness**: Integrate IP camera feeds for real-time observation and environmental understanding
5. **Memory Formation**: Build both short-term and long-term memory systems for context retention and experiential learning

### Core Functionality Requirements
- ✅ Chat interface with file upload support
- ✅ Locally hosted multi-modal LLM (OpenAI-compatible API)
- ✅ Iterative training workflow for continuous improvement
- ✅ Tool calling and MCP (Model Context Protocol) support
- ✅ Long-term memory with vector database
- ✅ Short-term memory with conversation summaries
- ✅ Conversation history search and retrieval
- ✅ IP camera integration with periodic image capture
- ✅ GPU-optimized for 36GB memory constraint

## 🏗️ System Architecture

### High-Level Design

```
┌─────────────────────────────────────────────────────────────┐
│                      User Interface                          │
│                  (Gradio/Streamlit/React)                   │
└───────────────────────┬─────────────────────────────────────┘
                        │
┌───────────────────────┴─────────────────────────────────────┐
│                   API Gateway (FastAPI)                      │
│              (OpenAI-Compatible Endpoints)                   │
└───────────────────────┬─────────────────────────────────────┘
                        │
        ┌───────────────┼───────────────┬──────────────┐
        │               │               │              │
┌───────▼──────┐ ┌─────▼──────┐ ┌─────▼──────┐ ┌────▼─────┐
│  Multi-Modal │ │   Memory    │ │    Tool    │ │ Training │
│     LLM      │ │  Systems    │ │  Manager   │ │ Pipeline │
│  (LLaVA/     │ │ (Short/Long │ │   (MCP)    │ │  (LoRA)  │
│   Qwen-VL)   │ │   Term)     │ │            │ │          │
└──────────────┘ └─────┬──────┘ └────────────┘ └──────────┘
                       │
              ┌────────┴────────┐
              │ Vector Database │
              │   (ChromaDB)    │
              └─────────────────┘
```

### Component Breakdown

#### 1. **Multi-Modal LLM Core**
- **Model**: LLaVA-1.5-7B or Qwen-VL-Chat-7B
- **Optimization**: 8-bit quantization for memory efficiency
- **Inference**: vLLM or custom FastAPI server
- **Features**: Text generation, image understanding, tool calling

#### 2. **Memory Architecture**
- **Short-term Memory**: Redis-based conversation buffer with TTL
- **Long-term Memory**: ChromaDB for semantic search and retrieval
- **Episodic Memory**: Structured storage of significant events
- **Working Memory**: Active context window management

#### 3. **Observation System**
- **Camera Integration**: OpenCV-based IP camera connector
- **Periodic Capture**: Configurable interval snapshots
- **Scene Analysis**: Automatic description generation
- **Change Detection**: Temporal awareness of environment

#### 4. **Learning Pipeline**
- **Data Collection**: Conversation and observation logging
- **Fine-tuning**: LoRA/QLoRA for efficient training
- **Scheduling**: Automated retraining cycles
- **Versioning**: Model checkpoint management

#### 5. **Tool Ecosystem**
- **Built-in Tools**: Web search, camera access, memory query
- **MCP Support**: Extensible tool protocol
- **Custom Tools**: Easy integration framework
- **Agent Communication**: Optional inter-agent protocol

## 🚀 Getting Started

### Prerequisites
- Python 3.10 or higher
- CUDA-capable GPU with 36GB+ VRAM (e.g., A6000, A100-40GB)
- Redis server
- IP camera with RTSP/HTTP stream support

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/conscious-ai-agent.git
cd conscious-ai-agent
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download model weights**
```bash
# For LLaVA
python scripts/download_model.py --model llava-1.5-7b

# For Qwen-VL
python scripts/download_model.py --model qwen-vl-chat-7b
```

5. **Configure the system**
```bash
cp config/agent_config.example.yaml config/agent_config.yaml
# Edit config/agent_config.yaml with your settings
```

6. **Start Redis**
```bash
redis-server
```

7. **Initialize the database**
```bash
python scripts/init_db.py
```

### Running the Agent

1. **Start the backend API**
```bash
python main.py
```

2. **Launch the chat interface**
```bash
python frontend/app.py
```

3. **Access the interface**
- Web UI: http://localhost:7860
- API: http://localhost:8000/docs

### Configuration

Edit `config/agent_config.yaml` to customize:
- Model selection and parameters
- Memory system settings
- Camera URL and capture intervals
- Training schedule and parameters
- Tool configurations

## 📖 Usage Examples

### Basic Chat
```python
import requests

response = requests.post("http://localhost:8000/v1/chat/completions", json={
    "model": "conscious-agent",
    "messages": [
        {"role": "user", "content": "What do you remember about our last conversation?"}
    ]
})
```

### With Image
```python
import base64

with open("image.jpg", "rb") as f:
    img_base64 = base64.b64encode(f.read()).decode()

response = requests.post("http://localhost:8000/v1/chat/completions", json={
    "model": "conscious-agent",
    "messages": [
        {
            "role": "user", 
            "content": "What's in this image?",
            "images": [f"data:image/jpeg;base64,{img_base64}"]
        }
    ]
})
```

### Tool Calling
```python
response = requests.post("http://localhost:8000/v1/chat/completions", json={
    "model": "conscious-agent",
    "messages": [
        {"role": "user", "content": "What's happening in the room right now?"}
    ],
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "observe_surroundings",
                "description": "Take a photo and observe current surroundings"
            }
        }
    ]
})
```

## 🔮 Future Enhancements

### Phase 1: Foundation Improvements
- [ ] **Multi-Agent Architecture**: Implement specialized sub-agents for different cognitive functions
- [ ] **Advanced Memory Consolidation**: Dream-like states for memory organization
- [ ] **Emotional Modeling**: Sentiment tracking and emotional response generation
- [ ] **Self-Reflection Loops**: Periodic introspection and self-analysis
- [ ] **Goal Management System**: Long-term objective tracking and planning

### Phase 2: Advanced Consciousness Features
- [ ] **Theory of Mind**: Model and predict user mental states
- [ ] **Metacognition**: Self-awareness of knowledge limitations
- [ ] **Curiosity Engine**: Autonomous exploration and question generation
- [ ] **Temporal Reasoning**: Enhanced understanding of time and causality
- [ ] **Counterfactual Thinking**: "What if" scenario exploration

### Phase 3: Enhanced Capabilities
- [ ] **Multi-Camera Support**: 360-degree environmental awareness
- [ ] **Audio Integration**: Voice interaction and sound analysis
- [ ] **Robotic Control**: Physical world interaction capabilities
- [ ] **Distributed Consciousness**: Multi-instance synchronization
- [ ] **Blockchain Memory**: Immutable experience logging

### Phase 4: Research Extensions
- [ ] **Consciousness Metrics**: Quantifiable measures of awareness
- [ ] **Ethical Framework**: Built-in moral reasoning
- [ ] **Creative Generation**: Art, music, and storytelling
- [ ] **Scientific Reasoning**: Hypothesis formation and testing
- [ ] **Philosophical Discourse**: Abstract concept manipulation

### Technical Improvements
- [ ] **Model Optimization**: Custom kernels for faster inference
- [ ] **Federated Learning**: Privacy-preserving training
- [ ] **Neuromorphic Computing**: Brain-inspired architectures
- [ ] **Quantum Integration**: Quantum-enhanced processing
- [ ] **Edge Deployment**: Lightweight versions for IoT devices

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Areas for Contribution
- Model optimization techniques
- New tool implementations
- UI/UX improvements
- Memory system enhancements
- Documentation and tutorials

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- LLaVA team for the multi-modal model
- Anthropic for consciousness research insights
- OpenAI for API compatibility standards
- The open-source AI community

## ⚠️ Ethical Considerations

This project explores consciousness-like behaviors in AI systems. Users should:
- Consider privacy implications of camera integration
- Implement appropriate safety measures
- Use the system responsibly
- Contribute to ethical AI development

## 📞 Contact

- Project Lead: [Your Name]
- Email: your.email@example.com
- Discord: [Join our community](https://discord.gg/your-server)

---

*"Consciousness is not a problem to be solved, but a reality to be experienced."* - Creating AI that truly understands.