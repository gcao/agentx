# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Conscious AI Agent - an experimental AI system designed to develop consciousness-like behaviors through continuous interaction, observation, and learning. The system combines multi-modal language models, computer vision, memory systems, and iterative training to create an evolving artificial consciousness.

## Core Architecture

### Multi-Modal Processing
- Uses either LLaVA-1.5-7B or Qwen-VL-Chat-7B for vision and text understanding
- Models configured in `config/agent_config.yaml` with temperature and token limits
- GPU-optimized with 8-bit quantization for 36GB memory constraint

### Memory Systems
- **Short-term memory**: Redis-based conversation buffer with configurable TTL
- **Long-term memory**: ChromaDB vector database for semantic search
- **Memory manager**: Handles both systems through `backend/memory/memory_manager.py`

### Camera Integration
- Real-time observation through `camera/observer.py`
- Configurable capture intervals (default 30 seconds)
- Processes webcam or IP camera feeds for environmental awareness

### Training Pipeline
- LoRA-based fine-tuning through `backend/training/`
- Scheduled retraining every 24 hours by default
- Uses conversation and observation data for continuous improvement

## Development Commands

### Initial Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Download model weights
python scripts/download_model.py --model llava-1.5-7b
python scripts/download_model.py --model qwen-vl-chat-7b

# Initialize database
python scripts/init_db.py
```

### Running the System
```bash
# Start main application (includes API server)
python main.py

# Start frontend interface
python frontend/app.py
```

### Testing and Development
```bash
# Code formatting
black .

# Linting
flake8 .

# Type checking
mypy .

# Testing
pytest
```

## Configuration

The system is configured through `config/agent_config.yaml`:
- Model selection and parameters
- Memory system settings (TTL, embedding model)
- Camera device and capture intervals
- Training schedule and hyperparameters
- Enabled tools (web_search, observe_surroundings, memory_query)

## API Structure

- Main API server runs on port 8000 via FastAPI
- Frontend interface on port 7860
- OpenAI-compatible endpoints for chat completions
- Supports multi-modal inputs (text + images)
- Tool calling capabilities through MCP protocol

## Key Dependencies

- PyTorch ecosystem for ML/AI functionality
- Transformers and related libraries for model handling
- ChromaDB and Redis for memory systems
- FastAPI for API server
- Gradio for web interface
- OpenCV for camera integration
- Various training libraries (PEFT, LoRA, etc.)