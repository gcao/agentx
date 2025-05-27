from peft import LoraConfig, get_peft_model, TaskType
from transformers import TrainingArguments, Trainer
import torch

class ModelFineTuner:
    def __init__(self, base_model, training_data_path):
        self.base_model = base_model
        self.training_data_path = training_data_path
        
    def prepare_lora_model(self):
        # LoRA configuration for efficient fine-tuning
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=16,  # LoRA rank
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
        )
        
        self.model = get_peft_model(self.base_model.model, peft_config)
        self.model.print_trainable_parameters()
    
    def train(self, num_epochs=3):
        # Load training data from conversations
        dataset = self.load_conversation_data()
        
        training_args = TrainingArguments(
            output_dir="./results",
            num_train_epochs=num_epochs,
            per_device_train_batch_size=1,  # Small batch for 36GB GPU
            gradient_accumulation_steps=8,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir="./logs",
            save_strategy="epoch",
            evaluation_strategy="epoch",
            fp16=True,  # Mixed precision training
            gradient_checkpointing=True,  # Memory optimization
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            tokenizer=self.base_model.processor.tokenizer,
        )
        
        trainer.train()
        
    def load_conversation_data(self):
        # Load and format conversation history for training
        conversations = memory_manager.get_training_data()
        # Format as instruction-following dataset
        return self.format_for_training(conversations)