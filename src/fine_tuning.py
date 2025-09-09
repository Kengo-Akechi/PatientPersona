import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig, 
    get_peft_model, 
    TaskType,
    prepare_model_for_kbit_training
)
from datasets import Dataset
import json
import os
from pathlib import Path
from typing import List, Dict, Callable, Optional
from datetime import datetime

class FineTuner:
    """Handles fine-tuning of models using LoRA"""
    
    def __init__(
        self,
        base_model: str,
        learning_rate: float = 1e-4,
        num_epochs: int = 3,
        batch_size: int = 4,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        target_modules: List[str] = None
    ):
        self.base_model = base_model
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.target_modules = target_modules or ["q_proj", "v_proj"]
        
        # Create directories
        self.models_dir = Path("models/finetuned")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.checkpoints_dir = Path("checkpoints")
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    
    def fine_tune(
        self, 
        training_data: List[Dict], 
        model_name: str,
        progress_callback: Optional[Callable] = None
    ) -> Optional[str]:
        """Fine-tune the model with the provided data"""
        
        try:
            # Load tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(self.base_model)
            
            # Add padding token if not present
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            model = AutoModelForCausalLM.from_pretrained(
                self.base_model,
                torch_dtype=torch.float16,
                device_map="auto" if torch.cuda.is_available() else "cpu"
            )
            
            # Prepare model for training
            if hasattr(model, 'enable_input_require_grads'):
                model.enable_input_require_grads()
            else:
                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)
                model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
            
            # Configure LoRA
            lora_config = LoraConfig(
                r=self.lora_rank,
                lora_alpha=self.lora_alpha,
                target_modules=self.target_modules,
                lora_dropout=0.1,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            
            # Apply LoRA to model
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
            
            # Prepare dataset
            dataset = self._prepare_dataset(training_data, tokenizer)
            
            # Training arguments
            output_dir = self.checkpoints_dir / f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            training_args = TrainingArguments(
                output_dir=str(output_dir),
                num_train_epochs=self.num_epochs,
                per_device_train_batch_size=self.batch_size,
                gradient_accumulation_steps=2,
                warmup_steps=100,
                learning_rate=self.learning_rate,
                fp16=torch.cuda.is_available(),
                logging_steps=10,
                save_strategy="epoch",
                evaluation_strategy="no",
                remove_unused_columns=False,
                dataloader_pin_memory=False,
            )
            
            # Data collator
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=False,
            )
            
            # Custom trainer with progress callback
            trainer = CustomTrainer(
                model=model,
                args=training_args,
                train_dataset=dataset,
                data_collator=data_collator,
                progress_callback=progress_callback,
            )
            
            # Start training
            trainer.train()
            
            # Save the model
            final_model_path = self.models_dir / model_name
            trainer.save_model(str(final_model_path))
            
            # Save tokenizer
            tokenizer.save_pretrained(str(final_model_path))
            
            # Save configuration
            config = {
                "base_model": self.base_model,
                "model_name": model_name,
                "training_params": {
                    "learning_rate": self.learning_rate,
                    "num_epochs": self.num_epochs,
                    "batch_size": self.batch_size,
                    "lora_rank": self.lora_rank,
                    "lora_alpha": self.lora_alpha,
                    "target_modules": self.target_modules
                },
                "training_data_size": len(training_data),
                "created_at": datetime.now().isoformat()
            }
            
            with open(final_model_path / "config.json", 'w') as f:
                json.dump(config, f, indent=2)
            
            return str(final_model_path)
            
        except Exception as e:
            print(f"Error during fine-tuning: {e}")
            return None
    
    def _prepare_dataset(self, training_data: List[Dict], tokenizer) -> Dataset:
        """Prepare the training dataset"""
        
        texts = []
        for conversation in training_data:
            # Extract conversation data
            doctor_text = conversation.get('doctor', '')
            patient_text = conversation.get('patient', '')
            persona = conversation.get('persona', 'calm')
            
            # Create training text with persona context
            training_text = self._format_training_text(doctor_text, patient_text, persona)
            texts.append(training_text)
        
        # Tokenize texts
        tokenized_texts = []
        for text in texts:
            tokens = tokenizer(
                text,
                truncation=True,
                padding=False,
                max_length=512,
                return_tensors="pt"
            )
            tokenized_texts.append({
                "input_ids": tokens["input_ids"].flatten(),
                "attention_mask": tokens["attention_mask"].flatten()
            })
        
        # Create dataset
        dataset = Dataset.from_list(tokenized_texts)
        return dataset
    
    def _format_training_text(self, doctor_text: str, patient_text: str, persona: str) -> str:
        """Format text for training"""
        
        persona_descriptions = {
            "calm": "You are a calm and cooperative patient.",
            "anxious": "You are an anxious and worried patient.", 
            "rude": "You are a rude and impatient patient.",
            "patient": "You are a very patient and understanding patient.",
            "confused": "You are a confused patient who needs repeated explanations.",
            "aggressive": "You are an aggressive and confrontational patient."
        }
        
        persona_desc = persona_descriptions.get(persona, "You are a patient.")
        
        formatted_text = f"""System: {persona_desc}

Doctor: {doctor_text}
Patient: {patient_text}<|endoftext|>"""
        
        return formatted_text


class CustomTrainer(Trainer):
    """Custom trainer with progress callback"""
    
    def __init__(self, progress_callback=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.progress_callback = progress_callback
        self.current_epoch = 0
        self.total_epochs = kwargs.get('args').num_train_epochs
    
    def on_epoch_begin(self, args, state, control, **kwargs):
        self.current_epoch = state.epoch
    
    def on_log(self, args, state, control, **kwargs):
        if self.progress_callback and 'loss' in state.log_history[-1]:
            loss = state.log_history[-1]['loss']
            progress = state.epoch / self.total_epochs
            self.progress_callback(
                epoch=int(state.epoch),
                loss=loss,
                progress=progress
            )
