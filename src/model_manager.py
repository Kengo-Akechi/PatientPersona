import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import os
from pathlib import Path
import json
from typing import Optional, Dict, Any

class ModelManager:
    """Manages fine-tuned models and inference"""
    
    def __init__(self):
        self.models_dir = Path("models/finetuned")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.loaded_model = None
        self.loaded_tokenizer = None
        self.current_model_name = None
    
    def load_model(self, model_name: str) -> bool:
        """Load a fine-tuned model"""
        try:
            model_path = self.models_dir / model_name
            
            if not model_path.exists():
                raise FileNotFoundError(f"Model {model_name} not found")
            
            # Load model config
            config_path = model_path / "config.json"
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            base_model = config['base_model']
            
            # Load base model and tokenizer
            self.loaded_tokenizer = AutoTokenizer.from_pretrained(base_model)
            base_model_obj = AutoModelForCausalLM.from_pretrained(
                base_model,
                torch_dtype=torch.float16,
                device_map="auto" if torch.cuda.is_available() else "cpu"
            )
            
            # Load LoRA adapter
            self.loaded_model = PeftModel.from_pretrained(
                base_model_obj,
                str(model_path)
            )
            
            self.current_model_name = model_name
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def generate_response(
        self, 
        doctor_input: str, 
        persona: str, 
        model_name: str,
        is_finetuned: bool = True,
        max_length: int = 150,
        temperature: float = 0.7
    ) -> str:
        """Generate patient response using fine-tuned model"""
        
        if is_finetuned and (not self.loaded_model or self.current_model_name != model_name):
            if not self.load_model(model_name):
                raise Exception(f"Failed to load model: {model_name}")
        
        # Create prompt with persona and context
        prompt = self._create_prompt(doctor_input, persona)
        
        # Tokenize input
        inputs = self.loaded_tokenizer.encode(prompt, return_tensors="pt")
        
        # Generate response
        with torch.no_grad():
            outputs = self.loaded_model.generate(
                inputs,
                max_length=max_length,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.loaded_tokenizer.eos_token_id,
                num_return_sequences=1
            )
        
        # Decode and extract response
        full_response = self.loaded_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the patient response part
        patient_response = self._extract_patient_response(full_response, prompt)
        
        return patient_response
    
    def _create_prompt(self, doctor_input: str, persona: str) -> str:
        """Create a formatted prompt for the model"""
        
        persona_descriptions = {
            "calm": "You are a calm and cooperative patient. You speak thoughtfully and ask relevant questions.",
            "anxious": "You are an anxious patient. You are worried, speak quickly, and seek reassurance.",
            "rude": "You are a rude and impatient patient. You are dismissive and question the doctor's authority.",
            "patient": "You are a very patient and understanding patient. You listen carefully and express gratitude.",
            "confused": "You are a confused patient. You have difficulty understanding and need repeated explanations.",
            "aggressive": "You are an aggressive patient. You are confrontational and defensive."
        }
        
        persona_desc = persona_descriptions.get(persona, "You are a patient speaking with a doctor.")
        
        prompt = f"""System: {persona_desc}

Doctor: {doctor_input}
Patient:"""
        
        return prompt
    
    def _extract_patient_response(self, full_response: str, prompt: str) -> str:
        """Extract the patient response from the full generated text"""
        
        # Remove the prompt from the response
        if prompt in full_response:
            response = full_response.replace(prompt, "").strip()
        else:
            response = full_response.strip()
        
        # Clean up the response
        lines = response.split('\n')
        patient_lines = []
        
        for line in lines:
            line = line.strip()
            if line and not line.startswith("Doctor:") and not line.startswith("System:"):
                patient_lines.append(line)
        
        return ' '.join(patient_lines) if patient_lines else response
    
    def list_models(self) -> list:
        """List all available fine-tuned models"""
        if not self.models_dir.exists():
            return []
        
        models = []
        for model_dir in self.models_dir.iterdir():
            if model_dir.is_dir() and (model_dir / "config.json").exists():
                models.append(model_dir.name)
        
        return models
    
    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific model"""
        model_path = self.models_dir / model_name
        config_path = model_path / "config.json"
        
        if not config_path.exists():
            return None
        
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def delete_model(self, model_name: str) -> bool:
        """Delete a fine-tuned model"""
        try:
            model_path = self.models_dir / model_name
            if model_path.exists():
                import shutil
                shutil.rmtree(model_path)
                return True
            return False
        except Exception as e:
            print(f"Error deleting model: {e}")
            return False
