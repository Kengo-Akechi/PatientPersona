import os
from pathlib import Path
from typing import Dict, Any

class Config:
    """Configuration management for the application"""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent.parent
        self.data_dir = self.base_dir / "data"
        self.models_dir = self.base_dir / "models"
        self.uploads_dir = self.base_dir / "uploads"
        
        # Create directories
        self.data_dir.mkdir(exist_ok=True)
        self.models_dir.mkdir(exist_ok=True)
        self.uploads_dir.mkdir(exist_ok=True)
        (self.models_dir / "finetuned").mkdir(exist_ok=True)
    
    @property
    def ollama_config(self) -> Dict[str, Any]:
        """Ollama configuration"""
        return {
            "base_url": os.getenv("OLLAMA_URL", "http://localhost:11434"),
            "timeout": int(os.getenv("OLLAMA_TIMEOUT", "30")),
            "max_retries": int(os.getenv("OLLAMA_MAX_RETRIES", "3"))
        }
    
    @property
    def training_config(self) -> Dict[str, Any]:
        """Training configuration defaults"""
        return {
            "default_learning_rate": float(os.getenv("DEFAULT_LEARNING_RATE", "1e-4")),
            "default_batch_size": int(os.getenv("DEFAULT_BATCH_SIZE", "4")),
            "default_epochs": int(os.getenv("DEFAULT_EPOCHS", "3")),
            "max_sequence_length": int(os.getenv("MAX_SEQUENCE_LENGTH", "512")),
            "lora_rank": int(os.getenv("LORA_RANK", "16")),
            "lora_alpha": int(os.getenv("LORA_ALPHA", "32"))
        }
    
    @property
    def evaluation_config(self) -> Dict[str, Any]:
        """Evaluation configuration"""
        return {
            "max_test_samples": int(os.getenv("MAX_TEST_SAMPLES", "100")),
            "evaluation_timeout": int(os.getenv("EVALUATION_TIMEOUT", "60")),
            "quality_threshold": float(os.getenv("QUALITY_THRESHOLD", "0.7")),
            "persona_threshold": float(os.getenv("PERSONA_THRESHOLD", "0.6"))
        }
    
    @property
    def ui_config(self) -> Dict[str, Any]:
        """UI configuration"""
        return {
            "max_upload_size": int(os.getenv("MAX_UPLOAD_SIZE", "10485760")),  # 10MB
            "supported_formats": ["csv", "json", "txt"],
            "default_personas": ["calm", "anxious", "rude", "patient", "confused", "aggressive"]
        }
    
    def get_model_path(self, model_name: str) -> Path:
        """Get path for a fine-tuned model"""
        return self.models_dir / "finetuned" / model_name
    
    def get_upload_path(self, filename: str) -> Path:
        """Get path for uploaded file"""
        return self.uploads_dir / filename
    
    def get_data_path(self, filename: str) -> Path:
        """Get path for data file"""
        return self.data_dir / filename
