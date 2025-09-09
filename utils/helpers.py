import streamlit as st
import pandas as pd
import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import hashlib

def save_uploaded_file(uploaded_file) -> str:
    """Save uploaded file and return path"""
    
    # Create uploads directory
    uploads_dir = Path("uploads")
    uploads_dir.mkdir(exist_ok=True)
    
    # Generate unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{uploaded_file.name}"
    file_path = uploads_dir / filename
    
    # Save file
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    return str(file_path)

def format_conversation(doctor_text: str, patient_text: str, persona: str) -> str:
    """Format conversation for display"""
    
    formatted = f"""
**Doctor:** {doctor_text}

**Patient ({persona}):** {patient_text}
"""
    return formatted

def validate_text_input(text: str, min_length: int = 1, max_length: int = 1000) -> bool:
    """Validate text input length and content"""
    
    if not text or not text.strip():
        return False
    
    text_length = len(text.strip())
    return min_length <= text_length <= max_length

def clean_text(text: str) -> str:
    """Clean and normalize text"""
    
    if not text:
        return ""
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove non-printable characters
    text = ''.join(char for char in text if char.isprintable() or char.isspace())
    
    return text

def calculate_text_statistics(text: str) -> Dict[str, Any]:
    """Calculate basic text statistics"""
    
    if not text:
        return {"words": 0, "characters": 0, "sentences": 0}
    
    words = len(text.split())
    characters = len(text)
    sentences = len(re.findall(r'[.!?]+', text))
    
    return {
        "words": words,
        "characters": characters,
        "sentences": sentences,
        "avg_word_length": characters / words if words > 0 else 0
    }

def create_download_link(data: Any, filename: str, link_text: str) -> str:
    """Create a download link for data"""
    
    if isinstance(data, pd.DataFrame):
        csv = data.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{link_text}</a>'
    elif isinstance(data, (dict, list)):
        json_str = json.dumps(data, indent=2)
        b64 = base64.b64encode(json_str.encode()).decode()
        href = f'<a href="data:file/json;base64,{b64}" download="{filename}">{link_text}</a>'
    else:
        # Text data
        b64 = base64.b64encode(str(data).encode()).decode()
        href = f'<a href="data:file/txt;base64,{b64}" download="{filename}">{link_text}</a>'
    
    return href

def format_timestamp(timestamp: str) -> str:
    """Format timestamp for display"""
    
    try:
        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except:
        return timestamp

def generate_model_id(base_model: str, training_params: Dict) -> str:
    """Generate unique model ID based on parameters"""
    
    # Create hash of parameters
    param_str = json.dumps(training_params, sort_keys=True)
    param_hash = hashlib.md5(param_str.encode()).hexdigest()[:8]
    
    # Clean base model name
    base_clean = re.sub(r'[^a-zA-Z0-9]', '_', base_model)
    
    return f"{base_clean}_{param_hash}"

def validate_persona(persona: str, valid_personas: List[str]) -> bool:
    """Validate persona against list of valid personas"""
    
    return persona.lower() in [p.lower() for p in valid_personas]

def extract_medical_entities(text: str) -> List[str]:
    """Extract medical entities from text (simple keyword-based)"""
    
    medical_keywords = [
        "pain", "ache", "hurt", "symptom", "fever", "headache", "nausea",
        "medication", "medicine", "pill", "treatment", "therapy", "surgery",
        "diagnosis", "condition", "disease", "illness", "infection",
        "doctor", "physician", "nurse", "hospital", "clinic", "appointment"
    ]
    
    text_lower = text.lower()
    found_entities = []
    
    for keyword in medical_keywords:
        if keyword in text_lower:
            found_entities.append(keyword)
    
    return list(set(found_entities))  # Remove duplicates

def calculate_readability_score(text: str) -> float:
    """Calculate simple readability score (0-1, higher is more readable)"""
    
    if not text:
        return 0.0
    
    words = text.split()
    sentences = re.findall(r'[.!?]+', text)
    
    if not words or not sentences:
        return 0.5
    
    avg_words_per_sentence = len(words) / len(sentences)
    avg_word_length = sum(len(word) for word in words) / len(words)
    
    # Simple readability calculation (normalized)
    # Shorter sentences and words are more readable
    sentence_score = max(0, 1 - (avg_words_per_sentence - 10) / 20)  # Optimal around 10 words
    word_score = max(0, 1 - (avg_word_length - 5) / 10)  # Optimal around 5 characters
    
    return (sentence_score + word_score) / 2

def format_error_message(error: Exception) -> str:
    """Format error message for user display"""
    
    error_msg = str(error)
    
    # Common error patterns and user-friendly messages
    error_mappings = {
        "connection": "Unable to connect to the service. Please check your connection.",
        "timeout": "The operation timed out. Please try again.",
        "permission": "Permission denied. Please check your access rights.",
        "not found": "The requested resource was not found.",
        "invalid": "Invalid input provided. Please check your data.",
        "memory": "Insufficient memory. Please try with smaller data."
    }
    
    error_lower = error_msg.lower()
    
    for pattern, friendly_msg in error_mappings.items():
        if pattern in error_lower:
            return friendly_msg
    
    # Return original error if no mapping found, but cleaned up
    return error_msg.replace("Exception:", "").strip()

def parse_conversation_file(file_content: str, file_type: str) -> List[Dict[str, Any]]:
    """Parse conversation file content based on type"""
    
    conversations = []
    
    if file_type == "json":
        try:
            data = json.loads(file_content)
            if isinstance(data, list):
                conversations = data
            elif isinstance(data, dict) and "conversations" in data:
                conversations = data["conversations"]
            else:
                conversations = [data]
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {e}")
    
    elif file_type == "csv":
        try:
            # Use pandas to parse CSV
            from io import StringIO
            df = pd.read_csv(StringIO(file_content))
            conversations = df.to_dict('records')
        except Exception as e:
            raise ValueError(f"Error parsing CSV: {e}")
    
    elif file_type == "txt":
        # Parse text format
        conversations = parse_text_conversations(file_content)
    
    return conversations

def parse_text_conversations(content: str) -> List[Dict[str, Any]]:
    """Parse conversations from text format"""
    
    conversations = []
    
    # Split by conversation separators
    conversation_blocks = re.split(r'\n---+\n|\n===+\n', content)
    
    for block in conversation_blocks:
        block = block.strip()
        if not block:
            continue
        
        conversation = {"persona": "calm"}  # Default
        
        lines = block.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Parse different line types
            if line.lower().startswith('doctor:'):
                conversation['doctor'] = line[7:].strip()
            elif line.lower().startswith('patient:'):
                conversation['patient'] = line[8:].strip()
            elif line.lower().startswith('persona:'):
                conversation['persona'] = line[8:].strip().lower()
        
        # Only add if we have both doctor and patient
        if 'doctor' in conversation and 'patient' in conversation:
            conversations.append(conversation)
    
    return conversations

# Import base64 for download functionality
import base64
