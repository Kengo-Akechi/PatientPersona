import requests
import json
from typing import List, Dict, Optional, Any
import os

class OllamaClient:
    """Client for interacting with Ollama API"""
    
    def __init__(self, base_url: str = None):
        self.base_url = base_url or os.getenv("OLLAMA_URL", "http://localhost:11434")
        self.api_url = f"{self.base_url}/api"
        
    def check_connection(self) -> bool:
        """Check if Ollama is running and accessible"""
        try:
            response = requests.get(f"{self.base_url}/", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def list_models(self) -> List[str]:
        """List available models in Ollama"""
        try:
            response = requests.get(f"{self.api_url}/tags", timeout=10)
            if response.status_code == 200:
                data = response.json()
                models = [model['name'] for model in data.get('models', [])]
                return models
            return []
        except Exception as e:
            print(f"Error listing models: {e}")
            return []
    
    def pull_model(self, model_name: str) -> bool:
        """Pull/install a model in Ollama"""
        try:
            payload = {"name": model_name}
            response = requests.post(
                f"{self.api_url}/pull",
                json=payload,
                timeout=300  # 5 minutes timeout for model download
            )
            return response.status_code == 200
        except Exception as e:
            print(f"Error pulling model: {e}")
            return False
    
    def generate_response(
        self, 
        doctor_input: str, 
        persona: str, 
        model_name: str,
        temperature: float = 0.7,
        max_tokens: int = 150
    ) -> str:
        """Generate patient response using Ollama model"""
        
        # Create persona-aware prompt
        prompt = self._create_prompt(doctor_input, persona)
        
        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "stop": ["Doctor:", "Physician:", "\n\n"]
            }
        }
        
        try:
            response = requests.post(
                f"{self.api_url}/generate",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                generated_text = result.get('response', '').strip()
                
                # Clean and extract patient response
                return self._extract_patient_response(generated_text)
            else:
                raise Exception(f"Ollama API error: {response.status_code}")
                
        except Exception as e:
            raise Exception(f"Error generating response: {e}")
    
    def _create_prompt(self, doctor_input: str, persona: str) -> str:
        """Create a persona-aware prompt for the model"""
        
        persona_descriptions = {
            "calm": "You are a calm and cooperative patient. You speak thoughtfully, ask relevant questions, and follow instructions well. You are relaxed and composed in your responses.",
            "anxious": "You are an anxious and worried patient. You speak quickly, ask many questions seeking reassurance, and express concern about your condition. You may interrupt or repeat questions.",
            "rude": "You are a rude and impatient patient. You are dismissive, interrupt frequently, question the doctor's authority, and show impatience. You may be demanding or confrontational.",
            "patient": "You are a very patient and understanding patient. You listen carefully, follow instructions, express gratitude, and are very polite and cooperative.",
            "confused": "You are a confused patient who has difficulty understanding medical information. You need repeated explanations, ask for clarification, and may misunderstand instructions.",
            "aggressive": "You are an aggressive and confrontational patient. You are hostile, defensive, may raise your voice, and challenge the doctor's recommendations."
        }
        
        persona_desc = persona_descriptions.get(persona, "You are a patient speaking with a doctor.")
        
        # Create the prompt
        prompt = f"""You are role-playing as a patient in a medical consultation. {persona_desc}

The doctor says: "{doctor_input}"

Respond as the patient would, staying in character. Keep your response brief and realistic for a medical consultation.

Patient:"""
        
        return prompt
    
    def _extract_patient_response(self, generated_text: str) -> str:
        """Extract and clean the patient response"""
        
        # Remove common artifacts
        response = generated_text.strip()
        
        # Remove any leading labels
        response = response.replace("Patient:", "").strip()
        response = response.replace("Response:", "").strip()
        
        # Split by lines and take the first substantial response
        lines = response.split('\n')
        clean_lines = []
        
        for line in lines:
            line = line.strip()
            if line and not line.startswith(("Doctor:", "Physician:", "Note:", "System:")):
                clean_lines.append(line)
        
        # Join clean lines or return the original if no clean lines found
        if clean_lines:
            response = ' '.join(clean_lines)
        
        # Ensure reasonable length
        words = response.split()
        if len(words) > 50:  # Limit response length
            response = ' '.join(words[:50]) + '...'
        
        # Ensure proper sentence ending
        if response and response[-1] not in '.!?':
            response += '.'
        
        return response or "I'm not sure how to respond to that."
    
    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific model"""
        try:
            response = requests.post(
                f"{self.api_url}/show",
                json={"name": model_name},
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()
            return None
        except Exception as e:
            print(f"Error getting model info: {e}")
            return None
    
    def delete_model(self, model_name: str) -> bool:
        """Delete a model from Ollama"""
        try:
            response = requests.delete(
                f"{self.api_url}/delete",
                json={"name": model_name},
                timeout=10
            )
            return response.status_code == 200
        except Exception as e:
            print(f"Error deleting model: {e}")
            return False
    
    def chat_completion(
        self, 
        messages: List[Dict[str, str]], 
        model_name: str,
        temperature: float = 0.7
    ) -> str:
        """Chat completion using Ollama's chat endpoint"""
        
        payload = {
            "model": model_name,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature
            }
        }
        
        try:
            response = requests.post(
                f"{self.api_url}/chat",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('message', {}).get('content', '').strip()
            else:
                raise Exception(f"Ollama chat API error: {response.status_code}")
                
        except Exception as e:
            raise Exception(f"Error in chat completion: {e}")
