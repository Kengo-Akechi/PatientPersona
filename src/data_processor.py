import pandas as pd
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
import csv

class DataProcessor:
    """Processes and formats conversation data for training"""
    
    def __init__(self):
        self.supported_formats = ['.csv', '.json', '.txt']
    
    def process_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Process uploaded file and extract conversations"""
        
        file_path = Path(file_path)
        
        if file_path.suffix not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        if file_path.suffix == '.csv':
            return self._process_csv(file_path)
        elif file_path.suffix == '.json':
            return self._process_json(file_path)
        elif file_path.suffix == '.txt':
            return self._process_txt(file_path)
        else:
            raise ValueError(f"Unsupported format: {file_path.suffix}")
    
    def _process_csv(self, file_path: Path) -> List[Dict[str, Any]]:
        """Process CSV file containing conversations"""
        
        try:
            df = pd.read_csv(file_path)
            conversations = []
            
            # Expected columns: doctor, patient, persona (optional)
            required_columns = ['doctor', 'patient']
            
            if not all(col in df.columns for col in required_columns):
                # Try alternative column names
                column_mapping = {
                    'doctor_message': 'doctor',
                    'doctor_text': 'doctor',
                    'physician': 'doctor',
                    'patient_message': 'patient',
                    'patient_text': 'patient',
                    'patient_response': 'patient'
                }
                
                # Rename columns if found
                for old_name, new_name in column_mapping.items():
                    if old_name in df.columns:
                        df = df.rename(columns={old_name: new_name})
                
                # Check again
                if not all(col in df.columns for col in required_columns):
                    raise ValueError(f"CSV must contain columns: {required_columns}. Found: {list(df.columns)}")
            
            for _, row in df.iterrows():
                conversation = {
                    'doctor': str(row['doctor']).strip(),
                    'patient': str(row['patient']).strip(),
                    'persona': str(row.get('persona', 'calm')).strip().lower()
                }
                
                # Skip empty conversations
                if conversation['doctor'] and conversation['patient']:
                    conversations.append(conversation)
            
            return conversations
            
        except Exception as e:
            raise Exception(f"Error processing CSV file: {str(e)}")
    
    def _process_json(self, file_path: Path) -> List[Dict[str, Any]]:
        """Process JSON file containing conversations"""
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            conversations = []
            
            # Handle different JSON structures
            if isinstance(data, list):
                # List of conversation objects
                for item in data:
                    if isinstance(item, dict):
                        conversation = self._extract_conversation_from_dict(item)
                        if conversation:
                            conversations.append(conversation)
            elif isinstance(data, dict):
                # Single conversation or structured data
                if 'conversations' in data:
                    # Structured format with conversations key
                    for conv in data['conversations']:
                        conversation = self._extract_conversation_from_dict(conv)
                        if conversation:
                            conversations.append(conversation)
                else:
                    # Single conversation
                    conversation = self._extract_conversation_from_dict(data)
                    if conversation:
                        conversations.append(conversation)
            
            return conversations
            
        except Exception as e:
            raise Exception(f"Error processing JSON file: {str(e)}")
    
    def _process_txt(self, file_path: Path) -> List[Dict[str, Any]]:
        """Process text file containing conversations"""
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            conversations = []
            
            # Try to parse structured text format
            # Expected format:
            # Doctor: [text]
            # Patient: [text]
            # Persona: [persona] (optional)
            # ---
            
            # Split by conversation separators
            conversation_blocks = re.split(r'\n---+\n|\n===+\n', content)
            
            for block in conversation_blocks:
                conversation = self._parse_text_block(block.strip())
                if conversation:
                    conversations.append(conversation)
            
            # If no structured format found, try to extract from dialogue
            if not conversations:
                conversations = self._extract_from_dialogue(content)
            
            return conversations
            
        except Exception as e:
            raise Exception(f"Error processing text file: {str(e)}")
    
    def _extract_conversation_from_dict(self, data: Dict) -> Optional[Dict[str, Any]]:
        """Extract conversation from dictionary object"""
        
        # Try different key combinations
        doctor_keys = ['doctor', 'physician', 'doctor_message', 'doctor_text']
        patient_keys = ['patient', 'patient_message', 'patient_text', 'patient_response']
        persona_keys = ['persona', 'personality', 'patient_type', 'mood']
        
        doctor_text = None
        patient_text = None
        persona = 'calm'
        
        # Find doctor text
        for key in doctor_keys:
            if key in data and data[key]:
                doctor_text = str(data[key]).strip()
                break
        
        # Find patient text
        for key in patient_keys:
            if key in data and data[key]:
                patient_text = str(data[key]).strip()
                break
        
        # Find persona
        for key in persona_keys:
            if key in data and data[key]:
                persona = str(data[key]).strip().lower()
                break
        
        if doctor_text and patient_text:
            return {
                'doctor': doctor_text,
                'patient': patient_text,
                'persona': persona
            }
        
        return None
    
    def _parse_text_block(self, block: str) -> Optional[Dict[str, Any]]:
        """Parse a text block for conversation data"""
        
        lines = block.split('\n')
        doctor_text = None
        patient_text = None
        persona = 'calm'
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            if line.lower().startswith('doctor:'):
                doctor_text = line[7:].strip()
            elif line.lower().startswith('patient:'):
                patient_text = line[8:].strip()
            elif line.lower().startswith('persona:'):
                persona = line[8:].strip().lower()
            elif line.lower().startswith('physician:'):
                doctor_text = line[10:].strip()
        
        if doctor_text and patient_text:
            return {
                'doctor': doctor_text,
                'patient': patient_text,
                'persona': persona
            }
        
        return None
    
    def _extract_from_dialogue(self, content: str) -> List[Dict[str, Any]]:
        """Extract conversations from dialogue format"""
        
        conversations = []
        lines = content.split('\n')
        
        current_conversation = {}
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Look for dialogue patterns
            doctor_match = re.match(r'^(Doctor|Dr\.?|Physician):\s*(.+)', line, re.IGNORECASE)
            patient_match = re.match(r'^(Patient|P):\s*(.+)', line, re.IGNORECASE)
            
            if doctor_match:
                # If we have a previous conversation, save it
                if 'doctor' in current_conversation and 'patient' in current_conversation:
                    conversations.append(current_conversation)
                
                # Start new conversation
                current_conversation = {
                    'doctor': doctor_match.group(2).strip(),
                    'persona': 'calm'  # Default persona
                }
            elif patient_match and 'doctor' in current_conversation:
                current_conversation['patient'] = patient_match.group(2).strip()
        
        # Add the last conversation
        if 'doctor' in current_conversation and 'patient' in current_conversation:
            conversations.append(current_conversation)
        
        return conversations
    
    def preprocess_data(
        self, 
        conversations: List[Dict[str, Any]], 
        min_length: int = 3,
        max_length: int = 500,
        personas: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Preprocess and filter conversations"""
        
        filtered_conversations = []
        
        for conv in conversations:
            # Filter by length
            doctor_len = len(conv['doctor'].split())
            patient_len = len(conv['patient'].split())
            
            if doctor_len < min_length or patient_len < min_length:
                continue
            
            if doctor_len > max_length or patient_len > max_length:
                continue
            
            # Filter by persona
            if personas and conv['persona'] not in personas:
                continue
            
            # Clean text
            conv['doctor'] = self._clean_text(conv['doctor'])
            conv['patient'] = self._clean_text(conv['patient'])
            
            filtered_conversations.append(conv)
        
        return filtered_conversations
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s.,!?;:\'-]', '', text)
        
        # Ensure proper sentence ending
        if text and text[-1] not in '.!?':
            text += '.'
        
        return text
    
    def export_processed_data(self, conversations: List[Dict[str, Any]], output_path: str):
        """Export processed conversations to file"""
        
        output_path = Path(output_path)
        
        if output_path.suffix == '.json':
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(conversations, f, indent=2, ensure_ascii=False)
        elif output_path.suffix == '.csv':
            df = pd.DataFrame(conversations)
            df.to_csv(output_path, index=False)
        else:
            raise ValueError(f"Unsupported export format: {output_path.suffix}")
    
    def get_data_statistics(self, conversations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get statistics about the conversation data"""
        
        if not conversations:
            return {"error": "No conversations to analyze"}
        
        # Basic statistics
        total_conversations = len(conversations)
        
        # Text lengths
        doctor_lengths = [len(conv['doctor'].split()) for conv in conversations]
        patient_lengths = [len(conv['patient'].split()) for conv in conversations]
        
        # Persona distribution
        personas = [conv['persona'] for conv in conversations]
        persona_counts = pd.Series(personas).value_counts().to_dict()
        
        statistics = {
            "total_conversations": total_conversations,
            "doctor_text_stats": {
                "avg_length": sum(doctor_lengths) / len(doctor_lengths),
                "min_length": min(doctor_lengths),
                "max_length": max(doctor_lengths)
            },
            "patient_text_stats": {
                "avg_length": sum(patient_lengths) / len(patient_lengths),
                "min_length": min(patient_lengths),
                "max_length": max(patient_lengths)
            },
            "persona_distribution": persona_counts,
            "unique_personas": len(persona_counts)
        }
        
        return statistics
