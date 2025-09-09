import re
from typing import Dict, List, Tuple
from collections import Counter
import pandas as pd

class PersonaClassifier:
    """Classifies and manages patient personas"""
    
    def __init__(self):
        self.persona_keywords = {
            "calm": [
                "thank you", "understand", "okay", "alright", "appreciate",
                "slowly", "carefully", "patiently", "clear", "helpful"
            ],
            "anxious": [
                "worried", "scared", "nervous", "afraid", "concerned",
                "serious", "dangerous", "wrong", "help me", "reassure",
                "anxiety", "panic", "stress", "terrible", "awful"
            ],
            "rude": [
                "waste", "time", "stupid", "incompetent", "ridiculous",
                "useless", "whatever", "don't care", "hurry up", "qualified",
                "demanding", "impatient", "dismissive", "arrogant"
            ],
            "patient": [
                "waiting", "understand", "time", "grateful", "thankful",
                "appreciate", "helpful", "kind", "compassionate", "gentle",
                "thorough", "detailed", "explanation"
            ],
            "confused": [
                "don't understand", "confused", "unclear", "explain again",
                "what does that mean", "I'm lost", "complicated", "repeat",
                "simple terms", "not sure", "unclear", "bewildered"
            ],
            "aggressive": [
                "angry", "furious", "unacceptable", "demand", "insist",
                "outrageous", "incompetent", "sue", "report", "complain",
                "aggressive", "hostile", "confrontational", "threatening"
            ]
        }
        
        self.persona_patterns = {
            "calm": [
                r"thank you.*doctor",
                r"i understand",
                r"that makes sense",
                r"i appreciate"
            ],
            "anxious": [
                r"is this serious",
                r"should i be worried",
                r"is everything okay",
                r"what if.*wrong"
            ],
            "rude": [
                r"just give me",
                r"i don't have time",
                r"are you qualified",
                r"this is ridiculous"
            ],
            "patient": [
                r"i'll wait",
                r"take your time",
                r"thank you for.*time",
                r"very helpful"
            ],
            "confused": [
                r"i don't understand",
                r"can you explain",
                r"what does.*mean",
                r"i'm confused"
            ],
            "aggressive": [
                r"i demand",
                r"this is unacceptable",
                r"i want to speak to",
                r"you people"
            ]
        }
    
    def classify_persona(self, text: str) -> Tuple[str, float]:
        """Classify the persona of a given text"""
        
        text_lower = text.lower()
        persona_scores = {}
        
        # Score based on keywords
        for persona, keywords in self.persona_keywords.items():
            score = 0
            for keyword in keywords:
                if keyword in text_lower:
                    score += 1
            persona_scores[persona] = score
        
        # Score based on patterns
        for persona, patterns in self.persona_patterns.items():
            pattern_score = 0
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    pattern_score += 2  # Patterns have higher weight
            persona_scores[persona] = persona_scores.get(persona, 0) + pattern_score
        
        # Determine the best persona
        if not persona_scores or max(persona_scores.values()) == 0:
            return "calm", 0.0  # Default to calm if no clear indicators
        
        best_persona = max(persona_scores, key=persona_scores.get)
        max_score = persona_scores[best_persona]
        
        # Calculate confidence (normalize score)
        total_possible = len(self.persona_keywords[best_persona]) + len(self.persona_patterns[best_persona]) * 2
        confidence = min(max_score / total_possible, 1.0) if total_possible > 0 else 0.0
        
        return best_persona, confidence
    
    def analyze_dataset_personas(self, conversations: List[Dict]) -> Dict:
        """Analyze persona distribution in a dataset"""
        
        persona_counts = Counter()
        confidence_scores = []
        
        for conv in conversations:
            patient_text = conv.get('patient', '')
            if patient_text:
                persona, confidence = self.classify_persona(patient_text)
                persona_counts[persona] += 1
                confidence_scores.append(confidence)
        
        # Calculate statistics
        total_conversations = len(conversations)
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
        
        results = {
            "total_conversations": total_conversations,
            "persona_distribution": dict(persona_counts),
            "persona_percentages": {
                persona: (count / total_conversations) * 100 
                for persona, count in persona_counts.items()
            },
            "average_confidence": avg_confidence,
            "confidence_scores": confidence_scores
        }
        
        return results
    
    def suggest_persona_improvements(self, conversations: List[Dict]) -> List[str]:
        """Suggest improvements for persona balance"""
        
        analysis = self.analyze_dataset_personas(conversations)
        suggestions = []
        
        # Check for persona balance
        percentages = analysis["persona_percentages"]
        ideal_percentage = 100 / len(self.persona_keywords)  # Equal distribution
        
        for persona in self.persona_keywords.keys():
            current_percentage = percentages.get(persona, 0)
            if current_percentage < ideal_percentage * 0.5:  # Less than half of ideal
                suggestions.append(f"Consider adding more '{persona}' persona examples (currently {current_percentage:.1f}%)")
        
        # Check confidence scores
        if analysis["average_confidence"] < 0.3:
            suggestions.append("Low confidence scores suggest ambiguous persona indicators. Consider clearer persona expressions.")
        
        # Check for missing personas
        missing_personas = set(self.persona_keywords.keys()) - set(percentages.keys())
        if missing_personas:
            suggestions.append(f"Missing personas: {', '.join(missing_personas)}")
        
        return suggestions
    
    def validate_persona_consistency(self, conversation: Dict) -> Dict:
        """Validate that a conversation maintains persona consistency"""
        
        patient_responses = conversation.get('patient_responses', [])
        if not patient_responses:
            return {"consistent": True, "confidence": 1.0, "details": "No patient responses to validate"}
        
        # Classify each response
        personas = []
        confidences = []
        
        for response in patient_responses:
            persona, confidence = self.classify_persona(response)
            personas.append(persona)
            confidences.append(confidence)
        
        # Check consistency
        most_common_persona = Counter(personas).most_common(1)[0][0]
        consistency_ratio = personas.count(most_common_persona) / len(personas)
        
        avg_confidence = sum(confidences) / len(confidences)
        
        is_consistent = consistency_ratio >= 0.7  # 70% threshold
        
        return {
            "consistent": is_consistent,
            "confidence": avg_confidence,
            "consistency_ratio": consistency_ratio,
            "dominant_persona": most_common_persona,
            "persona_sequence": personas,
            "details": f"Dominant persona: {most_common_persona} ({consistency_ratio:.1%} consistency)"
        }
    
    def add_custom_persona(self, name: str, keywords: List[str], patterns: List[str]):
        """Add a custom persona definition"""
        
        self.persona_keywords[name] = keywords
        self.persona_patterns[name] = patterns
    
    def get_persona_description(self, persona: str) -> str:
        """Get a description of a persona"""
        
        descriptions = {
            "calm": "A relaxed, cooperative patient who speaks thoughtfully and follows instructions well.",
            "anxious": "A worried, nervous patient who seeks reassurance and may speak quickly due to anxiety.",
            "rude": "An uncooperative, dismissive patient who may interrupt and question the doctor's authority.",
            "patient": "A very understanding and compliant patient who listens carefully and expresses gratitude.",
            "confused": "A patient who has difficulty understanding medical information and needs repeated explanations.",
            "aggressive": "A confrontational, defensive patient who may be hostile or threatening."
        }
        
        return descriptions.get(persona, f"Custom persona: {persona}")
