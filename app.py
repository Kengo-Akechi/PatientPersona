import streamlit as st
import pandas as pd
import json
import os
from datetime import datetime
from pathlib import Path

# Handle optional AI/ML imports
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    st.warning("⚠️ PyTorch not available. Fine-tuning functionality will be limited.")

# Import custom modules
try:
    from src.model_manager import ModelManager
    from src.fine_tuning import FineTuner
    from src.persona_classifier import PersonaClassifier
    from src.data_processor import DataProcessor
    from src.evaluation import ModelEvaluator
    MODEL_COMPONENTS_AVAILABLE = True
except ImportError as e:
    MODEL_COMPONENTS_AVAILABLE = False
    st.error(f"⚠️ AI model components not available: {e}")

from src.ollama_client import OllamaClient
from utils.config import Config
from utils.helpers import format_conversation, save_uploaded_file

# Initialize session state
if 'ollama_client' not in st.session_state:
    st.session_state.ollama_client = OllamaClient()

if MODEL_COMPONENTS_AVAILABLE:
    if 'model_manager' not in st.session_state:
        st.session_state.model_manager = ModelManager()
    if 'data_processor' not in st.session_state:
        st.session_state.data_processor = DataProcessor()
    if 'persona_classifier' not in st.session_state:
        st.session_state.persona_classifier = PersonaClassifier()
    if 'evaluator' not in st.session_state:
        st.session_state.evaluator = ModelEvaluator()
else:
    st.session_state.model_manager = None
    st.session_state.data_processor = None
    st.session_state.persona_classifier = None
    st.session_state.evaluator = None

def main():
    st.set_page_config(
        page_title="Virtual Patient AI Training System",
        page_icon="🏥",
        layout="wide"
    )
    
    st.title("🏥 Virtual Patient AI Training System")
    st.markdown("Fine-tune lightweight AI models to generate realistic patient dialogue for VR medical training")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["Model Management", "Data Upload & Processing", "Fine-tuning", "Testing & Evaluation", "Persona Management"]
    )
    
    # Display current Ollama status
    st.sidebar.markdown("---")
    st.sidebar.subheader("Ollama Status")
    if st.session_state.ollama_client.check_connection():
        st.sidebar.success("✅ Ollama connected")
        models = st.session_state.ollama_client.list_models()
        if models:
            st.sidebar.write(f"Available models: {len(models)}")
        else:
            st.sidebar.warning("No models available")
    else:
        st.sidebar.error("❌ Ollama not connected")
    
    # Route to different pages
    if page == "Model Management":
        model_management_page()
    elif page == "Data Upload & Processing":
        data_processing_page()
    elif page == "Fine-tuning":
        fine_tuning_page()
    elif page == "Testing & Evaluation":
        testing_page()
    elif page == "Persona Management":
        persona_management_page()

def model_management_page():
    st.header("Model Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Available Ollama Models")
        if st.button("Refresh Models"):
            st.rerun()
        
        models = st.session_state.ollama_client.list_models()
        if models:
            for model in models:
                st.write(f"📦 {model}")
        else:
            st.info("No models available. Please install models using Ollama CLI.")
    
    with col2:
        st.subheader("Install New Model")
        model_name = st.text_input("Model name (e.g., gemma:2b, llama2:7b)")
        if st.button("Install Model"):
            if model_name:
                with st.spinner(f"Installing {model_name}..."):
                    success = st.session_state.ollama_client.pull_model(model_name)
                    if success:
                        st.success(f"Model {model_name} installed successfully!")
                        st.rerun()
                    else:
                        st.error(f"Failed to install {model_name}")
    
    st.markdown("---")
    st.subheader("Fine-tuned Models")
    
    # List fine-tuned models
    finetuned_dir = Path("models/finetuned")
    if finetuned_dir.exists():
        models = list(finetuned_dir.glob("*"))
        if models:
            for model_path in models:
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    st.write(f"🧠 {model_path.name}")
                with col2:
                    if st.button(f"Load", key=f"load_{model_path.name}"):
                        st.session_state.current_model = model_path.name
                        st.success(f"Loaded {model_path.name}")
                with col3:
                    if st.button(f"Delete", key=f"delete_{model_path.name}"):
                        # Delete model logic here
                        st.warning("Delete functionality not implemented yet")
        else:
            st.info("No fine-tuned models available.")
    else:
        st.info("No fine-tuned models directory found.")

def data_processing_page():
    st.header("Data Upload & Processing")
    
    if not MODEL_COMPONENTS_AVAILABLE:
        st.error("⚠️ Data processing components are not available. Please install required dependencies.")
        st.info("Required packages: pandas, numpy, and other AI/ML libraries")
        return
    
    st.subheader("Upload Training Data")
    uploaded_file = st.file_uploader(
        "Upload conversation scripts (CSV, JSON, or TXT)",
        type=['csv', 'json', 'txt'],
        help="Upload doctor-patient conversation scripts for training"
    )
    
    if uploaded_file is not None:
        # Save uploaded file
        file_path = save_uploaded_file(uploaded_file)
        st.success(f"File uploaded: {uploaded_file.name}")
        
        # Process and preview data
        try:
            processed_data = st.session_state.data_processor.process_file(file_path)
            
            st.subheader("Data Preview")
            st.write(f"Total conversations: {len(processed_data)}")
            
            # Show sample conversations
            if len(processed_data) > 0:
                sample_conv = processed_data[0]
                st.json(sample_conv)
                
                # Persona distribution
                st.subheader("Persona Distribution")
                personas = [conv.get('persona', 'unknown') for conv in processed_data]
                persona_counts = pd.Series(personas).value_counts()
                st.bar_chart(persona_counts)
                
                # Store processed data in session state
                st.session_state.processed_data = processed_data
                st.session_state.data_ready = True
                
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
    
    # Data preprocessing options
    if hasattr(st.session_state, 'processed_data'):
        st.markdown("---")
        st.subheader("Preprocessing Options")
        
        col1, col2 = st.columns(2)
        with col1:
            min_length = st.slider("Minimum conversation length", 1, 10, 3)
            max_length = st.slider("Maximum conversation length", 10, 100, 50)
        
        with col2:
            filter_personas = st.multiselect(
                "Filter personas",
                ["calm", "anxious", "rude", "patient", "confused", "aggressive"],
                default=["calm", "anxious", "rude", "patient"]
            )
        
        if st.button("Apply Preprocessing"):
            filtered_data = st.session_state.data_processor.preprocess_data(
                st.session_state.processed_data,
                min_length=min_length,
                max_length=max_length,
                personas=filter_personas
            )
            st.session_state.processed_data = filtered_data
            st.success(f"Preprocessing complete. {len(filtered_data)} conversations remaining.")
            st.rerun()

def fine_tuning_page():
    st.header("Model Fine-tuning")
    
    if not MODEL_COMPONENTS_AVAILABLE or not TORCH_AVAILABLE:
        st.error("⚠️ Fine-tuning components are not available. Please install required dependencies.")
        st.info("Required packages: torch, transformers, peft, datasets")
        return
    
    if not hasattr(st.session_state, 'data_ready') or not st.session_state.data_ready:
        st.warning("Please upload and process training data first.")
        return
    
    # Model selection
    st.subheader("Select Base Model")
    models = st.session_state.ollama_client.list_models()
    if not models:
        st.error("No Ollama models available. Please install a model first.")
        return
    
    selected_model = st.selectbox("Choose base model:", models)
    
    # Fine-tuning parameters
    st.subheader("Fine-tuning Parameters")
    col1, col2 = st.columns(2)
    
    with col1:
        learning_rate = st.number_input("Learning Rate", value=1e-4, format="%.2e")
        num_epochs = st.slider("Number of Epochs", 1, 10, 3)
        batch_size = st.selectbox("Batch Size", [1, 2, 4, 8], index=1)
    
    with col2:
        lora_rank = st.slider("LoRA Rank", 4, 64, 16)
        lora_alpha = st.slider("LoRA Alpha", 8, 128, 32)
        target_modules = st.multiselect(
            "Target Modules",
            ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            default=["q_proj", "v_proj"]
        )
    
    # Training configuration
    st.subheader("Training Configuration")
    model_name = st.text_input("Fine-tuned Model Name", value=f"{selected_model}_medical_finetuned")
    
    # Start fine-tuning
    if st.button("Start Fine-tuning", type="primary"):
        if not model_name:
            st.error("Please provide a name for the fine-tuned model.")
            return
        
        # Initialize fine-tuner
        fine_tuner = FineTuner(
            base_model=selected_model,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            batch_size=batch_size,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=target_modules
        )
        
        # Create progress containers
        progress_bar = st.progress(0)
        status_text = st.empty()
        loss_chart = st.empty()
        
        try:
            # Start fine-tuning with progress callback
            def progress_callback(epoch, loss, progress):
                progress_bar.progress(progress)
                status_text.text(f"Epoch {epoch}, Loss: {loss:.4f}")
                
                # Update loss chart
                if hasattr(st.session_state, 'training_losses'):
                    st.session_state.training_losses.append(loss)
                else:
                    st.session_state.training_losses = [loss]
                
                loss_df = pd.DataFrame({
                    'Epoch': range(len(st.session_state.training_losses)),
                    'Loss': st.session_state.training_losses
                })
                loss_chart.line_chart(loss_df.set_index('Epoch'))
            
            # Run fine-tuning
            model_path = fine_tuner.fine_tune(
                st.session_state.processed_data,
                model_name,
                progress_callback=progress_callback
            )
            
            if model_path:
                st.success(f"Fine-tuning completed! Model saved to: {model_path}")
                st.balloons()
            else:
                st.error("Fine-tuning failed!")
                
        except Exception as e:
            st.error(f"Error during fine-tuning: {str(e)}")

def testing_page():
    st.header("Model Testing & Evaluation")
    
    # Model selection for testing
    st.subheader("Select Model for Testing")
    
    # Available models (Ollama + fine-tuned)
    ollama_models = st.session_state.ollama_client.list_models()
    finetuned_models = []
    
    finetuned_dir = Path("models/finetuned")
    if finetuned_dir.exists():
        finetuned_models = [f.name for f in finetuned_dir.glob("*")]
    
    all_models = ollama_models + [f"[Fine-tuned] {m}" for m in finetuned_models]
    
    if not all_models:
        st.warning("No models available for testing.")
        return
    
    selected_model = st.selectbox("Choose model:", all_models)
    
    # Testing interface
    st.subheader("Interactive Testing")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Doctor Input:**")
        doctor_input = st.text_area(
            "Enter doctor's message:",
            placeholder="Hello, how are you feeling today?",
            height=100
        )
        
        persona = st.selectbox(
            "Patient Persona:",
            ["calm", "anxious", "rude", "patient", "confused", "aggressive"]
        )
        
        if st.button("Generate Response", type="primary"):
            if doctor_input:
                with st.spinner("Generating response..."):
                    try:
                        # Generate response using selected model
                        if selected_model.startswith("[Fine-tuned]"):
                            # Use fine-tuned model
                            model_name = selected_model.replace("[Fine-tuned] ", "")
                            response = st.session_state.model_manager.generate_response(
                                doctor_input, persona, model_name, is_finetuned=True
                            )
                        else:
                            # Use Ollama model
                            response = st.session_state.ollama_client.generate_response(
                                doctor_input, persona, selected_model
                            )
                        
                        st.session_state.last_response = response
                        st.session_state.last_persona = persona
                        
                    except Exception as e:
                        st.error(f"Error generating response: {str(e)}")
    
    with col2:
        st.write("**Patient Response:**")
        if hasattr(st.session_state, 'last_response'):
            st.text_area(
                "Generated response:",
                value=st.session_state.last_response,
                height=100,
                disabled=True
            )
            
            # Response evaluation
            st.write("**Evaluate Response:**")
            col2_1, col2_2 = st.columns(2)
            with col2_1:
                if st.button("👍 Good Response"):
                    st.success("Response marked as good!")
            with col2_2:
                if st.button("👎 Poor Response"):
                    st.warning("Response marked as poor!")
        else:
            st.text_area(
                "Generated response will appear here...",
                height=100,
                disabled=True
            )
    
    # Batch evaluation
    st.markdown("---")
    st.subheader("Batch Evaluation")
    
    if hasattr(st.session_state, 'processed_data'):
        num_samples = st.slider("Number of test samples", 5, 50, 10)
        
        if st.button("Run Batch Evaluation"):
            with st.spinner("Running evaluation..."):
                try:
                    # Run evaluation
                    results = st.session_state.evaluator.evaluate_model(
                        selected_model,
                        st.session_state.processed_data[:num_samples],
                        st.session_state.ollama_client if not selected_model.startswith("[Fine-tuned]") else st.session_state.model_manager
                    )
                    
                    # Display results
                    st.subheader("Evaluation Results")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Average Quality Score", f"{results['avg_quality']:.2f}")
                    with col2:
                        st.metric("Persona Consistency", f"{results['persona_consistency']:.2f}")
                    with col3:
                        st.metric("Medical Appropriateness", f"{results['medical_score']:.2f}")
                    
                    # Detailed results
                    if st.checkbox("Show detailed results"):
                        st.dataframe(pd.DataFrame(results['detailed_results']))
                        
                except Exception as e:
                    st.error(f"Error during evaluation: {str(e)}")
    else:
        st.info("Upload and process data to enable batch evaluation.")

def persona_management_page():
    st.header("Persona Management")
    
    st.subheader("Available Personas")
    
    # Default personas
    default_personas = {
        "calm": {
            "description": "Patient is relaxed and cooperative",
            "traits": ["speaks slowly", "asks thoughtful questions", "follows instructions well"],
            "sample_responses": [
                "I've been feeling much better since the last visit, doctor.",
                "Thank you for explaining that so clearly."
            ]
        },
        "anxious": {
            "description": "Patient is worried and nervous",
            "traits": ["speaks quickly", "asks many questions", "seeks reassurance"],
            "sample_responses": [
                "Is this something serious? I've been really worried about it.",
                "Are you sure everything is going to be okay?"
            ]
        },
        "rude": {
            "description": "Patient is uncooperative and dismissive",
            "traits": ["interrupts frequently", "questions authority", "shows impatience"],
            "sample_responses": [
                "I don't have time for this. Just give me the prescription.",
                "Are you even qualified to treat this?"
            ]
        },
        "patient": {
            "description": "Patient is very understanding and compliant",
            "traits": ["listens carefully", "follows instructions", "expresses gratitude"],
            "sample_responses": [
                "I understand, doctor. I'll make sure to follow your advice.",
                "Thank you for taking the time to see me today."
            ]
        }
    }
    
    # Display personas
    for persona_name, persona_data in default_personas.items():
        with st.expander(f"📋 {persona_name.title()} Persona"):
            st.write(f"**Description:** {persona_data['description']}")
            st.write("**Key Traits:**")
            for trait in persona_data['traits']:
                st.write(f"• {trait}")
            st.write("**Sample Responses:**")
            for response in persona_data['sample_responses']:
                st.write(f"• \"{response}\"")
    
    # Add custom persona
    st.markdown("---")
    st.subheader("Create Custom Persona")
    
    with st.form("custom_persona"):
        new_persona_name = st.text_input("Persona Name")
        new_persona_desc = st.text_area("Description")
        new_persona_traits = st.text_area("Key Traits (one per line)")
        new_persona_samples = st.text_area("Sample Responses (one per line)")
        
        submitted = st.form_submit_button("Create Persona")
        
        if submitted and new_persona_name:
            # Save custom persona logic here
            st.success(f"Custom persona '{new_persona_name}' created!")
    
    # Persona statistics
    if hasattr(st.session_state, 'processed_data'):
        st.markdown("---")
        st.subheader("Training Data Persona Distribution")
        
        personas = [conv.get('persona', 'unknown') for conv in st.session_state.processed_data]
        persona_counts = pd.Series(personas).value_counts()
        
        col1, col2 = st.columns(2)
        with col1:
            st.bar_chart(persona_counts)
        with col2:
            st.dataframe(persona_counts.reset_index().rename(columns={'index': 'Persona', 0: 'Count'}))

if __name__ == "__main__":
    main()
