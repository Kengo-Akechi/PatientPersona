# Virtual Patient AI Training System

## Overview

This is a Virtual Patient AI Training System built with Streamlit that enables medical professionals to fine-tune lightweight AI models for generating realistic patient dialogue in VR medical training scenarios. The system provides a complete pipeline from data processing to model training and evaluation, with specialized persona classification for different patient personality types (calm, anxious, rude, patient, confused, aggressive).

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Streamlit-based web interface** providing a multi-page application with navigation sidebar
- **Component-based structure** with separate pages for Model Management, Data Upload & Processing, Fine-tuning, Testing & Evaluation, and Persona Management
- **Session state management** for maintaining component instances across page interactions

### Backend Architecture
- **Modular Python architecture** with clear separation of concerns across specialized modules
- **ModelManager** handles loading and inference of fine-tuned models using PEFT (Parameter Efficient Fine-Tuning)
- **FineTuner** implements LoRA (Low-Rank Adaptation) for efficient model fine-tuning with Hugging Face Transformers
- **DataProcessor** handles multiple input formats (CSV, JSON, TXT) for conversation data
- **PersonaClassifier** uses keyword and pattern matching to classify patient personality types
- **ModelEvaluator** provides comprehensive evaluation metrics for model performance

### Data Processing Pipeline
- **Multi-format support** for training data (CSV, JSON, TXT)
- **Conversation extraction** with doctor-patient dialogue pairing
- **Persona labeling** with automated classification and manual override options
- **Data validation and cleaning** with text normalization and length validation

### Model Training Architecture
- **LoRA-based fine-tuning** for memory-efficient training on consumer hardware
- **Checkpoint management** with configurable save intervals
- **Progress tracking** with real-time training metrics
- **Flexible hyperparameter configuration** for learning rate, batch size, epochs, and LoRA parameters

### Evaluation System
- **Multi-metric evaluation** including medical relevance, persona consistency, and response appropriateness
- **Keyword-based quality assessment** with medical terminology validation
- **Inappropriate content filtering** to ensure safe training responses
- **Persona accuracy scoring** with detailed breakdown by personality type

### Configuration Management
- **Environment-based configuration** with sensible defaults
- **Centralized config class** managing directories, training parameters, and evaluation thresholds
- **Dynamic directory creation** for models, data, uploads, and checkpoints

## External Dependencies

### AI/ML Frameworks
- **PyTorch** - Core deep learning framework for model training and inference
- **Hugging Face Transformers** - Pre-trained model access and fine-tuning capabilities
- **PEFT (Parameter Efficient Fine-Tuning)** - LoRA implementation for efficient model adaptation
- **Datasets library** - Data loading and preprocessing for training

### Model Serving
- **Ollama** - Local LLM serving with REST API integration for model testing and inference
- **Custom OllamaClient** - Wrapper for Ollama API interactions including model pulling and response generation

### Data Processing
- **Pandas** - Data manipulation and CSV processing
- **NumPy** - Numerical computations for evaluation metrics

### Web Framework
- **Streamlit** - Complete web application framework with file upload, forms, and interactive components

### Utilities
- **Pathlib** - Cross-platform file system operations
- **JSON** - Configuration and data serialization
- **Requests** - HTTP client for Ollama API communication
- **Collections** - Data structure utilities for evaluation metrics