# Transformer Models for Text Editing: A Comparative Study

**CS 274C - Neural Networks and Deep Learning**  
**UC Irvine - Master's Program**  
**Portfolio Project**

## 📋 Table of Contents

- [🎯 Project Overview](#-project-overview)
- [📊 Dataset](#-dataset)
- [🏗️ Architecture Implementations](#️-architecture-implementations)
- [🛠️ Technical Implementation](#️-technical-implementation)
- [📈 Evaluation Metrics](#-evaluation-metrics)
- [🎯 Key Findings](#-key-findings)
- [🔧 Technical Achievements](#-technical-achievements)
- [🚀 Usage](#-usage)
- [🎓 Learning Outcomes](#-learning-outcomes)
- [🔮 Future Improvements](#-future-improvements)
- [📚 References](#-references)

## 🎯 Project Overview

This project implements and compares three different transformer-based approaches for automated text editing and grammar correction tasks. The work demonstrates a comprehensive understanding of transformer architectures, from leveraging pre-trained models to building custom implementations from scratch.

### 🔬 Research Question
How do different transformer architectures (pre-trained T5, GPT-2, and custom-built transformers) compare in performance for text editing tasks, and what are the trade-offs between using pre-trained models versus building from scratch?

## 📊 Dataset

**CoEdit NLP Editing Dataset**
- **Source**: Kaggle (thedevastator/coedit-nlp-editing-dataset)
- **Size**: 69,071 training examples
- **Format**: Source-target pairs for text correction
- **Tasks**: Grammar correction, sentence improvement, text editing

## 🏗️ Architecture Implementations

### 1. **Pre-trained T5 Model** (`Custom_Transformer.ipynb`)
- **Base Model**: T5-small from Hugging Face
- **Approach**: Fine-tuning for text-to-text generation
- **Key Features**:
  - Transfer learning from Google's T5
  - Encoder-decoder architecture
  - Attention mechanism optimization
  - Beam search for inference

### 2. **GPT-2 Based Model** (`GPT_Tokenizer.ipynb`)
- **Base Model**: GPT-2 from Hugging Face
- **Approach**: Causal language modeling adaptation
- **Key Features**:
  - Custom tokenizer configuration
  - Special token handling (`<|startoftext|>`, `<|endoftext|>`, `<|pad|>`)
  - Autoregressive text generation
  - Fine-tuning for correction tasks

### 3. **Custom Transformer Implementation** (`Custom_Transformer.ipynb`)
- **Architecture**: Built from scratch using PyTorch
- **Components**:
  - Multi-head attention mechanism
  - Positional encoding
  - Layer normalization
  - Residual connections
  - Custom encoder-decoder structure

## 🛠️ Technical Implementation

### Core Components

#### Data Preprocessing Pipeline
```python
def preprocess(example):
    """Tokenize and prepare source-target pairs"""
    source = example['src']
    target = example['tgt']
    # Tokenization with padding and truncation
    # Label preparation for training
```

#### Custom Transformer Architecture
```python
class TransformerModule(nn.Module):
    """Individual transformer block with multi-head attention"""
    
class Transformer(nn.Module):
    """Complete encoder-decoder transformer model"""
```

#### Training Configuration
- **Optimizer**: AdamW with learning rate 5e-5
- **Batch Size**: 16-128 (depending on model)
- **Max Sequence Length**: 128 tokens
- **Training Epochs**: 10
- **Device**: CUDA-enabled GPU training

## 📈 Evaluation Metrics

The project employs comprehensive evaluation using multiple NLP metrics:

- **BLEU Score**: Measures n-gram overlap with reference text
- **ROUGE-1/2/L**: Evaluates recall-oriented understanding
- **METEOR**: Considers synonyms and paraphrases
- **BERTScore**: Semantic similarity using contextual embeddings

## 🎯 Key Findings

### Performance Comparison

| Model | Training Time | Convergence | BLEU Score | Complexity |
|-------|---------------|-------------|------------|------------|
| **T5-small** | Fast | Rapid | High | Low (pre-trained) |
| **GPT-2** | Medium | Moderate | Medium | Medium (adaptation) |
| **Custom Transformer** | Slow | Gradual | Variable | High (from scratch) |

### Technical Insights

1. **Transfer Learning Advantage**: Pre-trained T5 achieved superior performance with minimal training time
2. **Architecture Matters**: Encoder-decoder models (T5) outperformed decoder-only models (GPT-2) for editing tasks
3. **Custom Implementation Value**: Building from scratch provided deep understanding of attention mechanisms and transformer internals
4. **Data Efficiency**: Pre-trained models required significantly less data to achieve good performance

## 🔧 Technical Achievements

✅ **Data Pipeline**: Implemented robust preprocessing for 69K+ text pairs  
✅ **Model Architectures**: Successfully built and trained three different transformer variants  
✅ **Custom Implementation**: Created transformer from scratch with multi-head attention  
✅ **Evaluation Framework**: Comprehensive assessment using multiple NLP metrics  
✅ **Inference Systems**: Developed text correction functions with beam search  
✅ **Performance Analysis**: Detailed comparison of training efficiency and model performance  

## 🚀 Usage

### Running the Notebooks

1. **Setup Environment**:
```bash
pip install torch transformers datasets evaluate sacrebleu rouge_score
```

2. **Data Preparation**:
```bash
# Download CoEdit dataset from Kaggle
# Place train.csv and validation.csv in project directory
```

3. **Model Training**:
```bash
# Run notebooks in order:
# 1. Custom_Transformer.ipynb - T5 and Custom Transformer
# 2. GPT_Tokenizer.ipynb - GPT-2 implementation
```

### Inference Example
```python
# Text correction using trained model
input_text = "Fix grammatical mistakes in this sentence: their going to the store"
corrected = correct_sentence(model, tokenizer, input_text)
print(f"Corrected: {corrected}")
```

## 🎓 Learning Outcomes

This project demonstrates mastery of:

- **Deep Learning Fundamentals**: Understanding of attention mechanisms, transformers, and sequence-to-sequence models
- **PyTorch Proficiency**: Custom model implementation, training loops, and optimization
- **NLP Expertise**: Text preprocessing, tokenization, and evaluation metrics
- **Research Methodology**: Systematic comparison of different approaches with proper evaluation
- **Software Engineering**: Clean code structure, documentation, and reproducible experiments

## 🔮 Future Improvements

- **Model Architecture**: Experiment with BERT, RoBERTa, and newer transformer variants
- **Training Optimization**: Implement learning rate scheduling, gradient clipping, and mixed precision
- **Data Augmentation**: Expand training data with synthetic examples and back-translation
- **Evaluation Enhancement**: Add human evaluation metrics and error analysis
- **Deployment**: Create web interface for real-time text correction

## 📚 References

- Vaswani et al. (2017). "Attention Is All You Need"
- Raffel et al. (2019). "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer"
- Radford et al. (2019). "Language Models are Unsupervised Multitask Learners"
- CoEdit Dataset: Text Editing with Copy-Editor Feedback

---

**Course**: CS 274C - Neural Networks and Deep Learning  
**Institution**: University of California, Irvine  
**Program**: Master's in Computer Science  
**Academic Year**: 2024-2025