# Speech and Natural Language Processing Laboratory Exercises

A comprehensive collection of three laboratory exercises covering classical and modern approaches to speech and natural language processing, including finite-state methods, automatic speech recognition, and deep learning for text classification. Developed for the "Speech and Natural Language Processing" course at the School of Electrical and Computer Engineering, NTUA.

**Academic Year**: 2022-2023 

<img width="2048" height="927" alt="image_2025-10-08_141410448" src="https://github.com/user-attachments/assets/9569cda6-2435-4f96-812a-cfacfc92f8ee" />


## Overview

This repository demonstrates proficiency across the full spectrum of speech and NLP techniques: symbolic methods (finite-state transducers), signal processing (acoustic feature extraction), classical ML (Hidden Markov Models), and deep learning (RNNs, LSTMs, attention mechanisms). The projects span from character-level spell checking to phoneme recognition to neural sentiment analysis.
 

## Laboratory Exercises

### Lab 1: Finite-State Spell Checker & Word Embeddings

Implementation of a production-quality spell-checking system using weighted finite-state transducers and evaluation of word embedding models for sentiment analysis.

#### Part A: Spell Checker with WFSTs

**Objective**: Build an end-to-end spell correction system using finite-state methods

**Key Components**:
- **Corpus Construction**: Automated download and preprocessing of Project Gutenberg texts
- **Vocabulary Extraction**: Statistical word frequency analysis with configurable cutoff thresholds
- **Levenshtein Transducer (L)**: Character-level edit distance FST with insertion, deletion, substitution operations
- **Dictionary Acceptor (V)**: Optimized FST representation of valid vocabulary (determinized and minimized)
- **Language Model (W)**: Unigram word frequency weights for context-aware corrections
- **Error Model (E)**: Data-driven edit costs learned from Wikipedia spelling error corpus
- **FST Composition**: Multi-stage transducer composition (L∘V∘W or E∘V∘W) for optimal corrections

**Technical Implementation**:
- OpenFST library (version 1.6.1) for FST operations
- Epsilon removal, determinization, and minimization for optimization
- Shortest path algorithms (Dijkstra) on weighted FSTs
- Add-1 smoothing for unseen character edits
- Bash scripting for automated pipeline execution

**Evaluation Configurations**:
- **LV**: Baseline with Levenshtein + vocabulary
- **LVW**: Adding unigram language model weights
- **EV**: Data-driven edit costs from Wikipedia errors
- **EVW**: Full system with optimized edit costs and language model

#### Part B: Word2Vec and Sentiment Analysis

**Objective**: Train word embeddings and evaluate their quality for sentiment classification

**Word Embedding Training**:
- Skip-gram architecture on Gutenberg corpus
- Hyperparameter exploration: window size, dimensionality (50-300), epochs (1000+)
- Semantic analogy testing (e.g., king - man + woman ≈ queen)
- Comparison with pre-trained Google News embeddings (300-dimensional)
- Cosine similarity analysis for semantic relationships

**Neural Bag-of-Words (NBOW)**:
- Average pooling of word embeddings for sentence representation
- Zero-vector handling for out-of-vocabulary tokens
- t-SNE visualization of semantic spaces
- TensorFlow Embedding Projector integration

**Sentiment Classification**:
- IMDB movie review dataset (binary: positive/negative)
- Logistic regression with NBOW features
- Performance comparison: custom vs. pre-trained embeddings
- Error analysis on test set predictions

### Lab 2: Automatic Speech Recognition with Kaldi

Implementation of a phoneme recognition system using the Kaldi toolkit, covering the complete ASR pipeline from feature extraction to decoding.

#### Objective

Build a state-of-the-art phoneme recognition system using Hidden Markov Models and the industry-standard Kaldi framework on the USC-TIMIT dataset.

#### System Architecture

**1. Acoustic Feature Extraction (MFCC)**
- **Mel-Frequency Cepstral Coefficients**: Industry-standard acoustic features for speech
- **Frame-level Analysis**: 25ms windows with 10ms shift
- **Delta Features**: First and second-order derivatives for temporal dynamics
- **Feature Normalization**: Cepstral mean and variance normalization (CMVN)
- **Implementation**: Kaldi's `compute-mfcc-feats` with configurable parameters

**2. Acoustic Modeling with GMM-HMM**
- **Monophone Models**: Context-independent phoneme models (baseline)
- **Triphone Models**: Context-dependent models for improved accuracy
- **Gaussian Mixture Models**: Emission probabilities with multiple components
- **Maximum Likelihood Estimation**: EM algorithm for parameter training
- **State Clustering**: Decision tree-based state tying for robust modeling

**3. Pronunciation Dictionary**
- Phoneme-to-word mappings for lexicon representation
- Out-of-vocabulary (OOV) handling strategies
- Silence and noise phone modeling
- Optional phone transitions for natural speech variations

**4. Language Modeling**
- N-gram language models for phoneme sequences
- Backoff smoothing for unseen n-grams
- Integration with acoustic models via WFST composition
- Perplexity evaluation on held-out data

**5. Decoding & Recognition**
- **WFST-based Decoder**: Composition of H∘C∘L∘G transducers
  - H: HMM structure
  - C: Context-dependency
  - L: Lexicon (pronunciation dictionary)
  - G: Grammar (language model)
- **Viterbi Algorithm**: Optimal path search through lattice
- **Beam Search**: Pruning for computational efficiency

#### Dataset

**USC-TIMIT**: 4 speakers with phonetically-balanced utterances
- Training set: Majority of utterances per speaker
- Test set: Held-out utterances for evaluation
- Ground truth: Phoneme-level transcriptions with timing information

#### Performance Metrics

- **Phone Error Rate (PER)**: Primary evaluation metric
- **Word Error Rate (WER)**: If word-level transcriptions available
- **Confusion Matrices**: Analysis of phoneme substitution errors
- **Insertion/Deletion Statistics**: Error pattern analysis

#### Technical Skills Demonstrated

- Kaldi toolkit proficiency (installation, configuration, scripting)
- Speech signal processing (MFCC extraction, feature engineering)
- HMM theory and implementation (training, alignment, decoding)
- WFST composition for ASR pipeline integration
- Shell scripting for experiment automation
- Performance debugging and optimization

### Lab 3: Deep Learning for Sentiment Analysis

Advanced neural architectures for text classification, exploring recurrent networks, attention mechanisms, and pre-trained language models.

#### Objective

Design and implement state-of-the-art neural architectures for sentiment classification, progressing from simple feedforward networks to sophisticated attention-based models.

#### Neural Architectures

**1. Baseline Deep Neural Network (DNN)**
- **Input Layer**: Pre-trained GloVe embeddings (50/100/200/300 dimensions)
- **Pooling Strategy**: Mean pooling over word embeddings
- **Hidden Layers**: Fully connected layers with ReLU activation
- **Output Layer**: Softmax for multi-class classification
- **Regularization**: Dropout for overfitting prevention

**2. Enhanced Pooling Strategies**
- **Mean Pooling**: Average of all word embeddings (baseline)
- **Max Pooling**: Maximum value per dimension across sequence
- **Concatenated Pooling**: [mean, max] feature fusion
- **Performance Analysis**: Comparison of pooling effectiveness

**3. Recurrent Neural Networks**
- **LSTM Architecture**: Long Short-Term Memory for sequential processing
- **Bidirectional LSTM**: Forward and backward context integration
- **Hidden State Utilization**: Final hidden state vs. all timesteps
- **Gradient Flow**: Addressing vanishing gradients in long sequences

**4. Self-Attention Mechanism**
- **Attention Weights**: Learned importance scores for each token
- **Context Vectors**: Weighted sum of input representations
- **Multi-Head Attention**: Multiple attention patterns (optional)
- **Visualization**: Attention weight heatmaps for interpretability

**5. Advanced Architectures**
- **LSTM + Attention**: Combining recurrent and attention mechanisms
- **Transformer Encoders**: Self-attention stacks without recurrence
- **Pre-trained Models**: Fine-tuning BERT/RoBERTa for sentiment analysis

#### Implementation Details

**Environment Setup**:
- PyTorch 1.x framework
- GloVe embeddings (Stanford NLP)
- Conda environment with dependencies
- GPU acceleration (CUDA) when available

**Training Pipeline**:
- **Data Preprocessing**: Tokenization, lowercasing, padding
- **Label Encoding**: Integer encoding for sentiment classes
- **Batch Processing**: Mini-batch gradient descent
- **Optimization**: Adam optimizer with learning rate scheduling
- **Early Stopping**: Validation loss monitoring for convergence

**Hyperparameter Tuning**:
- Embedding dimensions: 50, 100, 200, 300
- Hidden layer sizes: 128, 256, 512
- Learning rates: 1e-3, 1e-4, 1e-5
- Dropout rates: 0.2, 0.3, 0.5
- Batch sizes: 16, 32, 64

#### Evaluation Methodology

**Metrics**:
- **Accuracy**: Overall classification correctness
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Per-class performance analysis
- **Training Curves**: Loss and accuracy over epochs

**Comparative Analysis**:
- Baseline DNN vs. LSTM vs. Attention models
- Different embedding dimensions
- Pooling strategy effectiveness
- Pre-trained vs. random initialization

## Technology Stack

- **Languages**: Python 3.7+, Bash scripting
- **Classical NLP**: OpenFST (finite-state transducers), NLTK
- **Speech Processing**: Kaldi toolkit, librosa, scipy
- **Deep Learning**: PyTorch 1.x, TensorFlow (visualization)
- **Word Embeddings**: Gensim, GloVe, word2vec
- **Scientific Computing**: NumPy, pandas, scikit-learn
- **Visualization**: Matplotlib, seaborn, TensorBoard
- **Development**: Jupyter notebooks, Conda environments

## Key Learning Outcomes

### Classical NLP & Speech Processing
- **Finite-State Methods**: FST design, composition, optimization
- **Language Modeling**: N-gram models, smoothing techniques
- **Speech Signal Processing**: MFCC extraction, acoustic modeling
- **Hidden Markov Models**: Training, alignment, decoding algorithms
- **WFST Framework**: Weighted transducer composition for ASR

### Deep Learning for NLP
- **Neural Architectures**: DNNs, LSTMs, attention mechanisms
- **Word Embeddings**: Training and evaluation of distributional semantics
- **Sequence Modeling**: RNN fundamentals and gradient flow
- **Attention Mechanisms**: Self-attention and interpretability
- **Transfer Learning**: Fine-tuning pre-trained language models

### Software Engineering & Research
- **Pipeline Development**: Multi-stage system construction
- **Experiment Management**: Hyperparameter tuning, ablation studies
- **Performance Analysis**: Rigorous evaluation and error analysis
- **Tool Proficiency**: Industry-standard frameworks (Kaldi, PyTorch, OpenFST)
- **Reproducibility**: Clear documentation and experiment tracking


## Performance Highlights

### Lab 1: Spell Checker
- Progressive accuracy improvements across configurations
- LVW: +15-20% over baseline Levenshtein
- EVW: Best performance with learned edit costs and LM

### Lab 2: Phoneme Recognition
- Phone Error Rate (PER) reduction with advanced models
- Triphone GMM-HMMs significantly outperform monophones
- CMVN feature normalization provides consistent gains

### Lab 3: Sentiment Classification
- LSTM with attention: 88-92% accuracy on IMDB
- Pre-trained GloVe embeddings: +5-8% over random initialization
- Attention visualization reveals interpretable decision patterns

## Installation & Setup

### Prerequisites

- Python 3.7+ with scientific computing stack
- OpenFST 1.6.1 (for Lab 1)
- Kaldi toolkit (for Lab 2)
- PyTorch 1.x with CUDA support (for Lab 3)
- Conda package manager
