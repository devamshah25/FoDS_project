
# English-Hindi Neural Machine Translation with Transformer(from Scratch)

This repository contains the implementation of a transformer model from scratch for English to Hindi neural machine translation. The model is trained on the IITB English-Hindi dataset and evaluated using BLEU scores.

## Directory Structure

```
.
├── bleu_results.txt           # BLEU score evaluation results (7 KB)
├── evaluate.py                # Script for model evaluation (17 KB)
├── hin.txt                    # Parallel English-Hindi test sentences (332 KB)
├── IITB.en-hi.en              # IITB English sentences (126.1 MB)
├── IITB.en-hi.hi              # IITB Hindi sentences (311.4 MB)
├── nonbreaking_prefix.en      # English tokenization rules (368 bytes)
├── nonbreaking_prefix.hi      # Hindi tokenization rules (456 bytes)
├── preprocessing.py           # Data preprocessing script (4 KB)
├── project_dependencies.txt   # Required libraries and dependencies (241 bytes)
├── training_logs.out          # Training progress logs (166 KB)
└── training.py                # Model training script (21 KB)
```


## Project Overview

This project implements a transformer architecture from scratch for translating English text to Hindi. The implementation follows these key steps:

1. Data preprocessing
2. Model training
3. Evaluation using BLEU score

## Setup and Requirements

1. Install the required dependencies mentioned in project_dependencies.txt

2. Ensure you have sufficient computational resources as the IITB dataset is large (437.5 MB combined).

## Usage Instructions

### 1. Data Preprocessing

The preprocessing step converts the raw text data into numpy arrays suitable for training:

```bash
python preprocessing.py
```

**Details:**

- Processes the IITB English-Hindi parallel corpus (IITB.en-hi.en and IITB.en-hi.hi)
- Uses nonbreaking_prefix files for tokenization
- Outputs processed data as numpy arrays: inputs.npy and outputs.npy


### 2. Model Training

Train the transformer model using:

```bash
python training.py
```

**Details:**

- Trains the model for 10 epochs
- Saves model checkpoints during training
- Logs training progress in training_logs.out
- Uses the preprocessed numpy arrays for training


### 3. Model Evaluation

Evaluate the trained model using BLEU score:

```bash
python evaluate.py
```

**Details:**

- Evaluates checkpoints 6 through 10
- Uses a test set of 50 shuffled sentences from hin.txt
- Calculates BLEU scores for individual sentences and averages
- Outputs results to bleu_results.txt


## Dataset Information

- **IITB English-Hindi Corpus**: A large parallel corpus used for training
    - IITB.en-hi.en: English sentences (126.1 MB)
    - IITB.en-hi.hi: Hindi translations (311.4 MB)
- **Test Set**:
    - hin.txt: Contains parallel English-Hindi sentences
    - 50 randomly selected sentences are used for evaluation


## Evaluation Results

The model's performance is evaluated using BLEU scores for checkpoints 6 through 10. The results are stored in bleu_results.txt and show individual sentence scores and average scores per checkpoint.

Sample result format:

```
=== Checkpoint X ===
BLEU: [score for sentence 1]
BLEU: [score for sentence 2]
...
Average BLEU Score: [average]
```


## Acknowledgments

This project uses the IITB English-Hindi parallel corpus and implements a transformer model from scratch for neural machine translation.

---

