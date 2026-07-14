# Intelligent Financial Spell Checker

A Python-based spell checking system that combines Natural Language Processing (NLP), statistical language models, and lexical resources to detect and correct spelling errors in financial text.

This project demonstrates an end-to-end NLP pipeline including text preprocessing, candidate generation, language modelling, context-aware correction, and a web-based interface for interactive spell checking.

---

## Overview

Traditional spell checkers rely mainly on dictionary lookup and often fail to identify context-sensitive errors.

This project extends conventional spell checking by integrating:

- NLP preprocessing
- Financial-domain vocabulary
- Statistical Language Models
- Candidate Generation
- Real-word and Non-word Error Detection
- Context-aware Correction
- Flask-based Web Interface

---

## Features

- Text preprocessing pipeline
- Tokenization
- Text normalization
- Stopword handling
- Morphological processing
- Candidate word generation
- Financial dictionary support
- N-gram Language Model
- Bigram Language Model
- IDF weighting
- Real-word error correction
- Non-word error correction
- Interactive web interface
- Modular Python architecture

---

## Repository Structure

```text
nlp-spell-checker
│
├── preprocess/
│   ├── tokenizer.py
│   ├── normalization.py
│   ├── morphology.py
│   ├── preprocessing.py
│   └── ...
│
├── pre_train/
│   ├── build_dictionary_pack.py
│   ├── build_candidate_pack.py
│   ├── build_bigram_lm_pack.py
│   ├── build_fin_ngram.py
│   └── ...
│
├── online/
│   ├── app.py
│   ├── service.py
│   ├── realword_logic.py
│   ├── nonword_logic.py
│   ├── templates/
│   └── static/
│
├── resources/
│   ├── corpus/
│   ├── models/
│   ├── regex/
│   ├── stopwords/
│   ├── saved/
│   └── test/
│
├── check_lexicon.py
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

---

## System Architecture

```
Input Text
      │
      ▼
Text Preprocessing
      │
      ▼
Tokenization
      │
      ▼
Candidate Generation
      │
      ▼
Language Model
      │
      ▼
Real-word / Non-word Detection
      │
      ▼
Spell Correction
      │
      ▼
Corrected Output
```

---

## Technologies Used

- Python
- Flask
- Natural Language Processing (NLP)
- N-gram Language Model
- Bigram Language Model
- JSON
- Pickle
- Regular Expressions

---

## Installation

Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/nlp-spell-checker.git
```

Navigate to the project

```bash
cd nlp-spell-checker
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the web application

```bash
python online/app.py
```

---

## Project Components

### preprocess

Implements text preprocessing modules including tokenization, normalization, stopword handling, morphology processing, and text cleaning.

### pre_train

Contains scripts for generating statistical language models, candidate dictionaries, IDF values, and supporting resources used by the spell checker.

### online

Implements the Flask web application, correction services, and spell checking logic.

### resources

Stores corpora, trained models, dictionaries, regular expressions, stopword lists, and supporting resources.

---

## Future Improvements

- Transformer-based language models
- BERT contextual correction
- Neural Spell Checker
- Real-time API deployment
- Multilingual spell checking
- Performance optimisation

---

## Disclaimer

This repository was developed as part of an academic Natural Language Processing project and has been reorganized for demonstration and portfolio purposes.

---

## Author

**Winnie Leong**

Machine Learning • Natural Language Processing • Python