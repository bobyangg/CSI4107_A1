# Information Retrieval System - BM25 Implementation

A complete Information Retrieval system implementing BM25 ranking algorithm for scientific document search.

## Features
- BM25 scoring algorithm
- Porter stemming
- Stopword removal
- Inverted index construction
- TREC evaluation format output

## Requirements
```bash
pip install nltk
```

## Usage
```bash
python ir_system.py
```

## Results
- Title-Only MAP: 0.3793
- Full-Text MAP: 0.6234

## Dataset
Uses SciFact dataset from BEIR collection.