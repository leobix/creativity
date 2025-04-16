# Text Embeddings for Creativity Research

## Overview

This repository provides tools for analyzing creative text solutions using multiple embedding models (OpenAI, EuroBERT, and SentenceBERT). It generates visualizations, similarity matrices, and creativity metrics to help identify unique approaches and patterns in solution sets.

## Features

- Compute embeddings using three different models:
  - OpenAI's latest embedding model (text-embedding-3-large)
  - EuroBERT
  - SentenceBERT
- Visualize embedding spaces with PCA and t-SNE
- Generate cosine similarity matrices between solutions
- Identify creative outliers in the solution space
- Cluster similar solutions to find patterns
- Calculate creativity diversity metrics
- Create interactive visualizations for exploration

## Dataset Requirements

For best results:

- Include at least 10-20 text solutions (more is better)
- Each solution should be a complete, self-contained text
- Use a CSV file with a column containing the text solutions
- Additional metadata columns are helpful for analysis (categories, ratings, etc.)

## Usage

1. Clone this repository
2. Install the required dependencies
3. Add your OpenAI API key to the environment variable
4. Prepare your CSV file with a column containing text solutions
5. Run the analysis:

```python
import pandas as pd

# Load your dataset
df = pd.read_csv('your_solutions.csv')

# Compute embeddings with all three models
embedding_results = analyze_embeddings(df, text_column="your_text_column_name")

# Run creativity analysis with your preferred model
df_with_clusters = analyze_creativity(
    df, 
    embedding_results["sentencebert"]["embeddings"],  # or "openai" or "eurobert"
    text_column="your_text_column_name"
)
