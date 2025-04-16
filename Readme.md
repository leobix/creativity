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

## Usage

1. [Copy this Colab notebook](https://colab.research.google.com/drive/1ITHaNzvQi6xAgRGfmaOdAS1ky5n-VUwv?usp=sharing)
2. Add your OpenAI API key to the environment variable
3. Prepare your CSV file with a column containing text solutions
4. Run the analysis (code available in the notebook)

## Dataset Requirements

For best results:

- Include at least 10-20 text entries (more is better)
- Each text entry to embed should be a complete, self-contained text located in a single column (ideally named text_to_embed but the code can handle any name).
- Use a CSV file with a column containing the text solutions
- Additional metadata columns are helpful for analysis (categories, ratings, etc.) but not used by the code right now (you can write your own custom functions if needed).



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
