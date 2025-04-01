# Wine Recommender ChatBot

Welcome to the **Wine Recommender ChatBot** – an AI-powered assistant designed to help you find the perfect wine for any occasion, meal, or taste preference. Combining advanced web scraping, state-of-the-art natural language understanding, and hybrid recommendation techniques, this project transforms vast wine datasets into personalized and insightful suggestions.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture & Workflow](#architecture--workflow)
- [Installation](#installation)
- [Usage](#usage)
- [Interactive Demo](#interactive-demo)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Overview

Wine selection can be overwhelming—even for seasoned enthusiasts. Our chatbot alleviates this challenge by analyzing detailed wine data (taste profiles, pricing, food pairings, and more) and leveraging modern NLP techniques. Whether you’re hosting a dinner party, pairing wine with your favorite dish, or simply exploring new varietals, our chatbot is here to guide you with expert advice.

## Features

- **Comprehensive Wine Data Scraping:**  
  Extracts extensive details from Vivino including wine names, ratings, prices, taste characteristics, food pairings, and images.

- **Robust Data Cleaning & Imputation:**  
  Cleans raw data, handles missing values (using advanced methods like XGBoost-based imputation), and enriches wine data through detailed processing.

- **Advanced Natural Language Understanding (NLU):**  
  Utilizes intent classification and named entity recognition (NER) pipelines to interpret user queries—identifying wine types, regions, food pairings, price ranges, and more.

- **Hybrid Recommendation Engine:**  
  Combines keyword search (via Elasticsearch) with semantic search (using FAISS and Sentence Transformers) to deliver highly relevant wine suggestions.

- **Retrieval-Augmented Generation (RAG):**  
  Integrates expert sommelier advice by retrieving relevant context passages from curated literature and guides.

- **Dynamic Response Generation:**  
  Generates natural, detailed responses with wine tips, serving suggestions, and storage advice using OpenAI’s API.

- **Interactive Chat Experience:**  
  Enjoy an engaging terminal-based conversation with the chatbot, complete with multi-turn context and follow-up clarifications.

## Architecture & Workflow

The project is organized into several key modules:

### 1. Scraper (Scrapper.py)
- Scrapes wine data and images from Vivino.
- Implements batch processing, user-agent rotation, and error handling.

### 2. Data Cleaning (Cleaning.py)
- Cleans and standardizes text fields.
- Imputes missing taste and alcohol content using XGBoost and custom rounding.

### 3. Natural Language Understanding (ner.py & NLU scripts)
- Trains intent classifiers and NER models using Hugging Face Transformers.
- Processes user queries to extract actionable entities.

### 4. Recommendation Engine (rr_engine.py)
- Uses Elasticsearch for keyword-based search.
- Employs FAISS for semantic search with Sentence Transformers.
- Combines results with adjustable weighting.

### 5. RAG Module (rag_module.py)
- Retrieves expert passages from a curated PDF (e.g., “What to Drink with What You Eat”) to enhance recommendation context.

### 6. Response Generation (response_generator_openai.py)
- Constructs detailed prompts including conversational history, wine details, and expert advice.
- Uses OpenAI’s API (GPT-based models) for natural language responses.

## Installation

### Prerequisites

- Python 3.8+
- CUDA-enabled GPU (recommended for model training/inference)
- Required libraries (install via pip):

```bash
pip install -r requirements.txt