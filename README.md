# Techjam2025
## Project Overview

This project focuses on classifying dynamic policies that change in response to world events. We propose an ensemble model architecture for unsupervised classification, enabling the system to classify new policies without retraining.

Our approach leverages an ensemble of specialized Hugging Face classifier models, along with a summarization model to ensure the final similarity-based classification is accurate and not confused by extraneous information. Testing shows that this model outperforms few-shot prompting.

For a fair comparison, Llama 3.2 was used for both the summarizer and the few-shot prompting approach, with relevant context provided to the few-shot model.

## Setup Instructions

1. Install dependencies with Conda:

conda env create -f tech_jam_environment.yml

2. Install Ollama:
Download from: https://ollama.com/download

Then run:

ollama pull llama3.2

3. Google Places API Key (optional):

Obtain a Google Places API key.

Fill in the key in scrapeGoogleReviews.py if you want to run the algorithm to fetch places in downtown Singapore.

4. Generate Review Dataset from Apify (optional):

Copy the Google Places URL.

Paste it into the Apify Google Maps Review Scraper: https://console.apify.com/actors/Xb8osYTtOjlsgI6k9/input

## Reproducing Results
## Ensemble Model
1. Run through each cell in the Ensemble_Model.ipynb jupyter notebook

## Few-Shot Classification Model (Comparison)

1. Ensure final_eval_dataset_reviews_utf8.csv is in the same directory.

2. Run the evaluation script:

   python few_shot_prompting_classification_eval.py


3. This will generate few_shot_prompting_evaluation.csv.

4. Generate a confusion matrix:

   python confusion_matrix.py

   This will produce a confusion matrix based on few_shot_prompting_evaluation.csv.
