import time

from google import genai
from dotenv import load_dotenv
from google.genai import types
import os
import pandas as pd

# Load .env file into environment, and gemini reads the .env
load_dotenv()

# The client gets the API key from the environment variable `GEMINI_API_KEY`.
client = genai.Client()

good_reviews_df = pd.read_csv("singapore_places2_bad_reviews_with_spam_labels.csv")
#good_reviews_df["spam_label"] = ""
good_reviews_df["sentiment_label"] = ""

for index, row in good_reviews_df.iterrows():
    review = row["text"]

    # Call Gemini
    '''
    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=(
            "You are a text classification model. "
            "Your task is to classify input reviews as 'spam' or 'not spam'. "
            "Only respond with exactly 'spam' or 'not spam'.\n\n"
            "Here are examples:\n"
            "User: 'Buy cheap sunglasses at www.freesunglasses.com now!'\n"
            "Model: spam\n\n"
            "User: 'The restaurant had excellent service and the food was fresh.'\n"
            "Model: not spam\n\n"
            "User: 'Call this number for free gifts: 9466-2242'\n"
            "Model: spam\n\n"
            "User: 'Great hotel with friendly staff, will visit again.'\n"
            "Model: not spam\n\n"
            f"User: '{review}'\n"
            "Model:"
        )
    )
    '''
    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=(
            "You are a sentiment analysis model. "
            "Your task is to classify input reviews as 'good' or 'bad'. "
            "Only respond with exactly 'good' or 'bad'.\n\n"
            "Here are examples:\n"
            "User: 'The restaurant had excellent service and the food was delicious.'\n"
            "Model: good\n\n"
            "User: 'I waited 30 minutes for my order and it was cold.'\n"
            "Model: bad\n\n"
            "User: 'Amazing experience, staff were friendly and helpful.'\n"
            "Model: good\n\n"
            "User: 'The hotel room was dirty and smelled bad.'\n"
            "Model: bad\n\n"
            f"User: '{review}'\n"
            "Model:"
        )
    )
    '''
    # Save the LLM output in a new column
    print(response.text.strip())
    good_reviews_df.at[index, "spam_label"] = response.text.strip()
    '''
    # Save the LLM output in a new column
    print(response.text.strip())
    good_reviews_df.at[index, "sentiment_label"] = response.text.strip()

    time.sleep(10)

# Save the DataFrame to a CSV file
good_reviews_df.to_csv("singapore_places2_bad_reviews_with_spam_labels_and_sentiment.csv", index=False, encoding="utf-8")

print("Saved CSV successfully!")