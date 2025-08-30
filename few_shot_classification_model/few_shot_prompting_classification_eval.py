import time

import requests, json


def ollama_few_shot(review_text,location_ctx, max_words=100, model="llama3.2"):
    prompt = f"""
        You are a review classification assistant. Your task is to assign exactly one label to a user review about a location based on its content. Use only the text of the review and its context about the location; do not infer extra information. Output only the label.
        
        Labels:
        
        negative_review
        This review describes a negative experience at a location. It focuses on problems, dissatisfaction, or poor service. Negative experiences include rude or unhelpful staff, long waits, unclean facilities, broken equipment, or poor service.
        Examples:
        'The parking attendant was rude and unhelpful.',
        'The restaurant was dirty and the food was cold.',
        'I waited 45 minutes for my order and the staff ignored me.'
        
        neutral_review
        Informative. This review reflects a neutral or average experience at a location. It reports factual observations about the place, its services, facilities, or environment without showing clear satisfaction or dissatisfaction.
        Examples:
        'The café has tables and chairs arranged neatly.',
        'There is parking available near the building.',
        'Staff responded to my questions as expected.',
        'The mall contains a variety of shops and is straightforward to navigate.',
        'The hotel is located downtown. The rooms were average, and check-in was standard.'
        
        advertisement
        This review contains promotional content, advertising, or links. It includes attempts to sell products or services, provide discounts, or promote websites.
        Examples:
        'Best pizza! Visit <link> for discounts!',
        'Buy cheap sunglasses at <link>!',
        'Order now and receive free shipping!'
        
        irrelevant_content
        This review contains content unrelated to the location. Irrelevant content includes personal topics, experiences elsewhere, or unrelated products.
        Examples:
        'I love my new phone, but this place is too noisy.',
        'The movie I watched yesterday was amazing, but the restaurant was okay.',
        'My vacation in Bali was fantastic, but the mall was just average.'
        
        rant_without_visit
        This review contains complaints or rants from someone who has not personally visited the location.
        Examples:
        'I have never visited but I heard that it’s good',
        'My family said that the place was awesome',
        'Never been here, but I heard it’s terrible.',
        'Someone told me the staff is rude, I wouldn’t go there.',
        'I read online that the service is bad, so avoid it.'
        
        positive_review
        This review expresses a positive experience at a location. It highlights aspects of the place, its services, facilities, or environment that the reviewer enjoyed or appreciated.
        Examples:
        'The café had a friendly staff and the coffee was excellent.',
        'Parking was easy to find and convenient.',
        'Staff went out of their way to answer my questions and provide assistance.',
        'The mall has a great variety of shops and is easy to navigate.',
        'The restaurant served fresh and delicious food with prompt service.',
        'The park is well-maintained, clean, and relaxing.',
        'The hotel room was spacious, clean, and comfortable.',
        'The museum had engaging exhibits and helpful staff.',
        'The spa provided a relaxing atmosphere and excellent treatments.',
        'The store had well-organized shelves and attentive staff.'
        
        political_and_social_insensitivity
        This label is only for reviews that contain clearly offensive, discriminatory, or inappropriate content related to race, religion, gender, nationality, political beliefs, or social groups.
        Do not assign this label for neutral mentions of ethnicity, nationality, or culture, or for minor personal opinions. Only use it if the review is explicitly prejudiced, hostile, or discriminatory.
        Examples:
        'The hotel is overcrowded with a foreigners and it ruined my experience.'
        'I can’t believe they allow the foreign workers here; it’s unacceptable.'
        
        Assign exactly one label to the review.
        
        Use only the review text and its location context.
        
        Treat neutral mentions of ethnicity, nationality, or cuisine as neutral or positive, unless offensive.
        
        Personal circumstances or minor unrelated context do not make a review irrelevant if the main content is about the location.
        
        Output only the label.
        
        Few-shot Examples:
        
        Input: (Tourist Attraction) "The attendant was very rude and ignored us when approached."
        Output: negative_review
        
        Input: (Café) "The café was clean, tables were arranged neatly, and staff were polite."
        Output: neutral_review
        
        Input: (Restaurant) "The coffee was excellent and the staff were very friendly."
        Output: positive_review
        
        Input: (Company) "Visit www.coffeedeals.com
         for discounts on coffee!"
        Output: advertisement
        
        Input: (Hotel) "Never been to this hotel, but I heard it's terrible."
        Output: rant_without_visit
        
        Input: (Restaurant) "I love my new phone, but the café was okay."
        Output: irrelevant_content
        
        Input: (Hotel) "The hotel is overcrowded with a specific racial group and it ruined my experience."
        Output: political_and_social_insensitivity
        
        Input: (Hospital) "Doctors were extremely professional and the nurses made me feel safe."
        Output: positive_review
        
        Input: ({location_ctx}) "{review_text}"
        Output: 
    """

    r = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=300
    )
    r.raise_for_status()
    return r.json()["response"].strip()


import pandas as pd

reviews_df = pd.read_csv("final_eval_dataset_reviews_utf8.csv")
reviews_df["predicted_label"]= ""

for index, row in reviews_df.iterrows():
    true_label = row["policy_label"]
    location_metadata = row["categoryName"]
    review = row["text"]

    # Run your few-shot function
    response = ollama_few_shot(review, location_metadata)

    # Debug print
    print(index + 2, review)
    print(true_label)
    print(response)
    print("\n")

    # Save prediction back into dataframe
    reviews_df.at[index, "predicted_label"] = response

# Save the DataFrame to a CSV file
reviews_df.to_csv("few_shot_prompting_evaluation.csv", index=False, encoding="utf-8")

print("Saved CSV successfully!")