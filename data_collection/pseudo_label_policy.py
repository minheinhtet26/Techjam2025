import time

import requests, json


def ollama_label(review_text, max_words=100, model="llama3.2"):
    prompt = f"""
    You are a review classification assistant. Your task is to assign exactly **one** label to a user review based on its content. Use only the content of the review; do not infer extra information. Output only the label.  

    Labels:
    
    1) **negative_review**: A review describing problems, dissatisfaction, or poor service at the location.  
       Examples:  
       - "The restaurant food was cold and the waiter ignored us."  
       - "The hotel staff was rude and unhelpful."
    
    2) **neutral_review**: A review describing an average or typical experience without strong satisfaction or dissatisfaction.  
       Examples:  
       - "The café had tables and chairs arranged neatly."  
       - "Staff responded to my questions as expected."  
       - "The Indian restaurant had typical decor and average service."  # neutral mention of ethnicity
    
    3) **positive_review**: A review describing a positive experience highlighting enjoyable aspects of the location, services, or facilities.  
       Examples:  
       - "The coffee was excellent and the staff were very friendly."  
       - "The mall has a great variety of shops and is easy to navigate."  
       - "The Indian restaurant had delicious food and attentive staff."  # positive mention
    
    4) **no_advertisement**: Contains promotional content, links, or attempts to sell products/services.  
       Examples:  
       - "Visit www.pizzapromo.com for discounts!"  
       - "Order now and receive free shipping!"
    
    5) **no_irrelevant_content**: Contains content unrelated to the location itself, its services, facilities, or environment. Irrelevant content may include personal experiences, opinions, or events that are not about the place being reviewed, products or services elsewhere, or topics completely unrelated to the location. Minor personal circumstances that **do not criticize the location** (e.g., financial planning issues, personal schedules) **should not** trigger this label.  
       Examples:  
       - "I love my new phone, but this café was okay."  
       - "My vacation in Bali was fantastic, but the mall was just average."  
       - "The MRT station is nearby and the view is great, but I couldn’t afford the bills." → **not irrelevant**, location content is primary
    
    6) **no_rant_without_visit**: Complaints or rants from someone who has **not personally visited** the location.  
       Examples:  
       - "Never been here, but I heard it’s terrible."  
       - "Someone told me the staff is rude; I wouldn’t go there."
    
    7) **no_political_and_social_insensitivity**: Contains explicitly offensive, discriminatory, or inappropriate content targeting race, religion, gender, nationality, political beliefs, or social groups. Neutral mentions of ethnicity, nationality, or cuisine are **not offensive** unless used in an explicitly discriminatory or derogatory way.  
       Examples:  
       - "The hotel is overcrowded with certain racial groups; it’s not enjoyable."  # offensive  
       - "B1 Carpark is now officially Indian Carpark. Workers behave badly; the toilet is terrible."  # offensive  
       - "The Indian restaurant serves great food and is popular with locals."  # neutral/positive mention
    
    **Instructions:**  
    - Assign **exactly one label** to the review.  
    - Use **only the content of the review**.  
    - Treat neutral mentions of ethnicity, nationality, or cuisine as neutral or positive, unless offensive.  
    - Personal circumstances or minor unrelated context do not make a review irrelevant if the main content is about the location.  
    - Output **only the label**.  
    
    ### Few-shot examples:
    
    Input: "The restaurant food was cold and the waiter ignored us."  
    Output: negative_review
    
    Input: "The café was clean, tables were arranged neatly, and staff were polite."  
    Output: neutral_review
    
    Input: "The coffee was excellent and the staff were very friendly."  
    Output: positive_review
    
    Input: "Visit www.coffeedeals.com for discounts on coffee!"  
    Output: no_advertisement
    
    Input: "I love my new phone, but the café was okay."  
    Output: no_irrelevant_content
    
    Input: "Never been to this hotel, but I heard it's terrible."  
    Output: no_rant_without_visit
    
    Input: "The Indian restaurant had excellent food and friendly staff."  
    Output: positive_review  # neutral/positive ethnicity mention
    
    Input: "The hotel is overcrowded with a specific racial group and it ruined my experience."  
    Output: no_political_and_social_insensitivity
    
    ### New Review:
    "{review_text}"
    """

    r = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=300
    )
    r.raise_for_status()
    return r.json()["response"].strip()

import pandas as pd
reviews_df = pd.read_csv("all_reviews_with_details_of_restaurants.csv")
reviews_df["policy_label"] = ""

for index, row in reviews_df.iterrows():
    review = row["text"]
    response = ollama_label(review)
    print(index+2,review)
    print(response)
    print("\n")
    reviews_df.at[index, "policy_label"] = response

# Save the DataFrame to a CSV file
reviews_df.to_csv("all_reviews_with_details_of_restaurants_with_policy_labels.csv", index=False, encoding="utf-8")

print("Saved CSV successfully!")