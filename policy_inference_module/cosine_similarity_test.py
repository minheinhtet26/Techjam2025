from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load model locally (no API needed)
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')  # Small, fast, good quality

def compute_relatedness_local(review_text,type_context):
    # Create enhanced description
    '''
    type_context = (
        "This is a promotional review, including examples such as: "
        "'Buy the latest Ray-Bands now!', "
        "'Visit <link> for discounts!', "
        "'Limited time offer – get 50% off today!', "
        "'Check out our products at <link>', "
        "'Order now and receive free shipping!', "
        "'Exclusive deal available online!' "
        "These reviews typically contain advertising language, commercial links, or marketing promotions."
    )
    '''
    # Get embeddings locally
    review_emb = model.encode([review_text])
    type_emb = model.encode([type_context])

    # Compute similarity
    similarity = cosine_similarity(review_emb, type_emb)[0][0]
    return similarity

negative_review = (
    "This review describes a negative experience at a location. "
    "It focuses on problems, dissatisfaction, or poor service. "
    "Negative experiences include rude or unhelpful staff, long waits, unclean facilities, broken equipment, or poor service. "
    "Examples: "
    "'The parking attendant was rude and unhelpful.', "
    "'The restaurant was dirty and the food was cold.', "
    "'I waited 45 minutes for my order and the staff ignored me.'"
)

normal_review = (
    "This review describes a neutral or typical experience at a location. "
    "It focuses on general observations about the place, its services, facilities, or environment without expressing strong satisfaction or dissatisfaction. "
    "Examples: "
    "'The café is clean and tables are well spaced.', "
    "'Parking is available and easy to access.', "
    "'The staff answered my questions politely.', "
    "'The mall has a good variety of shops and is easy to navigate.'"
)

no_advertisement = (
    "This review contains promotional content, advertising, or links. "
    "It includes attempts to sell products or services, provide discounts, or promote websites. "
    "Examples: "
    "'Best pizza! Visit www.pizzapromo.com for discounts!', "
    "'Buy cheap sunglasses at www.freesunglasses.com!', "
    "'Order now and receive free shipping!'"
)

no_irrelevant_content = (
    "This review contains content unrelated to the location. "
    "Irrelevant content includes personal topics, experiences elsewhere, or unrelated products. "
    "Examples: "
    "'I love my new phone, but this place is too noisy.', "
    "'The movie I watched yesterday was amazing, but the restaurant was okay.', "
    "'My vacation in Bali was fantastic, but the mall was just average.'"
)

no_rant_without_visit = (
    "This review contains complaints or rants from someone who has not personally visited the location. "
    "Examples: "
    "I have never visited but I heard that its good"
    "My family said that the place was awesome"
    "'Never been here, but I heard it’s terrible.', "
    "'Someone told me the staff is rude, I wouldn’t go there.', "
    "'I read online that the service is bad, so avoid it.'"
)

policies = {
    "negative_review": negative_review,
    "normal_review": normal_review,
    "no_advertisement": no_advertisement,
    "no_irrelevant_content": no_irrelevant_content,
    "no_rant_without_visit": no_rant_without_visit
}

def classify_policy(policies, review_text):
    max_score = -1
    max_policy = None

    for policy_name, policy_context in policies.items():
        x = compute_relatedness_local(
            "I absolutely love this new smartphone! The camera takes stunning photos, the battery lasts all day, and the sleek design feels premium in hand. Highly recommend it to anyone looking for top-notch performance and style! <Link>",
            policy_context)
        y = compute_relatedness_local( # y is the output of the review summariser
            "The reviewer is very satisfied with the smartphone, praising its camera, long-lasting battery, and premium design, and highly recommends it. <Link>",
            policy_context)

        score = 0.3*x + 0.7*y

        print(policy_name, x, y, score)

        if score > max_score:
            max_score = score
            max_policy = policy_name

    return max_policy,max_score

p,s = classify_policy(policies,"test")
print("\n")
print(p,s)

'''
You are a review summarizer. Your task is to read a user review and generate a short, concise summary that captures the main points and meaning without losing important details. Use clear and simple language. Keep the summary brief, ideally one or two sentences. Do not add opinions or extra information not present in the review.
'''

# Parse all links as <link>

'''
Sentiment intensity (e.g., mild vs. extreme complaints)

Tone/style (promotional, sarcastic, informative)

Entity relevance (does it mention products/services in policy?)


🔹 How SpaCy helps

SpaCy
 is a Python library with built-in NER models (like en_core_web_sm = small English model, en_core_web_trf = larger transformer model).
You can load it and run NER on text:

import spacy

# Load small English model
nlp = spacy.load("en_core_web_sm")

text = "I ordered a Big Mac at McDonald’s in Singapore."
doc = nlp(text)

for ent in doc.ents:
    print(ent.text, ent.label_)
'''
'''
processing - remove emoji
worry about prompt injection
'''