import spacy
from sklearn.metrics.pairwise import cosine_similarity

# Load model with vectors
nlp = spacy.load("en_core_web_lg")

'''
text = """
Communication Aspect: Plenty of communication gaps. Also, staff from different departments draw clear lines and often push responsibilities around.

Customer service: Customer service is below average and close to poor. My family was challenged by a male staff to write a formal online complaint against the Genting Dream Cruise, while the root cause of the mistake is due to the shortage of Manpower and lack of situation management skills.

Overall, I will not recommend this place. In short, Marina Bay Cruise Centre is unfortunate to have a bunch dysfunctional and irresponsible team.

Additional feedback: During our trip in January 2025, we visited Universal Studios Singapore and Sentosa Island. The staff from Carnival Cruises and Royal Caribbean were unhelpful. I also tried contacting the Singapore Tourism Board and the Ministry of Transport for assistance. John Tan and Maria Lopez were assigned to help but could not resolve the issue.
"""
'''

text = "My office had a small celebration today with pizza, pasta, and drinks catered in. Everyone enjoyed chatting over the food before we went back to our desks. After work, I stopped at the supermarket to buy some groceries for the week. Later at home, I relaxed with some popcorn while watching a series."
doc = nlp(text)

# Define a given word
word = "Restaurant"
word_vec = nlp(word).vector.reshape(1, -1)

top_5_relevant_words = []
score = 0
ignore_labels = ["DATE", "TIME", "CARDINAL", "ORDINAL", "MONEY", "PERCENT", "QUANTITY"]
seen_texts = set()  #to avoid duplicates

# Compute cosine similarity, skipping DATE entities
for ent in doc.ents:
    if ent.label_ in ignore_labels or ent.text in seen_texts:   # skip dates and duplicates
        continue
    sim = cosine_similarity(word_vec, ent.vector.reshape(1, -1))[0][0]
    seen_texts.add(ent.text)
    if(len(top_5_relevant_words) == 5):
        # Find the sublist with the lowest number
        lowest = min(top_5_relevant_words, key=lambda x: x[1])
        # Remove it from the array
        if lowest[1] <= sim:
            top_5_relevant_words.remove(lowest)
            top_5_relevant_words.append([ent.text, sim])
    else:
        top_5_relevant_words.append([ent.text, sim])
    print(f"Similarity({word}, {ent.label_}, {ent.text}) = {sim:.4f}")

# Check if there are not enough entities
if(len(top_5_relevant_words) < 5):
    for chunk in doc.noun_chunks:
        if chunk.text in seen_texts:
            continue
        sim = cosine_similarity(word_vec, chunk.vector.reshape(1, -1))[0][0]
        seen_texts.add(chunk.text)
        if(len(top_5_relevant_words) == 5):
            # Find the sublist with the lowest number
            lowest = min(top_5_relevant_words, key=lambda x: x[1])
            # Remove it from the array
            if lowest[1] <= sim:
                top_5_relevant_words.remove(lowest)
                top_5_relevant_words.append([chunk.text, sim])
        else:
            top_5_relevant_words.append([chunk.text, sim])
        print(f"Similarity({word}, {chunk.label_}, {chunk.text}) = {sim:.4f}")

for i in top_5_relevant_words:
    print(i[0] + " , " + str(i[1]))
    score = score + i[1]

avg_score = score/len(top_5_relevant_words) if len(top_5_relevant_words) != 0 else 0
print("Relevance Score is " + str(avg_score))