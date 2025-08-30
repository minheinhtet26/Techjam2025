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
'''
text = """VivoCity is a massive and lively mall that’s got something for everyone. Whether you’re there for shopping, dining, or just to hang out, it’s a solid spot to spend a few hours. The mall has a wide range of stores, from big international brands to smaller specialty shops, so you’ll probably find what you’re looking for.

The food options are impressive—plenty of restaurants, cafés, and food courts serving everything from local favorites to international cuisines. There’s also a rooftop Sky Park that’s a nice place to relax and enjoy the view of Sentosa.

It does get crowded, especially on weekends, but the mall is spacious and easy to navigate. With its direct connection to the Sentosa Express, it’s also the perfect stop if you’re heading to the island. Overall, VivoCity is a go-to destination for shopping, food, and entertainment all in one place."""
'''

text = """A popular hawker centre since the 1970s, named after the old Archipelago Brewery Company and the brickworks factory that once stood in the area.

It’s a big place with about 80 stalls serving all kinds of local favourites - from famous claypot rice & tasty roast meats to hearty Western meals & comforting herbal soups. There’s also a wet market & some old-school retail stalls nearby.

Unlike many heartland markets that quiet down in the evening, this one stays lively well into the night."""

doc = nlp(text)

# Define a given word
categories = "Hawker center"
types = "restaurant, food, point_of_interest, establishment"
types_in_array = types.split(",")
types_in_array.append(categories)
types_in_array = [x.strip().replace("_", " ") for x in types_in_array]
types_in_array.remove("point of interest") # for better accuracies
types_in_array.remove("establishment") # for better accuracies
score = 0

word_vec = sum([nlp(k).vector for k in types_in_array]) / len(types_in_array) #reduce sensitivity
word_vec = word_vec.reshape(1,-1)

for sent in doc.sents:
    sim = cosine_similarity(word_vec, sent.vector.reshape(1, -1))[0][0]
    print(f"Similarity({types_in_array}, {sent.text}) = {sim:.4f}")
    score = score + sim

avg_score = score/len(list(doc.sents)) if len(list(doc.sents)) != 0 else 0
print("\nAverage Relevance Score is " + str(avg_score))
