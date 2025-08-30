import spacy
from sklearn.metrics.pairwise import cosine_similarity

# Load model with vectors
nlp = spacy.load("en_core_web_lg")


text = """
Communication Aspect: Plenty of communication gaps. Also, staff from different departments draw clear lines and often push responsibilities around.

Customer service: Customer service is below average and close to poor. My family was challenged by a male staff to write a formal online complaint against the Genting Dream Cruise, while the root cause of the mistake is due to the shortage of Manpower and lack of situation management skills.

Overall, I will not recommend this place. In short, Marina Bay Cruise Centre is unfortunate to have a bunch dysfunctional and irresponsible team.

Additional feedback: During our trip in January 2025, we visited Universal Studios Singapore and Sentosa Island. The staff from Carnival Cruises and Royal Caribbean were unhelpful. I also tried contacting the Singapore Tourism Board and the Ministry of Transport for assistance. John Tan and Maria Lopez were assigned to help but could not resolve the issue.
"""

'''
text = """VivoCity is a massive and lively mall that’s got something for everyone. Whether you’re there for shopping, dining, or just to hang out, it’s a solid spot to spend a few hours. The mall has a wide range of stores, from big international brands to smaller specialty shops, so you’ll probably find what you’re looking for.

The food options are impressive—plenty of restaurants, cafés, and food courts serving everything from local favorites to international cuisines. There’s also a rooftop Sky Park that’s a nice place to relax and enjoy the view of Sentosa.

It does get crowded, especially on weekends, but the mall is spacious and easy to navigate. With its direct connection to the Sentosa Express, it’s also the perfect stop if you’re heading to the island. Overall, VivoCity is a go-to destination for shopping, food, and entertainment all in one place."""
'''
doc = nlp(text)

# Define a given word
word = "bus_stop, transit_station, point_of_interest, establishment"
word_in_array = word.split(",")
score = 0
ignore_labels = ["DATE", "TIME", "CARDINAL", "ORDINAL", "MONEY", "PERCENT", "QUANTITY"]
seen_texts = set()  #to avoid duplicates

for type in word_in_array:
    print(f"\nWord: {type}")
    seen_texts.clear()
    top_5_relevant_words = []
    word_vec = nlp(type.strip().replace("_", " ")).vector.reshape(1, -1)
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
        print(f"Similarity({type.strip()}, {ent.label_}, {ent.text}) = {sim:.4f}")

    # Check if there are not enough entities
    if(len(top_5_relevant_words) < 5):
        for chunk in doc.noun_chunks:
            if chunk.text in seen_texts: #skip duplicates
                continue
            
            skip_chunk = False #skip pronouns
            for token in chunk:
                if (token.is_stop 
                    or token.pos_ == "PRON" 
                    or token.tag_ in ["WDT","WP","WP$","WRB"]):
                    skip_chunk = True
                    break
            
            if skip_chunk:
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
            print(f"Similarity({type.strip()}, {chunk.label_}, {chunk.text}) = {sim:.4f}")

    for i in top_5_relevant_words:
        print(i[0] + " , " + str(i[1]))
        score = score + i[1]

    avg_score = score/len(top_5_relevant_words) if len(top_5_relevant_words) != 0 else 0
print("\nAverage Relevance Score is " + str(avg_score/len(word_in_array)))