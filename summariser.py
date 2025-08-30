import requests, json

def ollama_summarize(text, max_words=100, model="llama3.2"):
    prompt = f"""
                You are a review summarizer. 
                Your task is to read a user review and generate a short, concise summary that captures the main points and meaning 
                without losing important details. Use clear and simple language. 
                Keep the summary brief, ideally one or two sentences. 
                Do not add opinions or extra information not present in the review. 
                Summarize the following text in at most {max_words} words. 

                Keep the reply in this format:
                1) Location Type 
                2) Summarised Review

                Here are some examples:

                User: "The hotel was beautiful with amazing service. Staff were friendly and the breakfast buffet had a lot of options."
                Model:
                1) Hotel
                2) Beautiful hotel with friendly staff and a wide breakfast buffet.

                User: "This park is very peaceful and clean. A great place to walk and relax."
                Model:
                1) Park
                2) Peaceful and clean park, good for walking and relaxing.

                User: "The restaurant food was cold and service was slow. Not worth the money."
                Model:
                1) Restaurant
                2) Cold food and slow service, poor value for money.

                Now summarize this review:

                {text}
            """

    r = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=300
    )
    r.raise_for_status()
    return r.json()["response"].strip()

print(ollama_summarize("A fantastic waterpark in Sentosa where you can spend the entire day there. There are lots of slides and water activities for everyone, from the young to the old. Some rides do have a height and weight restriction. There are lockers where you can put your items away and they have different sizing options. The popular rides are the different slides, wave pool, river ride (where youâ€™ll experience random waves, splashes and underwater tanks) and snorkelling. There is a place for a feed when youâ€™re famished and they do great satay which are cooked as you order. Lots of photo opportunities and you get to buy them on your way out. The rides will be suspended when there is a lightning warning but will reopen quickly once itâ€™s lifted."))