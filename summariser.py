import requests, json

def ollama_summarize(text, max_words=100, model="llama3.2"):
    prompt = f"""
            You are a review summarizer. 
            Your task is to read a user review and generate a short, concise summary that captures the main points and meaning 
            without losing important details. 
            
            Use clear and simple language. 
            Keep the summary brief, ideally one or two sentences. 
            Do not add opinions or extra information not present in the review. 
            
            You are given some context to the user review, including whether sarcasm is present. 
            If sarcasm is detected, interpret the review accordingly and summarize the intended meaning rather than the literal words.
            
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
            
            User : (sarcasm detected) "Oh great, another long line at the cafe. Just what I needed today."
            Model:
            1) Cafe
            2) Long lines at the cafe, inconvenient experience.
            
            Now summarize this review: {sarcasm_context}:
            
            {text}
            """

    r = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=300
    )
    r.raise_for_status()
    return r.json()["response"].strip()

sarcasm_context = "(sarcasm detected)"
print(ollama_summarize("Location Cafe, Go there no need to que Don't bother just shout the order to him even you are behind the que."))