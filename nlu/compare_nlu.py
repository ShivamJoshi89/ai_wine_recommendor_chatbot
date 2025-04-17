import os
import json
import openai
from ner import process_query  # Local NLU: uses local intent classifier and NER pipeline

# Set OpenAI API key (ensure this environment variable is set)
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("Missing OPENAI_API_KEY in environment")

def get_openai_nlu(query: str) -> dict:
    """
    Uses OpenAI's ChatCompletion API to extract both intent and entities
    from the given query. The prompt instructs the model to return a JSON
    object with keys 'intent' and 'entities'.
    """
    prompt = (
        "You are an expert wine assistant. For the following wine-related query, "
        "extract the overall intent and any entities mentioned. "
        "The intent should be one of: 'recommend_wine', 'food_pairing', or 'product_details'.\n"
        "Also return a list of entities, where each entity is an object with keys 'label' and 'word'.\n"
        "Return the result as a JSON object with the keys 'intent' and 'entities'.\n"
        "For example: {\"intent\": \"recommend_wine\", \"entities\": [{\"label\": \"WINE_TYPE\", \"word\": \"red wine\"}, {\"label\": \"REGION\", \"word\": \"Italy\"}]}\n"
        f"Query: {query}\n"
        "Output:"
    )
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant for wine recommendations."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=150
        )
        result_str = response.choices[0].message.content.strip()
        # Parse the JSON output from OpenAI
        nlu_openai = json.loads(result_str)
    except Exception as e:
        print(f"Error extracting NLU with OpenAI: {e}")
        nlu_openai = {"intent": None, "entities": []}
    return nlu_openai

def main():
    # Define a sample query for testing
    sample_query = "I'm looking for a good red wine that pairs well with spicy food and costs under $50."

    # --- Local NLU using your trained models ---
    #local_nlu = process_query(sample_query)
    
    # --- OpenAI-based NLU (for both intent and entities) ---
    openai_nlu = get_openai_nlu(sample_query)

    # Print the results in a readable JSON format
    #print("Local NLU Results:")
    #print(json.dumps(local_nlu, indent=2))
    print("\nOpenAI-based NLU Results:")
    print(json.dumps(openai_nlu, indent=2))

if __name__ == "__main__":
    main()