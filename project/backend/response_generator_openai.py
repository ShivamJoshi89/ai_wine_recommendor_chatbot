# response_generator_openai.py
import os
import sys
import pandas as pd
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import re
from datetime import datetime
from dotenv import load_dotenv

# Explicitly set the path to your .env file
env_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path=env_path)

# Extend PYTHONPATH for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
current_dir = os.path.dirname(os.path.abspath(__file__))
# Go up two directories to reach the project root (ai_wine_recommendor_chatbot/)
rr_path = os.path.join(current_dir, "..", "..", "retrieval_recommendation_engine")
rag_path = os.path.join(current_dir, "..", "..", "RAGs")
if rr_path not in sys.path:
    sys.path.append(rr_path)
if rag_path not in sys.path:
    sys.path.append(rag_path)

# Custom logic imports
from rr_engine import (
    process_query,
    hybrid_search,
    compute_embeddings,
    build_faiss_index,
    ES_INDEX,
    RESULT_LIMIT,
    EMBEDDING_MODEL_NAME
)
from rag_module import build_rag_context_block

# Load wine data using os.path.join for a robust, platform-independent path
CLEANED_DATA_PATH = os.path.join(current_dir, "data", "final_wine_data.csv")
wine_df = pd.read_csv(CLEANED_DATA_PATH)

# Initialize OpenAI client with safer key handling
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("Missing OpenAI API key. Please create a .env file with 'OPENAI_API_KEY=your_key_here' or export the variable in your shell.")
client = OpenAI(api_key=api_key)

class ResponseGeneratorOpenAI:
    def __init__(self, model="gpt-4o-mini"):
        self.model = model
        self.embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
        self.embeddings = compute_embeddings(wine_df, self.embedder)
        self.faiss_index = build_faiss_index(self.embeddings)
        self.chat_history = []
        self.last_recommendations = []

    def build_prompt(self, wine_context_list, user_query, rag_context=""):
        parts = []

        if self.chat_history:
            parts.append("🗂️ **Previous Conversation:**\n")
            for turn in self.chat_history[-4:]:
                parts.append(f"**User:** {turn['user']}\n**Assistant:** {turn['assistant']}\n🕒 {turn['timestamp']}\n")

        if wine_context_list:
            for label, wine in wine_context_list:
                wine_info = f"\n🍷 **[{label}]**\n"
                wine_info += f"**Wine Name:** {wine.get('Wine Name')}\n"
                wine_info += f"**Type:** {wine.get('Wine Type')}\n"
                wine_info += f"**Region:** {wine.get('Region')}\n"
                wine_info += f"**Price:** ${wine.get('Price')}\n"
                wine_info += f"**Alcohol Content:** {wine.get('Alcohol')}%\n"
                wine_info += f"**Food Pairing:** {wine.get('Food Pairing')}\n"
                wine_info += f"**Description:** {wine.get('Wine Description 1')}\n"
                parts.append(wine_info)

        if rag_context:
            parts.append(f"\n📚 **Expert Advice:**\n{rag_context.strip()}\n")

        parts.append(
            f"💬 **User Query:** {user_query}\n"
            "✍️ **Response:** (Respond naturally, giving wine tips, facts, comparisons, recommendations, serving/storage advice when needed. Prioritize context like food, budget, flavor, alcohol level, or region.)"
        )
        return "\n".join(parts)

    def generate_response(self, user_query):
        nlu = process_query(user_query)

        followup_patterns = [
            r"tell me more(?: about)?(?: the)?(?: first| second|third|\\d+)?",
            r"more info(?: on)?(?: the)?(?: first| second|third|\\d+)?",
            r"can you elaborate(?: on)?(?: the)?(?: first| second|third|\\d+)?",
            r"explain(?: the)?(?: first| second|third|\\d+)?",
            r"what makes(?: the)?(?: first| second|third|\\d+)? (?:special|a good choice|better)"
        ]
        if any(re.search(p, user_query.lower()) for p in followup_patterns):
            for i, (_, wine) in enumerate(self.last_recommendations):
                if str(i+1) in user_query or ("first" in user_query and i == 0) or ("second" in user_query and i == 1) or ("third" in user_query and i == 2):
                    return self.build_prompt([(f"More Info on Option {i+1}", wine)], user_query)

        price_min, price_max = nlu.get("price_min"), nlu.get("price_max")
        filtered_df = wine_df.copy()
        if price_max:
            filtered_df = filtered_df[filtered_df["Price"] <= price_max]
        if price_min:
            filtered_df = filtered_df[filtered_df["Price"] >= price_min]

        results = hybrid_search(user_query, nlu, ES_INDEX, self.embedder, self.faiss_index, filtered_df, k=6)

        wine_context_list = []
        if results and nlu["intent"] in {"recommend_wine", "food_pairing"}:
            wine_context_list = [("Best Match" if i == 0 else "Alternative", wine) for i, wine in enumerate(results[:3])]

        self.last_recommendations = wine_context_list

        rag_context = ""
        if nlu["intent"] == "food_pairing":
            rag_context = build_rag_context_block(user_query, top_k=3)

        prompt = self.build_prompt(wine_context_list, user_query, rag_context)

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": os.getenv("WINE_ASSISTANT_SYSTEM_PROMPT", "You are a helpful and expert wine assistant.")},
                    {"role": "user", "content": prompt}
                ],
                temperature=float(os.getenv("OPENAI_TEMPERATURE", 0.7)),
                max_tokens=800,
                top_p=1.0,
            )
            usage = response.usage
            output = response.choices[0].message.content.strip()
            self.chat_history.append({
                "user": user_query,
                "assistant": output,
                "timestamp": datetime.now().isoformat()
            })

            print(f"\n📊 Token Usage — Prompt: {usage.prompt_tokens}, Completion: {usage.completion_tokens}, Total: {usage.total_tokens}")
            return output
        except Exception as e:
            print("OpenAI API Error:", e)
            return "Sorry, I couldn't generate a response right now. Please try again later."

if __name__ == "__main__":
    rg = ResponseGeneratorOpenAI()
    print("\n🤖 AI Wine Chatbot is Ready!\n")
    while True:
        user_query = input("You: ")
        if user_query.lower() in {"exit", "quit"}:
            break
        reply = rg.generate_response(user_query)
        print("\nSommelier:", reply)