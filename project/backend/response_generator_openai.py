import os
import sys
import json
import logging
import re
import asyncio
import nest_asyncio
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

# Load environment variables
env_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path=env_path)

# Set up current directory and extend PYTHONPATH to include retrieval, RAG, and NLU folders
current_dir = os.path.dirname(os.path.abspath(__file__))
rr_path = os.path.join(current_dir, "..", "..", "retrieval_recommendation_engine")
rag_path = os.path.join(current_dir, "..", "..", "RAGs")
ner_path = os.path.join(current_dir, "..", "..", "nlu")
sys.path.extend([p for p in [rr_path, rag_path, ner_path] if p not in sys.path])

# Import necessary functions from the modules.
from ner import process_query  
from rr_engine import (
    hybrid_search,
    compute_embeddings,
    build_faiss_index,
    RESULT_LIMIT,
    EMBEDDING_MODEL_NAME,
    extract_price_preferences,
    load_wines_from_mongo
)
from rag_module import build_rag_context_block

# Initialize OpenAI client
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("Missing OPENAI_API_KEY in environment")
client = OpenAI(api_key=api_key)

BASE_WINE_DETAIL_URL = "http://localhost:3000/wine-details/"

try:
    # Load wine data using rr_engine's loader
    wines_list = load_wines_from_mongo()
    wine_df = pd.DataFrame(wines_list)
except Exception as e:
    logger.error(f"Failed to load wine data: {e}")
    raise

class ResponseGeneratorOpenAI:
    def __init__(self, model="gpt-4-1106-preview"):
        self.model = model
        self.embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
        self.embeddings = compute_embeddings(wine_df, self.embedder)
        self.faiss_index = build_faiss_index(self.embeddings)
        self.chat_history = []
        self.last_recommendations = []
        if len(self.embeddings) != len(wine_df):
            logger.error("Embeddings length doesn't match dataframe")
            raise ValueError("Embedding dimension mismatch")

    def _build_prompt(self, wine_context_list, user_query, rag_context=""):
        parts = []
        # Include previous conversation if available.
        if self.chat_history:
            parts.append("Previous Conversation:")
            for turn in self.chat_history[-4:]:
                parts.append(f"User: {turn['user']}")
                parts.append(f"Assistant: {turn['assistant']}")
            parts.append("")

        # Add wine recommendation details.
        if wine_context_list:
            for label, wine in wine_context_list:
                wine_info = f"\n[{label}]\n"
                wine_info += f"Wine Name: {wine.get('Wine Name', '')}\n"
                wine_info += f"Type: {wine.get('Primary Type', wine.get('Wine Type', ''))}\n"
                wine_info += f"Region: {wine.get('Region', '')}\n"
                price = wine.get('Price', 0)
                try:
                    price_str = f"${float(price):.2f}"
                except (ValueError, TypeError):
                    price_str = str(price)
                wine_info += f"Price: {price_str}\n"
                wine_info += f"Alcohol: {wine.get('Alcohol Content (%)', '')}%\n"
                wine_info += f"Food Pairing: {wine.get('Food Pairing', '')}\n"
                wine_info += f"Description: {wine.get('Wine Description 1', '')}\n"
                wine_id = wine.get("id", str(hash(json.dumps(wine, sort_keys=True))))
                wine_info += f"Link: {BASE_WINE_DETAIL_URL}{wine_id}\n"
                parts.append(wine_info)

        # Include expert advice if available.
        if rag_context:
            parts.append("\nExpert Advice:")
            parts.append(rag_context.strip())
            parts.append("")

        greeting = "" if self.chat_history else "Begin with a friendly greeting."
        parts.append(
            f"User Query: {user_query}\n"
            f"Instructions: {greeting} Provide a detailed, engaging response in plain text.\n"
            "When recommending wines:\n"
            "- Present as a numbered list with clear formatting\n"
            "- For each wine include: Name, Price, Region, Description\n"
            "- Add serving temperature and food pairing suggestions\n"
            "- Include storage advice when relevant\n"
            "- End with a personalized closing message\n"
            "Format in clear paragraphs with line breaks for readability."
        )
        return "\n".join(parts)

    def generate_response(self, user_query):
        try:
            # Run NLU on the user's query.
            nlu_result = process_query(user_query)
            logger.info(f"Processed NLU result: {nlu_result}")

            # Check for follow-up questions.
            if self._is_followup_question(user_query):
                return self._handle_followup(user_query)

            # Extract price preferences from query.
            pref_min, pref_max = extract_price_preferences(user_query)
            price_min, price_max = (None, None)
            if pref_min is not None or pref_max is not None:
                price_min = pref_min
                price_max = pref_max

            # Start with a copy of the full wine DataFrame.
            filtered_df = wine_df.copy()

            # Apply price filtering.
            if "Price" in filtered_df.columns:
                if price_max is not None:
                    filtered_df = filtered_df[filtered_df["Price"] <= price_max]
                if price_min is not None:
                    filtered_df = filtered_df[filtered_df["Price"] >= price_min]

            # --- Apply NLU-derived filters ---
            for ent in nlu_result["entities_pipeline"] + nlu_result["entities_custom"]:
                label = ent["label"].upper()  # e.g., "WINE_TYPE", "REGION"
                word = ent["word"]
                if label == "WINE_TYPE":
                    filtered_df = filtered_df[
                        filtered_df["Wine Type"].str.contains(word, case=False, na=False)
                    ]
                elif label == "REGION":
                    filtered_df = filtered_df[
                        filtered_df["Region"].str.contains(word, case=False, na=False)
                    ]
                elif label == "GRAPE_VARIETY":
                    filtered_df = filtered_df[
                        filtered_df["Grape Type List"].str.contains(word, case=False, na=False)
                    ]
                elif label == "FOOD_PAIRING":
                    filtered_df = filtered_df[
                        filtered_df["Food Pairing"].str.contains(word, case=False, na=False)
                    ]

            logger.info(f"After NLU filtering, {len(filtered_df)} wines remain.")

            # Run hybrid search over the filtered data.
            results = []
            if not filtered_df.empty:
                results = hybrid_search(
                    user_query,
                    nlu_result,
                    self.embedder,       # model
                    self.faiss_index,    # faiss_index
                    filtered_df,         # wine_df
                    k=RESULT_LIMIT
                )

            # Construct the context list.
            wine_context_list = []
            if results and nlu_result.get("intent") in {"recommend_wine", "food_pairing"}:
                wine_context_list = [
                    ("Best Match" if i == 0 else f"Alternative {i}", wine)
                    for i, wine in enumerate(results[:3])
                ]
            self.last_recommendations = wine_context_list

            rag_context = ""
            if nlu_result.get("intent") == "food_pairing":
                rag_context = build_rag_context_block(user_query, top_k=3)

            prompt = self._build_prompt(wine_context_list, user_query, rag_context)
            logger.debug(f"Generated prompt (first 500 chars): {prompt[:500]}...")

            response = self._call_openai(prompt)
            self._update_chat_history(user_query, response)
            return response

        except Exception as e:
            logger.error(f"Response generation failed: {e}", exc_info=True)
            return "I'm having trouble generating a response. Please try again later."

    def _call_openai(self, prompt):
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a professional sommelier. Provide detailed, accurate wine recommendations with clear formatting."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1200,
                top_p=0.9
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return "I couldn't connect to the AI service. Please try again."

    def _update_chat_history(self, query, response):
        self.chat_history.append({
            "user": query,
            "assistant": response,
            "timestamp": datetime.now().isoformat()
        })
        if len(self.chat_history) > 5:
            self.chat_history.pop(0)

    def _is_followup_question(self, query):
        if not self.last_recommendations:
            return False
        followup_patterns = [
            r"tell me more(?: about)?(?: the)?(?: first|second|third|\d+)?",
            r"more info(?: on)?(?: the)?(?: first|second|third|\d+)?",
            r"can you elaborate(?: on)?(?: the)?(?: first|second|third|\d+)?",
            r"explain(?: the)?(?: first|second|third)?",
            r"what makes(?: the)?(?: first|second|third)? (?:special|good|better)"
        ]
        return any(re.search(p, query.lower()) for p in followup_patterns)

    def _handle_followup(self, query):
        wine_idx = 0
        query_lower = query.lower()
        if "second" in query_lower:
            wine_idx = 1
        elif "third" in query_lower:
            wine_idx = 2
        else:
            num_match = re.search(r'\d+', query)
            if num_match:
                wine_idx = min(int(num_match.group()) - 1, len(self.last_recommendations) - 1)

        if wine_idx >= len(self.last_recommendations):
            return "I don't have that recommendation saved. Could you please ask again?"

        wine = self.last_recommendations[wine_idx][1]
        prompt = (
            f"Provide detailed sommelier information about this wine:\n\n"
            f"Name: {wine.get('Wine Name', 'Unknown')}\n"
            f"Type: {wine.get('Primary Type', wine.get('Wine Type', 'Unknown'))}\n"
            f"Region: {wine.get('Region', 'Unknown')}\n"
        )
        price = wine.get("Price", 0)
        price_str = ""
        try:
            price_str = f"${float(price):.2f}"
        except Exception:
            price_str = str(price)
        prompt += f"Price: {price_str}\n"
        prompt += (
            f"Alcohol: {wine.get('Alcohol Content (%)', 'Unknown')}%\n"
            f"Food Pairings: {wine.get('Food Pairing', 'Not specified')}\n"
            f"Description: {wine.get('Wine Description 1', 'No description available')}\n\n"
            "Include detailed tasting notes, ideal serving temperature, decanting recommendations, "
            "cellaring potential, and additional food pairing suggestions. "
            "Format your response in plain text with clear paragraphs."
        )

        response = self._call_openai(prompt)
        self._update_chat_history(query, response)
        return response

if __name__ == "__main__":
    logger.info("Initializing Wine Recommendation Assistant...")
    try:
        assistant = ResponseGeneratorOpenAI()
        logger.info("Assistant ready! Type 'exit' to quit.")
        while True:
            try:
                query = input("\nYou: ").strip()
                if query.lower() in ('exit', 'quit'):
                    break
                if not query:
                    continue
                logger.info(f"Processing query: {query}")
                response = assistant.generate_response(query)
                print(f"\nSommelier: {response}")
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Error processing query: {e}")
                print("Sorry, I encountered an error. Please try again.")
    except Exception as e:
        logger.error(f"Failed to initialize assistant: {e}")
        print("Failed to initialize the assistant. Please check the logs.")
