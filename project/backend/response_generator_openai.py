# project/backend/response_generator_openai.py
import os
import sys
import json
import logging
import re
import pandas as pd
from datetime import datetime

import nest_asyncio
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

# -----------------------------
# Load environment variables
# -----------------------------
# IMPORTANT: load from repo root .env (not project/backend/.env)
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
env_path = os.path.join(repo_root, ".env")
load_dotenv(dotenv_path=env_path)

# -----------------------------
# Extend PYTHONPATH (legacy structure)
# -----------------------------
rr_path = os.path.join(repo_root, "retrieval_recommendation_engine")
rag_path = os.path.join(repo_root, "RAGs")
nlu_path = os.path.join(repo_root, "nlu")
for p in [rr_path, rag_path, nlu_path]:
    if p not in sys.path:
        sys.path.append(p)

# -----------------------------
# NLU import (safe)
# -----------------------------
try:
    # preferred import (package style)
    from nlu.ner import process_query
except Exception:
    try:
        # fallback if your PYTHONPATH hack is relied on
        from ner import process_query  # type: ignore
    except Exception:
        process_query = None
        logger.warning("NLU process_query import failed; running with fallback intent/entities.")

# Import necessary functions from the modules.
from rr_engine import (
    hybrid_search,
    compute_embeddings,
    build_faiss_index,
    RESULT_LIMIT,
    EMBEDDING_MODEL_NAME,
    extract_price_preferences,
    load_wines_from_mongo,
)
from rag_module import build_rag_context_block

# -----------------------------
# OpenAI Client (don’t hard crash)
# -----------------------------
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key) if api_key else None

# Frontend base URL for wine details (set in env for deployment)
BASE_WINE_DETAIL_URL = os.getenv("BASE_WINE_DETAIL_URL", "http://localhost:3000/wine-details/").rstrip("/") + "/"

# Load wine data once at import (ok for now; later we can lazy-load/caching)
try:
    wines_list = load_wines_from_mongo()
    wine_df = pd.DataFrame(wines_list)
except Exception as e:
    logger.error(f"Failed to load wine data from MongoDB: {e}")
    # Keep server alive; endpoints using this will handle empty df
    wine_df = pd.DataFrame()

class ResponseGeneratorOpenAI:
    def __init__(self, model: str = None):
        # Default model can be set via env
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.temperature = float(os.getenv("OPENAI_TEMPERATURE", "0.7"))

        # Embedder + FAISS (only if we have data)
        self.embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

        if wine_df.empty:
            self.embeddings = []
            self.faiss_index = None
        else:
            self.embeddings = compute_embeddings(wine_df, self.embedder)
            self.faiss_index = build_faiss_index(self.embeddings)

        self.chat_history = []
        self.last_recommendations = []

        if not wine_df.empty and len(self.embeddings) != len(wine_df):
            logger.error("Embeddings length doesn't match dataframe")
            raise ValueError("Embedding dimension mismatch")

    def _build_prompt(self, wine_context_list, user_query, rag_context=""):
        parts = []

        # Previous conversation
        if self.chat_history:
            parts.append("Previous Conversation:")
            for turn in self.chat_history[-4:]:
                parts.append(f"User: {turn['user']}")
                parts.append(f"Assistant: {turn['assistant']}")
            parts.append("")

        # Wine recommendation details
        if wine_context_list:
            for label, wine in wine_context_list:
                wine_info = f"\n[{label}]\n"
                wine_info += f"Wine Name: {wine.get('Wine Name', '')}\n"
                wine_info += f"Type: {wine.get('Primary Type', wine.get('Wine Type', ''))}\n"
                wine_info += f"Region: {wine.get('Region', '')}\n"

                price = wine.get("Price", 0)
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

        # RAG expert advice
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

    def generate_response(self, user_query: str) -> str:
        try:
            # 1) NLU (safe fallback)
            if process_query:
                nlu_result = process_query(user_query)
            else:
                nlu_result = {"intent": "recommend_wine", "entities_pipeline": []}

            logger.info(f"Processed NLU result: {nlu_result}")

            # Follow-up handling
            if self._is_followup_question(user_query):
                return self._handle_followup(user_query)

            # 2) Price preferences
            pref_min, pref_max = extract_price_preferences(user_query)
            price_min, price_max = (pref_min, pref_max)

            # 3) Filter dataframe
            filtered_df = wine_df.copy()

            if not filtered_df.empty and "Price" in filtered_df.columns:
                if price_max is not None:
                    filtered_df = filtered_df[filtered_df["Price"] <= price_max]
                if price_min is not None:
                    filtered_df = filtered_df[filtered_df["Price"] >= price_min]

            # 4) Apply NLU-derived filters (now only entities_pipeline)
            entities = nlu_result.get("entities_pipeline", []) or []
            if not filtered_df.empty:
                for ent in entities:
                    label = str(ent.get("label", "")).upper()
                    word = str(ent.get("word", ""))

                    if not word:
                        continue

                    if label == "WINE_TYPE" and "Wine Type" in filtered_df.columns:
                        filtered_df = filtered_df[filtered_df["Wine Type"].str.contains(word, case=False, na=False)]
                    elif label == "REGION" and "Region" in filtered_df.columns:
                        filtered_df = filtered_df[filtered_df["Region"].str.contains(word, case=False, na=False)]
                    elif label == "GRAPE_VARIETY" and "Grape Type List" in filtered_df.columns:
                        filtered_df = filtered_df[filtered_df["Grape Type List"].str.contains(word, case=False, na=False)]
                    elif label == "FOOD_PAIRING" and "Food Pairing" in filtered_df.columns:
                        filtered_df = filtered_df[filtered_df["Food Pairing"].str.contains(word, case=False, na=False)]

            logger.info(f"After filtering, {len(filtered_df)} wines remain.")

            # 5) Search
            results = []
            if not filtered_df.empty and self.faiss_index is not None:
                results = hybrid_search(
                    user_query,
                    nlu_result,
                    self.embedder,
                    self.faiss_index,
                    filtered_df,
                    k=RESULT_LIMIT,
                )

            # 6) Context formatting
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

            # 7) OpenAI call
            response = self._call_openai(prompt)
            self._update_chat_history(user_query, response)
            return response

        except Exception as e:
            logger.error(f"Response generation failed: {e}", exc_info=True)
            return "I'm having trouble generating a response right now. Please try again."

    def _call_openai(self, prompt: str) -> str:
        if client is None:
            return "AI service is not configured (missing OPENAI_API_KEY)."

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": os.getenv(
                            "WINE_ASSISTANT_SYSTEM_PROMPT",
                            "You are a professional sommelier. Provide detailed, accurate wine recommendations with clear formatting.",
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
                max_tokens=1200,
                top_p=0.9,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return "I couldn't connect to the AI service. Please try again."

    def _update_chat_history(self, query: str, response: str):
        self.chat_history.append(
            {"user": query, "assistant": response, "timestamp": datetime.now().isoformat()}
        )
        if len(self.chat_history) > 5:
            self.chat_history.pop(0)

    def _is_followup_question(self, query: str) -> bool:
        if not self.last_recommendations:
            return False
        followup_patterns = [
            r"tell me more(?: about)?(?: the)?(?: first|second|third|\d+)?",
            r"more info(?: on)?(?: the)?(?: first|second|third|\d+)?",
            r"can you elaborate(?: on)?(?: the)?(?: first|second|third|\d+)?",
            r"explain(?: the)?(?: first|second|third)?",
            r"what makes(?: the)?(?: first|second|third)? (?:special|good|better)",
        ]
        return any(re.search(p, query.lower()) for p in followup_patterns)

    def _handle_followup(self, query: str) -> str:
        wine_idx = 0
        query_lower = query.lower()
        if "second" in query_lower:
            wine_idx = 1
        elif "third" in query_lower:
            wine_idx = 2
        else:
            num_match = re.search(r"\d+", query)
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
        try:
            price_str = f"${float(price):.2f}"
        except Exception:
            price_str = str(price)

        prompt += (
            f"Price: {price_str}\n"
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