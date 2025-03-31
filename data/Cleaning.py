import os
import pandas as pd
import re
import warnings
import ast
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from tqdm import tqdm
import random

warnings.simplefilter('ignore')

# ------------------ Custom Rounding Function for Alcohol Content ------------------ #
def custom_round_alcohol(value):
    """
    Custom round the alcohol content value:
      - Fraction < 0.35 -> round down to .00
      - 0.35 <= Fraction < 0.70 -> round to .50
      - Fraction >= 0.70 -> round up to the next integer (with .00)
    """
    base = int(value)
    frac = value - base
    if frac < 0.35:
        return base + 0.00
    elif frac < 0.70:
        return base + 0.50
    else:
        return base + 1.00

# ------------------ Data Cleaning Functions ------------------ #
def load_and_clean_data(input_file_path):
    """
    Load the dataset, drop duplicate product links, and return the cleaned DataFrame.
    """
    df = pd.read_csv(input_file_path, on_bad_lines='skip')
    df_cleaned = df.drop_duplicates("Product Link", keep="first")
    return df_cleaned

def extract_vintage(wine_name):
    match = re.search(r'(19|20)\d{2}', wine_name)
    return match.group(0) if match else None

def clean_text(text, handle_missing=True):
    if pd.isna(text) and handle_missing:
        return 'unknown'
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def clean_grape_type(text):
    """
    Clean the grape type string and convert it into a list of individual grape types.
    """
    if pd.isna(text):
        return ['unknown']
    text = re.sub(r'\d+%', '', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r',\s*$', '', text)
    if ',' in text:
        return [s.strip() for s in text.split(',') if s.strip()]
    else:
        return [text] if text else ['unknown']

def clean_allergens(text):
    if pd.isna(text) or text in ['Unknown', 'N/A']:
        return 'no allergen information'
    return text.lower()

def clean_food_item(item: str):
    item = item.strip().lower()
    pattern = r'(.*?)\s*\((.*?)\)'
    match = re.match(pattern, item)
    if match:
        main_item = match.group(1).strip()
        sub_items_str = match.group(2).strip()
        result = []
        if main_item:
            result.append(main_item)
        sub_items = [s.strip() for s in sub_items_str.split(',')]
        result.extend(sub_items)
        return result
    else:
        return [item]

def clean_food_pairing(food_data):
    if pd.isna(food_data):
        return ["no food pairing information"]
    if isinstance(food_data, str):
        food_data_str = food_data.strip()
        if food_data_str.startswith('[') and food_data_str.endswith(']'):
            try:
                parsed_data = ast.literal_eval(food_data_str)
                food_list = parsed_data if isinstance(parsed_data, list) else [food_data_str]
            except (SyntaxError, ValueError):
                food_list = [f.strip() for f in food_data_str.split(',') if f.strip()]
        else:
            food_list = [f.strip() for f in food_data_str.split(',') if f.strip()]
    elif isinstance(food_data, list):
        food_list = food_data
    else:
        return ["no food pairing information"]
    
    cleaned = []
    for raw_item in food_list:
        if not isinstance(raw_item, str) or not raw_item.strip():
            continue
        sub_items = clean_food_item(raw_item)
        final_sub_items = [re.sub(r'[^a-zA-Z\s]', '', si).strip() for si in sub_items if si.strip()]
        cleaned.extend(final_sub_items)
    return cleaned if cleaned else ["no food pairing information"]

def clean_price(price):
    if pd.isna(price) or price in ['Unknown', 'N/A']:
        return None
    price = re.sub(r'[^\d.]', '', price)
    return float(price)

def clean_alcohol_content(text):
    if pd.isna(text) or text in ['Unknown', 'N/A']:
        return None
    text = re.sub(r'[^\d.]', '', text)
    return float(text)

# ------------------ Primary Type Extraction ------------------ #
def extract_primary_type_from_wine_name(wine_name):
    """
    Extract the primary wine type from the wine name using a comprehensive list of known types.
    Do NOT remove the primary type from the original wine name.
    """
    primary_types = [
        "aglianico",
        "albariño", "albarino", 
        "alvarinho",
        "arinto",
        "baga",
        "barbera",
        "blaufränkisch", "blaufrankisch",
        "bonarda",
        "cabernet sauvignon",
        "cariñena", "carignan",
        "carménère", "carmenere",
        "castelão", "castelao",
        "champagne",
        "chardonnay",
        "chenin blanc",
        "corvina",
        "dolcetto",
        "dornfelder",
        "fernão pires", "fernao pires",
        "gamay",
        "garnacha", "grenache",
        "godello",
        "grauburgunder", "pinot gris",
        "grüner veltliner", "gruner veltliner",
        "macabeo", "viura",
        "malbec",
        "mencia",
        "merlot",
        "monastrell", "mourvèdre", "mourvedre",
        "montepulciano",
        "müller-thurgau", "muller-thurgau",
        "muscat", "moscato",
        "nebbiolo",
        "nero d'avola",
        "pais",
        "palomino",
        "petite sirah", "petit syrah", "durif",
        "pinot grigio", "pinot noir",
        "prosecco",
        "riesling",
        "rosé", "rose",
        "sangiovese",
        "sauvignon blanc",
        "scheurebe",
        "sémillon", "semillon",
        "shiraz", "syrah",
        "silvaner", "sylvaner",
        "tempranillo", "tinta roriz",
        "torrontés", "torrontes",
        "touriga nacional",
        "trebbiano", "ugni blanc",
        "trincadeira",
        "verdejo",
        "vernaccia",
        "viognier",
        "weißburgunder", "weissburgunder", "pinot blanc",
        "welschriesling",
        "zinfandel",
        "zweigelt"
    ]
    found = None
    for pt in primary_types:
        if re.search(re.escape(pt), wine_name, re.IGNORECASE):
            found = pt
            break
    if found is None:
        found = "unknown"
    return found

def determine_primary_type_from_grape(grape_str):
    """
    Given the raw grape type string (before cleaning), determine the primary type.
    If percentages are specified (e.g. '94%Tinto Fino,6%Merlot'), choose the grape with the highest percentage.
    Otherwise, choose the first grape type.
    """
    grapes = [g.strip() for g in grape_str.split(',') if g.strip()]
    best = None
    best_pct = 0
    for g in grapes:
        match = re.match(r'(\d+)%\s*(.*)', g, re.IGNORECASE)
        if match:
            pct = int(match.group(1))
            grape_type = match.group(2).strip().lower()
            if pct > best_pct:
                best_pct = pct
                best = grape_type
        else:
            if best is None:
                best = g.lower()
    return best if best else "unknown"

# ------------------ Duplicate Winery Removal ------------------ #
def remove_duplicate_winery(wine_name, winery):
    """
    Remove duplicate occurrences of the winery name at the beginning of the wine name.
    Ensures that only one instance of the winery name is present with a space following it.
    """
    pattern = re.compile(r"^(%s)(\s+\1)+" % re.escape(winery), re.IGNORECASE)
    new_name = pattern.sub(r"\1", wine_name)
    if new_name.lower().startswith(winery.lower()) and not new_name[len(winery):].startswith(" "):
        new_name = new_name[:len(winery)] + " " + new_name[len(winery):]
    return new_name

# ------------------ DATA PROCESSING ------------------ #
def process_data(df, output_file_path):
    """
    Process and clean the dataset:
      - Clean text columns, price, allergens, food pairing, alcohol content, etc.
      - Extract vintage from the Wine Name.
      - Clean text for specified columns.
      - Remove duplicate winery names from the Wine Name column.
      - Extract primary wine type from Wine Name and add it as a new column.
      - Retain the original Wine Name (with the winery) in the "Wine Name" column.
      - Clean grape type into a list ("Grape Type List") and also store as a comma-separated string ("Grape Type").
      - If primary type is 'unknown', determine it from the raw grape type.
      - Round off "Gentle to Fizzy" to 2 decimals.
      - Save the cleaned data.
    """
    df['Vintage'] = df['Wine Name'].apply(extract_vintage)
    # Remove vintage from Wine Name for display but keep original winery intact.
    df['Wine Name'] = df['Wine Name'].apply(lambda x: re.sub(r'(19|20)\d{2}', '', x).strip())
    
    # Clean text columns.
    for col in ['Wine Name', 'Winery', 'Country', 'Region', 'Wine Type']:
        df[col] = df[col].apply(clean_text)
    
    # Remove duplicate winery names from Wine Name.
    df['Wine Name'] = df.apply(lambda row: remove_duplicate_winery(row['Wine Name'], row['Winery']), axis=1)
    
    # Extract primary wine type from Wine Name and add a new column.
    df['primary type'] = df['Wine Name'].apply(extract_primary_type_from_wine_name)
    
    # Clean grape type and store two versions.
    grape_cleaned = df['Grape Type'].apply(clean_grape_type)
    df['Grape Type List'] = grape_cleaned
    df['Grape Type'] = grape_cleaned.apply(lambda x: ", ".join(x))
    
    # If primary type is still 'unknown', determine it from the raw grape type.
    df['primary type'] = df.apply(lambda row: determine_primary_type_from_grape(row['Grape Type'])
                                  if row['primary type'] == "unknown" else row['primary type'], axis=1)
    
    df['Price'] = df['Price'].apply(clean_price)
    
    df['Both Descriptions Null'] = df[['Wine Description 1', 'Wine Description 2']].isnull().all(axis=1)
    print(f"Rows with both descriptions null: {df['Both Descriptions Null'].sum()}")
    
    df['Wine Description 1'] = df['Wine Description 1'].apply(clean_text)
    df['Wine Description 2'] = df['Wine Description 2'].apply(clean_text)
    df['Allergens'] = df['Allergens'].apply(clean_allergens)
    df['Food Pairing'] = df['Food Pairing'].apply(clean_food_pairing)
    df['Alcohol Content'] = df['Alcohol Content'].apply(clean_alcohol_content)
    df.rename(columns={'Alcohol Content': 'Alcohol Content (%)'}, inplace=True)
    
    df.rename(columns={
        'Light': 'Light to Bold',
        'Smooth': 'Smooth to Tannic',
        'Dry': 'Dry to Sweet',
        'Soft': 'Soft to Acidic',
        'Unnamed: 21': 'Gentle to Fizzy'
    }, inplace=True)
    
    df['Vintage'].fillna('Non-Vintage', inplace=True)
    
    # Fill "Gentle to Fizzy": if Wine Type is not sparkling -> fill with 0; if sparkling -> fill with sparkling mean.
    if "Gentle to Fizzy" in df.columns:
        is_sparkling = df["Wine Type"].str.contains("sparkling", case=False, na=False)
        df.loc[~is_sparkling, "Gentle to Fizzy"] = df.loc[~is_sparkling, "Gentle to Fizzy"].fillna(0)
        sparkling_mean = df.loc[is_sparkling, "Gentle to Fizzy"].mean()
        df.loc[is_sparkling, "Gentle to Fizzy"] = df.loc[is_sparkling, "Gentle to Fizzy"].fillna(sparkling_mean)
        df["Gentle to Fizzy"] = df["Gentle to Fizzy"].round(2)
    
    df.to_csv(output_file_path, index=False)
    print(f"Cleaned data saved to {output_file_path}")
    return df

# ------------------ XGBoost Imputation for Taste Characteristics ------------------ #
def fill_taste_with_xgboost(df):
    target_cols = ["Light to Bold", "Smooth to Tannic", "Dry to Sweet", "Soft to Acidic"]
    print("\nMissing taste values BEFORE XGBoost imputation:")
    print(df[target_cols].isnull().sum())
    
    df_train = df.dropna(subset=target_cols).copy()
    missing_mask = df[target_cols].isnull().any(axis=1)
    print(f"Rows with missing taste data: {missing_mask.sum()}")
    
    if missing_mask.sum() == 0:
        print("No missing taste characteristics to fill with XGBoost.")
        return df
    
    df_missing = df[missing_mask].copy()
    
    num_features = ["Price", "Rating", "Alcohol Content (%)"]
    cat_features = ["Wine Type", "Grape Type"]
    
    X_train = df_train[num_features + cat_features]
    y_train = df_train[target_cols]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
        ]
    )
    
    xgb_reg = xgb.XGBRegressor(objective="reg:squarederror", random_state=42)
    multi_output_regressor = MultiOutputRegressor(xgb_reg)
    
    pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("multi_output_regressor", multi_output_regressor)
    ])
    
    pipe.fit(X_train, y_train)
    
    X_missing = df_missing[num_features + cat_features]
    y_missing_pred = pipe.predict(X_missing)
    y_missing_pred = np.round(y_missing_pred, 2)
    
    for i, col in enumerate(target_cols):
        df.loc[missing_mask, col] = y_missing_pred[:, i]
    
    print("Missing taste characteristics filled using XGBoost (rounded to 2 decimals).")
    print("Missing taste values AFTER XGBoost imputation:")
    print(df[target_cols].isnull().sum())
    
    return df

# ------------------ XGBoost Imputation for Alcohol Content (%) ------------------ #
def fill_alcohol_with_xgboost(df):
    target = "Alcohol Content (%)"
    print("\nMissing Alcohol Content (%) BEFORE XGBoost imputation:")
    print(df[target].isnull().sum())
    
    df_train = df[df[target].notnull()].copy()
    missing_mask = df[target].isnull()
    print(f"Rows with missing Alcohol Content (%): {missing_mask.sum()}")
    
    if missing_mask.sum() == 0:
        print("No missing Alcohol Content (%) to fill with XGBoost.")
        return df
    
    df_missing = df[missing_mask].copy()
    
    num_features = ["Price", "Rating", "Light to Bold", "Smooth to Tannic", "Dry to Sweet", "Soft to Acidic"]
    cat_features = ["Wine Type", "Grape Type"]
    
    X_train = df_train[num_features + cat_features]
    y_train = df_train[target]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
        ]
    )
    
    xgb_reg = xgb.XGBRegressor(objective="reg:squarederror", random_state=42)
    
    pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", xgb_reg)
    ])
    
    pipe.fit(X_train, y_train)
    
    X_missing = df_missing[num_features + cat_features]
    y_missing_pred = pipe.predict(X_missing)
    
    y_missing_pred_rounded = [custom_round_alcohol(val) for val in y_missing_pred]
    
    df.loc[missing_mask, target] = y_missing_pred_rounded
    print("Missing Alcohol Content (%) filled using XGBoost and custom rounding.")
    print("Missing Alcohol Content (%) AFTER XGBoost imputation:")
    print(df[target].isnull().sum())
    
    return df

# ------------------ Main Execution ------------------ #
if __name__ == "__main__":
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Build file paths relative to the script's location
    input_file_path = os.path.join(script_dir, "wine_data_with_images.csv")
    cleaned_output_path = os.path.join(script_dir, "cleaned_wine_data.csv")
    final_output_path = os.path.join(script_dir, "final_wine_data.csv")
    
    # 1. Load and clean data.
    df_cleaned = load_and_clean_data(input_file_path)
    
    # 2. Process data: cleaning, feature extraction, etc.
    df_processed = process_data(df_cleaned, cleaned_output_path)
    
    # 3. Fill missing taste characteristics using XGBoost.
    df_with_taste = fill_taste_with_xgboost(df_processed)
    
    # 4. Fill missing Alcohol Content (%) using XGBoost and custom rounding.
    df_with_alcohol = fill_alcohol_with_xgboost(df_with_taste)
    
    # 5. Save only the final DataFrame.
    df_with_alcohol.to_csv(final_output_path, index=False)
    print(f"\nFinal Data saved to '{final_output_path}'.")
    
    # Print info of the final cleaned DataFrame.
    print("\n=== Final DataFrame Info ===")
    print(df_with_alcohol.info())

print("Updated cleaning.py module loaded successfully.")