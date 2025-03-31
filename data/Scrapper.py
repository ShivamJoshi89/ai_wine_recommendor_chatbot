import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import time
import os
import random
from tqdm import tqdm

# Configurations
BATCH_SIZE = 100  # Process 100 links at a time
START_INDEX = 0   # Resume from last processed index
image_folder = "wine_images"
os.makedirs(image_folder, exist_ok=True)  # Create folder if it doesn't exist

# User-Agent headers
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36',
]

# Function to extract score from taste characteristics
def extract_score(style_string):
    left_match = re.search(r'left:(\d+\.?\d*)%', style_string)
    width_match = re.search(r'width:(\d+\.?\d*)%', style_string)
    
    if left_match and width_match:
        left = float(left_match.group(1))
        width = float(width_match.group(1))
        position = left + (width / 2)
        return round(position / 100, 2)  # Normalize score
    return None  # Return None if parsing fails

# Extract taste characteristics
def extract_taste_characteristics(product_link, headers):
    try:
        response = requests.get(product_link, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, "html.parser")

        taste_scores = {}
        characteristic_rows = soup.find_all("tr", class_="tasteStructure__tasteCharacteristic--jLtsE")

        for row in characteristic_rows:
            # Extract the characteristic name (e.g., "Acidity", "Tannin", etc.)
            characteristic_name = row.find("div", class_="tasteStructure__property--CLNl_").text.strip()
            
            # Extract the progress bar style for the characteristic
            progress_bar = row.find("span", class_="indicatorBar__progress--3aXLX")
            if progress_bar and "style" in progress_bar.attrs:
                score = extract_score(progress_bar["style"])
                taste_scores[characteristic_name] = score

        taste_scores["Product Link"] = product_link
        return taste_scores
    except Exception as e:
        print(f"Error extracting taste characteristics for {product_link}: {e}")
        return {"Product Link": product_link}

# Download images
def download_image(image_url, wine_name):
    if image_url == "Image Not Found":
        return "No Image Available"

    image_name = re.sub(r'[^a-zA-Z0-9]', '_', wine_name) + ".jpg"
    image_path = os.path.join(image_folder, image_name)
    
    try:
        response = requests.get(image_url, stream=True, timeout=10)
        if response.status_code == 200:
            with open(image_path, 'wb') as file:
                for chunk in response.iter_content(1024):
                    file.write(chunk)
            return image_path
    except Exception as e:
        print(f"Error downloading {image_url}: {e}")
    
    return "Download Failed"

# Base URL (First Page)
base_url = "https://www.vivino.com/explore?e=eJw1zUEOgjAQheHbzJoKCavZeQPjyhhTSjFVCqRTUG5v5bWr9yedfPWBFXk3saor8vrLqq3I7LwKGb5ezrSk9-fAmw7ORj3SHHrurRiau507K_GxOPMW-sTbPZ0ec8LUmAbT5rcm8ZLi_8MROpRYS8QcZswxlJve5nDlZilhhYSnF8EGDBUkPGCQwMA4gB8oEVuT"

# Initialize variables
current_url = base_url  # Start with the first page
all_product_links = []  # Store all scraped product links

try:
    # Loop through pages
    MAX_PAGES = 71  # Set a limit to avoid infinite scraping
    page_count = 0

    while current_url and page_count < MAX_PAGES:
        print(f"Scraping Page {page_count + 1}: {current_url}")
        
        # Fetch the page content with a random user agent
        headers = {'User-Agent': random.choice(USER_AGENTS)}
        response = requests.get(current_url, headers=headers)
        soup = BeautifulSoup(response.content, "html.parser")
        
        # Extract all wine product links
        links = soup.find_all("a", class_="anchor_anchor__m8Qi- wineCard__cardLink--3F_uB wineCard__notSuperCard--268Xo")
        base_url = "https://www.vivino.com"
        
        # Store valid links
        for link in links:
            href = link.get('href')
            if href:
                all_product_links.append(base_url + href)

        # Find the "Next" button
        next_button = soup.find("span", class_="vivino-mui-14ngluw-componentChildren", string="Next")

        # If "Next" button exists, get the next page URL
        if next_button:
            next_page = next_button.find_parent("a")
            if next_page and "href" in next_page.attrs:
                current_url = base_url + next_page["href"]
            else:
                break  # No more pages, exit loop
        else:
            break  # No more pages, exit loop

        page_count += 1
        time.sleep(2)

except KeyboardInterrupt:
    print("\n🛑 Scraping stopped by user. Saving collected data...")
    df = pd.DataFrame(all_product_links, columns=["Product Links"])
    df.to_csv("partial_wine_data.csv", index=False)
    print("✅ Partial data saved to 'partial_wine_data.csv'.")
    exit()

# Remove duplicate product links
all_product_links = list(set(all_product_links))  # Remove duplicates
print(f"Total Unique Products Scraped: {len(all_product_links)}")

# Step 1: Fetch the main webpage
webpage = requests.get(base_url, headers={'User-Agent': random.choice(USER_AGENTS)})
soup = BeautifulSoup(webpage.content, "html.parser")

# Step 2: Find all product links
product_links = all_product_links

# Load existing data to resume progress
if os.path.exists("wine_data_with_images.csv"):
    existing_df = pd.read_csv("wine_data_with_images.csv")
    processed_links = set(existing_df["Product Link"])
    product_links = [link for link in all_product_links if link not in processed_links]
else:
    product_links = all_product_links
    
# Total batch count
total_batches = len(product_links) // BATCH_SIZE + (1 if len(product_links) % BATCH_SIZE else 0)

# Initialize batch_wine_data outside the loop
batch_wine_data = []

# ✅ Main Progress Bar (Tracks overall batch progress)
with tqdm(total=total_batches, desc="Total Progress", unit="batch") as main_pbar:

    # Process in batches of 100
    for start in range(START_INDEX, len(product_links), BATCH_SIZE):
        batch_links = product_links[start:start + BATCH_SIZE]
        print(f"\nProcessing Batch {start // BATCH_SIZE + 1}/{total_batches}: {len(batch_links)} links")

        # ✅ Batch Progress Bar (Tracks per-wine progress in each batch)
        with tqdm(total=len(batch_links), desc="Batch Progress", unit="wine") as batch_pbar:

            for product_link in batch_links:
                try:
                    headers = {'User-Agent': random.choice(USER_AGENTS)}
                    response = requests.get(product_link, headers=headers, timeout=10)
                    soup = BeautifulSoup(response.content, "html.parser")

                    # Extract data fields
                    winery = soup.find("a", class_="wineHeadline-module__winery--3b7KA")
                    winery_name = winery.get_text(strip=True) if winery else "Winery Not Found"

                    wine_element = soup.find("a", class_="wineHeadline-module__link--G1mKm")
                    wine_name_with_vintage = wine_element.find_parent().get_text(strip=True) if wine_element else "Wine Not Found"

                    price_element = soup.find("span", class_="purchaseAvailability__currentPrice--3mO4u")
                    price = price_element.get_text(strip=True) if price_element else "Price Not Available"

                    rating_element = soup.find("div", class_="vivinoRating_averageValue__uDdPM")
                    rating = rating_element.get_text(strip=True) if rating_element else "Rating Not Found"

                    country_element = soup.find("a", class_="anchor_anchor__m8Qi-", attrs={"data-cy": "breadcrumb-country"})
                    country = country_element.get_text(strip=True) if country_element else "Country Not Found"

                    region_element = soup.find("a", class_="anchor_anchor__m8Qi-", attrs={"data-cy": "breadcrumb-region"})
                    region = region_element.get_text(strip=True) if region_element else "Region Not Found"

                    wine_type_element = soup.find("a", class_="anchor_anchor__m8Qi-", attrs={"data-cy": "breadcrumb-winetype"})
                    wine_type = wine_type_element.get_text(strip=True) if wine_type_element else "Wine Type Not Found"

                    # Extract Wine Facts Table
                    wine_facts = {}
                    wine_facts_table = soup.find("table", class_="wineFacts__wineFacts--2Ih8B")

                    if wine_facts_table:
                        for row in wine_facts_table.find_all("tr"):
                            key_element = row.find("th")
                            value_element = row.find("td")
                            if key_element and value_element:
                                key = key_element.get_text(strip=True)
                                value = value_element.get_text(strip=True)
                                wine_facts[key] = value
                    grape_type = wine_facts.get("Grapes", "Grape Type Not Found")

                    wine_description_1 = wine_facts.get("Wine description", "N/A")

                    editor_note_element = soup.find("div", class_="fullEditorNote__editorsNote--1sTVM")
                    wine_description_2 = (editor_note_element.find("div", class_="readMoreText__text--1SNh4").get_text(" ", strip=True)
                                  if editor_note_element else "N/A")

                    # Extract Food Pairings
                    food_pairing_elements = soup.find_all("a", attrs={"class": "anchor_anchor__m8Qi- foodPairing__imageContainer--2CtYR"})
                    food_pairing_html = str(food_pairing_elements)
                    pattern = r'<div[^>]*>(.*?)<\/div>'
                    food_pairings = re.findall(pattern, food_pairing_html)
                    food_pairing_text = "; ".join(filter(None, [food.strip() for food in food_pairings])) if food_pairings else "No Food Pairing Available"

                    image_url = "Image Not Found"
                    img_tags = soup.find_all("img", class_="wineLabel-module__image--3HOnd")
                    if img_tags:
                        for img_tag in img_tags:
                            if "src" in img_tag.attrs:
                                src = img_tag["src"]
                                image_url = "https:" + src if src.startswith("//") else src
                                break

                    # Download image
                    image_path = download_image(image_url, wine_name_with_vintage)

                    # Extract taste characteristics
                    taste_scores = extract_taste_characteristics(product_link, headers)

                    # Create wine entry
                    wine_entry = {
                        "Product Link": product_link,
                        "Winery": winery_name,
                        "Wine Name": wine_name_with_vintage,
                        "Country": country,
                        "Region": region,
                        "Wine Type": wine_type,
                        "Grape Type": grape_type,
                        "Price": price,
                        "Rating": rating,
                        "Wine Description 1": wine_facts.get("Wine description", "N/A"),
                        "Wine Description 2": wine_description_2,
                        "Food Pairing": food_pairings,
                        "Alcohol Content": wine_facts.get("Alcohol content", "N/A"),
                        "Allergens": wine_facts.get("Allergens", "N/A"),
                        "Bottle Closure": wine_facts.get("Bottle closure", "N/A"),
                        "Image URL": image_url,
                        "Image Path": image_path,
                        **taste_scores  # Merge taste characteristics into the wine entry
                    }

                    # Append to batch data
                    batch_wine_data.append(wine_entry)

                    # Update batch progress bar
                    batch_pbar.update(1)

                    # Random delay to avoid overloading the server
                    time.sleep(random.uniform(2, 5))

                except Exception as e:
                    print(f"Error scraping {product_link}: {e}")

        # Save batch data to CSV
        df_batch = pd.DataFrame(batch_wine_data)
        df_batch.to_csv("wine_data_with_images.csv", mode="a", header=not os.path.exists("wine_data_with_images.csv"), index=False)

        # Clear batch_wine_data for the next batch
        batch_wine_data = []

        # Update main progress bar
        main_pbar.update(1)

        # Random delay between batches
        time.sleep(random.uniform(10, 20))

print("🎉 All batches processed and saved successfully!")