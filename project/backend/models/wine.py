from typing import Union
from pydantic import BaseModel

class Wine(BaseModel):
    id: str
    wine_id: str
    winery: str
    wine_name: str
    country: str
    region: str
    wine_type: str
    grape_type_list: str
    price: float
    rating: float
    wine_description_1: str
    wine_description_2: str
    food_pairing: str
    alcohol_content: float
    allergens: str
    bottle_closure: str
    light_to_bold: str
    smooth_to_tannic: str
    dry_to_sweet: str
    soft_to_acidic: str
    gentle_to_fizzy: str
    vintage: Union[int, str]  # Accept both integers and strings
    primary_type: str
    image_url: str
