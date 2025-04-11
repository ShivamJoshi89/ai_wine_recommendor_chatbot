# project/backend/models/wine_list_response.py

from typing import List
from pydantic import BaseModel
from .wine import Wine

class WineListResponse(BaseModel):
    wines: List[Wine]
    total: int
