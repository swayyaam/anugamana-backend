from typing import Optional
from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(..., max_length=500)
    limit: int = Field(default=5, ge=1, le=20)
    chapter: Optional[int] = Field(default=None, ge=1, le=18)
