from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class CollectionCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    description: str = Field(default="", max_length=2000)


class CollectionRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    description: str
    created_at: datetime
    # Over all the collection's documents, summed by the server: a client
    # holding one page of documents would undercount.
    document_count: int = 0
    chunk_count: int = 0
    total_tokens: int = 0
