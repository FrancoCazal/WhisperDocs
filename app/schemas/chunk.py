from pydantic import BaseModel, ConfigDict


class ChunkRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    document_id: int
    chunk_index: int
    text: str
    token_count: int
