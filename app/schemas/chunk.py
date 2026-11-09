from pydantic import BaseModel, ConfigDict


class ChunkRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    document_id: int
    chunk_index: int
    text: str
    token_count: int
    # The rules of app/services/instruction_scan.py the text matches.
    instruction_flags: list[str] = []
