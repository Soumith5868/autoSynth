from pydantic import BaseModel
from typing import List, Literal, Optional

class ColumnSchema(BaseModel):
    name: str
    type: Literal["int", "float", "categorical", "string", "datetime"]
    format: Optional[str] = None
    values: Optional[List[str]] = None  # only if type == "categorical"


class SchemaObject(BaseModel):
    use_case: str
    columns: List[ColumnSchema]

class SchemaPrompt(BaseModel):
    use_case: str
    prompt: Optional[str] = None
    csv_path: Optional[str] = None
