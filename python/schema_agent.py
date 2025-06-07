import json
import pandas as pd
from openai import OpenAI
from pydantic import ValidationError
from schema_models import SchemaObject, SchemaPrompt, ColumnSchema


class SchemaAgent:
    def __init__(self, llm_client: OpenAI):
        self.llm = llm_client

    def generate_from_prompt(self, schema_prompt: SchemaPrompt) -> SchemaObject:
        assert schema_prompt.prompt, "Prompt is required"
        system_msg = (
            "You are a strict schema generator. Return ONLY a JSON object like:\n"
            "{\n"
            "  \"columns\": [\n"
            "    {\"name\": \"age\", \"type\": \"int\", \"min\": 0, \"max\": 120},\n"
            "    {\"name\": \"gender\", \"type\": \"categorical\", \"values\": [\"M\", \"F\"]},\n"
            "    {\"name\": \"admission_date\", \"type\": \"datetime\", \"format\": \"%Y-%m-%d\"}\n"
            "  ]\n"
            "}"
        )
        user_msg = f"Use-case: {schema_prompt.use_case}\nPrompt: {schema_prompt.prompt}"

        response = self.llm.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ]
        )

        text = response.choices[0].message.content

        try:
            # Try to extract valid JSON from potentially messy output
            json_start = text.find('{')
            json_end = text.rfind('}') + 1
            parsed = json.loads(text[json_start:json_end])
            return SchemaObject(use_case=schema_prompt.use_case, **parsed)
        except (json.JSONDecodeError, ValidationError) as e:
            print(f"❌ LLM output invalid: {e}")
            raise ValueError(f"LLM returned malformed or invalid schema.\nRaw output:\n{text}")

    def generate_from_csv(self, schema_prompt: SchemaPrompt) -> SchemaObject:
        assert schema_prompt.csv_path, "CSV path is required"
        df = pd.read_csv(schema_prompt.csv_path)
        cols = []

        for col in df.columns:
            dtype = df[col].dtype
            col_type = "string"
            if pd.api.types.is_integer_dtype(dtype):
                col_type = "int"
            elif pd.api.types.is_float_dtype(dtype):
                col_type = "float"
            elif pd.api.types.is_datetime64_any_dtype(dtype):
                col_type = "datetime"
            elif pd.api.types.is_categorical_dtype(dtype) or df[col].nunique() < 10:
                col_type = "categorical"

            col_schema = ColumnSchema(
                name=col,
                type=col_type,
                min=float(df[col].min()) if col_type in ["int", "float"] else None,
                max=float(df[col].max()) if col_type in ["int", "float"] else None,
                values=list(map(str, df[col].dropna().unique())) if col_type == "categorical" else None,
                format="%Y-%m-%d" if col_type == "datetime" else None
            )
            cols.append(col_schema)

        return SchemaObject(use_case=schema_prompt.use_case, columns=cols)
