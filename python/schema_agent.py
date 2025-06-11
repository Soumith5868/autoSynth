import json
import pandas as pd
from contextlib import suppress
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
from pydantic import ValidationError
import os

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)

def test_llm(client):
    test = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": "Say hello"}]
    )
    print("✅ Test LLM response:", test.choices[0].message.content)


class SchemaAgent:
    def __init__(self, llm_client: OpenAI):
        self.llm = llm_client

    def generate_from_prompt(self, schema_prompt: SchemaPrompt) -> SchemaObject:
        assert schema_prompt.prompt, "Prompt is required"

        system_msg = """You are a strict schema generator. Return ONLY a valid JSON object in this format.
Do not include explanations, markdown, or extra text.

{
  "columns": [
    {"name": "age", "type": "int"},
    {"name": "gender", "type": "categorical", "values": ["M", "F"]},
    {"name": "admission_date", "type": "datetime", "format": "%Y-%m-%d"}
  ]
}

Supported types: "int", "float", "categorical", "string", "datetime"
"""

        user_msg = f"Use-case: {schema_prompt.use_case}\nPrompt: {schema_prompt.prompt}"

        response = self.llm.chat.completions.create(
            model="deepseek-chat",
            messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
            ]
        )
<<<<<<< HEAD

        text = response.choices[0].message.content
        if not text or not text.strip():
            raise ValueError("❌ Empty response from LLM. Check API key, base URL, or network.")
        print("✅ LLM Output:\n", text)
        if not text or not text.strip():
            raise ValueError("LLM response is empty or invalid. Check API status or quota.")
=======
        text = response.choices[0].message.content.strip()
>>>>>>> 477ae0d75550d473f109f9dc72018672e80892f9

        try:
            json_start = text.find('{')
            json_end = text.rfind('}') + 1
            parsed = json.loads(text[json_start:json_end])
            return SchemaObject(use_case=schema_prompt.use_case, **parsed)
        except (json.JSONDecodeError, ValidationError) as e:
            print("❌ LLM returned invalid JSON or schema.")
            print("Raw output:\n", text)
            raise ValueError(f"Schema parsing failed: {e}")

    def generate_from_csv(self, schema_prompt: SchemaPrompt) -> SchemaObject:
        assert schema_prompt.csv_path, "CSV path is required"

        try:
            df = pd.read_csv(schema_prompt.csv_path)
        except Exception as e:
            raise ValueError(f"Failed to read CSV file: {e}")

        cols = []

        for col in df.columns:
            series = df[col].dropna()
            dtype = series.dtype
            unique_count = series.nunique()

            col_type = "string"
            format_hint = None
            values = None

            if pd.api.types.is_bool_dtype(dtype) or set(series.unique()).issubset({0, 1, True, False}):
                col_type = "categorical"
                values = ["0", "1"]
            elif pd.api.types.is_integer_dtype(dtype):
                col_type = "int"
            elif pd.api.types.is_float_dtype(dtype):
                col_type = "float"
            elif pd.api.types.is_datetime64_any_dtype(dtype):
                col_type = "datetime"
                format_hint = "%Y-%m-%d"
            elif unique_count <= 20 and all(isinstance(x, str) for x in series.unique()):
                col_type = "categorical"
                values = list(map(str, series.unique()))
            else:
                with suppress(Exception):
                    if any(pd.to_datetime(series, errors="coerce").notna()):
                        col_type = "datetime"
                        format_hint = "%Y-%m-%d"

            col_schema = ColumnSchema(
                name=col,
                type=col_type,
                values=values,
                format=format_hint
            )
            cols.append(col_schema)

        return SchemaObject(use_case=schema_prompt.use_case, columns=cols)
