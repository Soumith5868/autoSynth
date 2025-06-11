<<<<<<< HEAD
import pandas as pd
from ctgan import CTGAN
from typing import List
from faker import Faker
import random

class CTGANGeneratorAgent:
    def __init__(self):
        self.faker = Faker()
        self.model = CTGAN(epochs=100)
        self.generated_categories = {}

    def get_dynamic_category(self, col_name: str, generator_fn, max_unique=200):
        if col_name not in self.generated_categories:
            self.generated_categories[col_name] = list({generator_fn() for _ in range(max_unique)})
        return random.choice(self.generated_categories[col_name])
    
    
    def generate_fake_data_from_schema(self, schema: SchemaObject, n=100) -> pd.DataFrame:
        rows = []
        for _ in range(n):
            row = {}
            for col in schema.columns:
                if col.type == "int":
                    row[col.name] = random.randint(int(col.min or 0), int(col.max or 100))
                elif col.type == "float":
                    row[col.name] = round(random.uniform(col.min or 0.0, col.max or 100.0), 2)
                elif col.type == "categorical":
                    row[col.name] = random.choice(col.values or ["Unknown"])
                elif col.type == "datetime":
                    fmt = col.format or "%Y-%m-%d"
                    row[col.name] = self.faker.date_between(start_date='-5y', end_date='today').strftime(fmt)
                elif col.type == "string":
                    row[col.name] = None
                else:
                    row[col.name] = None
            rows.append(row)
        return pd.DataFrame(rows)

    def fit_ctgan_on_fake_data(self, fake_df: pd.DataFrame, schema: SchemaObject):
        cat_cols = [col.name for col in schema.columns if col.type == "categorical"]
        self.model.fit(fake_df, discrete_columns=cat_cols)

    def sample(self, n=100) -> pd.DataFrame:
        return self.model.sample(n)

    def generate_from_schema(self, schema: SchemaObject, n=100) -> pd.DataFrame:
        fake_df = self.generate_fake_data_from_schema(schema, n=100)
    
        # Keep only valid CTGAN columns
        valid_types = [,"int", "float", "categorical"]
        safe_cols = [col.name for col in schema.columns if col.type in valid_types]
        fake_df = fake_df[safe_cols]

        self.fit_ctgan_on_fake_data(fake_df, schema)
        return self.sample(n)
=======

import pandas as pd
import random
from faker import Faker
from typing import List, Dict, Optional
from schema_models import SchemaObject, ColumnSchema, SchemaPrompt
from openai import OpenAI
import json

faker = Faker()

class GeneratorAgent:
    def __init__(self, llm_client: OpenAI):
        self.llm = llm_client

    def _call_llm_column_classification(self, schema: SchemaObject) -> List[Dict]:
        system_prompt = """You are a helpful assistant that classifies columns for synthetic data generation.
For each column in the schema, respond with:
- column_name
- can_use_faker: YES or NO
- categorical: YES or NO
- faker_name: (if can_use_faker is YES)
- categories: (if categorical is YES)

Respond only in a JSON list like:
[
  {"column_name": "name", "can_use_faker": "YES", "categorical": "NO", "faker_name": "name"},
  {"column_name": "account_type", "can_use_faker": "NO", "categorical": "YES", "categories": ["Free", "Pro", "Enterprise"]}
]
"""

        user_prompt = "Schema Columns:\n" + "\n".join(
            [f"- {col.name} ({col.type})" for col in schema.columns]
        )

        response = self.llm.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        raw = response.choices[0].message.content.strip()
        json_start = raw.find('[')
        json_end = raw.rfind(']') + 1
        return eval(raw[json_start:json_end])  # Should ideally use json.loads

    def _call_llm_generate_values(self, column_name: str, n: int) -> List[str]:
        prompt = f"""
    You are a synthetic data assistant.

    Your task is to generate {n} realistic and diverse values for a column named '{column_name}'.
    Respond ONLY with a JSON array of strings, like this:

    [
    "value1",
    "value2",
    "value3"
    ]

    Do not include explanations, comments, or markdown — only the array.
    Now generate values for column: '{column_name}'
    """

        response = self.llm.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": prompt.strip()}
            ]
        )

        text = response.choices[0].message.content.strip()
        try:
            json_start = text.find('[')
            json_end = text.rfind(']') + 1
            parsed = json.loads(text[json_start:json_end])
            if not isinstance(parsed, list):
                raise ValueError("Parsed object is not a list.")
            return parsed[:n]
        except Exception as e:
            print("⚠️ Failed to parse LLM output as JSON array. Fallback to dummy values.")
            return [f"{column_name}_value_{i}" for i in range(n)]

    def generate_sample(self, schema_prompt: SchemaPrompt, schema: SchemaObject, n: int = 200) -> pd.DataFrame:

        if schema_prompt.csv_path:
            try:
                df = pd.read_csv(schema_prompt.csv_path)
                return df.sample(n=min(n, len(df)), replace=True).reset_index(drop=True)
            except Exception:
                pass

        column_plan = self._call_llm_column_classification(schema)
        df = pd.DataFrame()

        for col in column_plan:
            col_name = col["column_name"]

            if col["can_use_faker"] == "YES":
                faker_func = getattr(faker, col.get("faker_name", "name"), lambda: "unknown")
                df[col_name] = [faker_func() for _ in range(n)]

            elif col["categorical"] == "YES":
                categories = col.get("categories", ["A", "B", "C"])
                df[col_name] = [random.choice(categories) for _ in range(n)]

            else:
                values = self._call_llm_generate_values(col_name, n)
                df[col_name] = values

        return df
>>>>>>> 477ae0d75550d473f109f9dc72018672e80892f9
