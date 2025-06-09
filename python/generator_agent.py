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