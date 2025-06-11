from pathlib import Path
import json
fn_txt_path = Path("/app/dataset/first_names.txt")

with open(fn_txt_path, "r", encoding="utf-8") as f:
    names = [line.strip() for line in f if line.strip()]

print("✅ Loaded", len(names), "names")

json_path = "/app/dataset/first_names.json"
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(names, f, indent=2)


ln_txt_path = Path("/app/dataset/last_names.txt")

with open(ln_txt_path, "r", encoding="utf-8") as f:
    names = [line.strip() for line in f if line.strip()]

print("✅ Loaded", len(names), "names")

json_path = "/app/dataset/last_names.json"
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(names, f, indent=2)