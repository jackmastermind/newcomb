from dotenv import load_dotenv, dotenv_values
from openai import OpenAI
from glob import glob
import argparse
import json
import os

load_dotenv()
prompt_id = dotenv_values(".env")["PROMPT_DESCRIBE"]
client = OpenAI()

parser = argparse.ArgumentParser()
parser.add_argument("--pattern", default="data/tmp/fdt-ABC-*.json",
                    help="glob pattern for schema files")
args = parser.parse_args()

pattern = args.pattern

for i, fp in enumerate(glob(pattern)):
    output_path = f"data/tmp/fdt-description-{i}.json"
    if os.path.exists(output_path):
        print(f"Skipping {i}: {output_path} already exists")
        continue

    with open(fp) as file:
        schemas = file.read()

    # call API
    response = client.responses.create(
        prompt={
            "id": prompt_id,
            "variables": {"schemas": schemas}
        }
    )

    # write to file
    with open(output_path, "w") as f:
        f.write(response.output_text)