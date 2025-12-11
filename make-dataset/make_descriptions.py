from dotenv import load_dotenv, dotenv_values
from openai import OpenAI
from glob import glob
import argparse
import json

load_dotenv()
prompt_id = dotenv_values(".env")["PROMPT_DESCRIBE"]
client = OpenAI()

parser = argparse.ArgumentParser()
parser.add_argument("--pattern", default="data/tmp/fdt-ABC-*.json",
                    help="glob pattern for schema files")
args = parser.parse_args()

pattern = args.pattern

for i, fp in enumerate(glob(pattern)):
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
    with open(f"data/tmp/fdt-description-{i}.json", "w") as f:
        f.write(response.output_text)