from dotenv import load_dotenv, dotenv_values
from openai import OpenAI
import argparse
import json
import os
import re

load_dotenv()
prompt_id = dotenv_values(".env")["PROMPT_THEMES"]
client = OpenAI()

parser = argparse.ArgumentParser()
parser.add_argument("-n", type=int, default=1000, help="number of themes to generate")
args = parser.parse_args()

n = args.n
output_path = f"data/tmp/themes-{n}.json"

if os.path.exists(output_path):
    print(f"Skipping: {output_path} already exists")
else:
    # call API
    response = client.responses.create(
        prompt={
            "id": prompt_id,
            "variables": {"n": str(n)}
        }
    )

    # remove comments
    textout = re.sub(' *#.*', '', response.output_text)

    # write to file
    with open(output_path, "w") as f:
        f.write(textout)