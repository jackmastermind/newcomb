from dotenv import load_dotenv, dotenv_values
from openai import OpenAI
import argparse
import json
import re

load_dotenv()
prompt_id = dotenv_values(".env")["PROMPT_THEMES"]
client = OpenAI()

parser = argparse.ArgumentParser()
parser.add_argument("-n", type=int, default=1000, help="number of themes to generate")
args = parser.parse_args()

n = args.n

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
with open(f"data/tmp/themes-{n}.json", "w") as f:
    f.write(textout)