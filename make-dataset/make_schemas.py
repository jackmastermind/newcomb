from dotenv import load_dotenv, dotenv_values
from openai import OpenAI
import argparse
import json

load_dotenv()
prompt_id = dotenv_values(".env")["PROMPT_ABC"]
client = OpenAI()

parser = argparse.ArgumentParser()
parser.add_argument("--themepath", default="data/tmp/themes-1000.json",
                    help="filepath of themes JSON")
parser.add_argument("--batch-size", type=int, default=10,
                    help="number of themes per API call")
args = parser.parse_args()

themepath = args.themepath
batch_size = args.batch_size

with open(themepath) as f:
    themes = json.load(f)

for i, j in enumerate(range(0, len(themes), batch_size)):
    chunk = themes[j:j+batch_size]

    # call API
    response = client.responses.create(
        prompt={
            "id": prompt_id,
            "variables": {"themes": str(chunk)}
        }
    )

    # write to file
    with open(f"data/tmp/fdt-ABC-{i}.json", "w") as f:
        f.write(response.output_text)