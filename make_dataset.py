from dotenv import load_dotenv, dotenv_values
from openai import OpenAI, APIStatusError
import os
import json

load_dotenv()
prompt_id = dotenv_values(".env")["PROMPT_ABC"]
client = OpenAI()

number = 102
assert number % 3 == 0
iters = 50

convo = client.conversations.create()

try:
    response = client.responses.create(
        prompt={
            "id": prompt_id,
            "variables": {"n": str(number)}
        },
        conversation=convo.id
    )
    textout = response.output_text
    print(textout)
    with open("data/tmp/fdt-0.json", "w") as f:
        json.dump({"response": textout}, f)

    for i in range(1, iters):
        response = client.responses.create(
            input=f"Good. Generate {number} more schemas.",
            conversation=convo.id
        )
        textout = response.output_text
        print(textout)
        with open(f"data/tmp/fdt-{i}.json", "w") as f:
            json.dump({"response": textout}, f)

except APIStatusError as e:
    print(f"HTTP error {e.status_code}: {e.message}")

print()
