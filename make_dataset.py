from dotenv import load_dotenv, dotenv_values
from openai import OpenAI
from openai.types.responses.response_reasoning_summary_text_delta_event import ResponseReasoningSummaryTextDeltaEvent as ResponseDelta
import os

load_dotenv()
prompt_id = dotenv_values(".env")["PROMPT"]
client = OpenAI()
number = 100
iters = 50

convo = client.conversations.create()
print(convo)
stream = client.responses.create(
    model='gpt-5.1',
    reasoning={'effort': 'low'},
    prompt={
        "id": prompt_id,
        "version": "2",
        "variables": {
            "number": str(number)
        }
    },
    stream=True,
    conversation=convo.id
)

os.remove('output.tsv')
for i in range(1, iters + 1):
    textout = ""
    for event in stream:
        if hasattr(event, 'delta') and not isinstance(event, ResponseDelta):
            textout += event.delta
            print(event.delta, end='')
    with open('output.tsv', 'a') as file:
        file.write(textout + '\n')
    
    if i < iters:
        stream = client.responses.create(
            model="gpt-5.1",
            reasoning={"effort": "low"},
            input=f"Good. Generate {str(number)} more rows for the same .tsv. You do not need to output the header.",
            stream=True,
            conversation=convo.id
        )

print()