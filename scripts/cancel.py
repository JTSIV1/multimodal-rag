from together import Together
import os

from dotenv import load_dotenv

load_dotenv()

client = Together(
    api_key=os.environ.get("TOGETHER_API_KEY"),
)

ids = [
    "06256bc9-f00d-4d54-9968-2c7488ceeae4",
    "2a52718b-8a16-441f-ab51-5868782a21a4"
]

for id in ids:
    batch = client.batches.cancel(id)
    print(batch)