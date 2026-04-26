# Docs for v1 can be found by changing the above selector ^
from together import Together
import os

from dotenv import load_dotenv

load_dotenv()

client = Together(
    api_key=os.environ.get("TOGETHER_API_KEY"),
)

ids = [
    "da8cd5b2-62dc-475b-bc27-e230f491f3b6"
]

for id in ids:
    batch = client.batches.cancel(id)
    print(batch)