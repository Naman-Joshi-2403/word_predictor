import pandas as pd
import os
import re
from dotenv import load_dotenv

####### Project Root path
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

####### Load Env variable
load_dotenv(os.path.join(PROJECT_ROOT, "dev.env"))
input_dataset = os.getenv("INPUT_DATA")
clean_text_output =  os.getenv("CLEAN_TEXT_OUTPUT")

####### Load Dataset in large text corpus
df = pd.read_csv(os.path.join(PROJECT_ROOT, input_dataset))
texts = df['text'].dropna().astype(str)
raw_text = " ".join(texts.to_list())

####### Cleaning text
raw_text = raw_text.lower()
raw_text = re.sub(r"http\S+|www\S+", " ", raw_text) ##### remove urls
raw_text = re.sub(r"\[[0-9]+\]", " ", raw_text)   ###### remove reference links example: [a][1]
raw_text = re.sub(r"[^a-z0-9,.!?'\-\s]", " ", raw_text)  ######## remove special character
raw_text = re.sub(r"\s+", " ", raw_text)
raw_text = raw_text.strip()

########## replacing encoding words
replacements = {
        "â€“": "-",
        "â€”": "-",
        "â€˜": "'",
        "â€™": "'",
        "â€œ": '"',
        "â€�": '"'
    }

for encoded, value in replacements.items():
    raw_text = raw_text.replace(encoded, value)

with open(os.path.join(PROJECT_ROOT, clean_text_output), mode='w', encoding="utf-8") as file:
    file.write(raw_text)
