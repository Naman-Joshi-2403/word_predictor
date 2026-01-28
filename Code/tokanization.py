import spacy 
from collections import Counter
import pickle
from dotenv import load_dotenv
import os

########### load envirnment variable
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
load_dotenv(os.path.join(PROJECT_ROOT, "dev.env"))
clear_text = os.getenv('CLEAN_TEXT_OUTPUT')

########### load small model 
nlp = spacy.load("en_core_web_sm", disable=["ner", "parser", "tagger"])


########### read clear text
with open(os.path.join(PROJECT_ROOT, clear_text), 'r', encoding="utf-8") as file:
    raw_text = file.read()

doc= nlp(raw_text)
tokens = []
