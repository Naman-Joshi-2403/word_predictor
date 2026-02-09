import spacy 
from collections import Counter
import pickle
from config import config
import os

########### load envirnment variable
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
config = config()
clear_text = config.CLEAN_TEXT_OUTPUT

########### load small model 
nlp = spacy.load("en_core_web_sm", disable=["ner", "parser", "tagger", "lemmatizer"])

########### read clear text  x
with open(os.path.join(PROJECT_ROOT, clear_text), 'r', encoding="utf-8") as file:
    raw_text = file.read()

########### Tokenization in chunks
tokens = []
chunk_size = config.CHUNK_SIZE

for i in range(0, len(raw_text), chunk_size): 
    chunk = raw_text[i : i + chunk_size]
    doc= nlp(chunk)
    
    for token in doc:
        if token.is_space:
            continue

        if token.is_alpha or token.is_digit or token.text in [".", ",", "!", "!", "?", "'", "-"]:
            tokens.append(token.text)

######## Vocab
word_frequency = Counter(tokens)
print("Vocabulary size (raw):", len(word_frequency))

min_freq = config.MIN_FREQ
filter_vocab = {word : freq for word, freq in word_frequency.items() if freq >= min_freq}

######## Create vocab mapping
pad_token = "<PAD>"
unk_token = "<UNK>"

word_to_index = {
    pad_token : 0,
    unk_token : 1
}

index_to_word = {
    0 : pad_token,
    1 : unk_token
} 

for idx, value in enumerate(filter_vocab.keys(), start=2):
    word_to_index[value] = idx
    index_to_word[idx] = value

vocab_size = len(word_to_index)
print("Final vocabulary size (with special tokens):", vocab_size)

encoded_raw_text = [word_to_index.get(word, word_to_index[unk_token]) for word in tokens]
print("Encoded text length:", len(encoded_raw_text))
print("UNK token count:", encoded_raw_text.count(word_to_index[unk_token]))

######## Saving Vocab
vocab_dir = os.path.join(PROJECT_ROOT, config.vocab_path)

with open(os.path.join(vocab_dir, "word_to_index.pkl"), "wb") as file:
    pickle.dump(word_to_index, file)

with open(os.path.join(vocab_dir, "index_to_word.pkl"), "wb") as file:
    pickle.dump(index_to_word, file)

with open(os.path.join(vocab_dir, "encoded_text.pkl"), "wb") as file:
    pickle.dump(encoded_raw_text, file)
