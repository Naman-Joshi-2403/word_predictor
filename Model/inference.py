import tensorflow as tf
import numpy as np 
import pickle

model = tf.keras.models.load_model(r"C:\Users\naman\OneDrive\Desktop\Guvi_Projects\Next Word Prediction\Model\word_pred_LSTM_model.keras")
model.make_predict_function()

with open(r"C:\Users\naman\OneDrive\Desktop\Guvi_Projects\Next Word Prediction\Tokenizer\Vocab\word_to_index.pkl", "rb") as file:
    word_to_index = pickle.load(file)

with open(r"C:\Users\naman\OneDrive\Desktop\Guvi_Projects\Next Word Prediction\Tokenizer\Vocab\index_to_word.pkl", "rb") as file:
    index_to_word = pickle.load(file)


max_seq_len = model.input_shape[1]

def word_to_sequence(text):
    words = text.lower().split()
    return [word_to_index.get(word, 0) for word in words]

def generate_text(seed_text, num_word = 15):
    generate_words = []
    current_text = seed_text.lower()

    for item in range(num_word):
        token_list = word_to_sequence(current_text)

        token_list = tf.keras.preprocessing.sequence.pad_sequences(
            [token_list],
            maxlen = max_seq_len,
            padding = "pre" 
        )

        prediction = model.predict(token_list, verbose=0)[0]

        next_index = np.argmax(prediction)
        next_word = index_to_word.get(next_index)

        if not next_word:
            break

        generate_words.append(next_word)
        current_text += " " + next_word

        return " ".join(generate_words)

while True:
    seed = input("Enter your Text : ")

    if seed.lower() == "exit":
        break
        
    if len(seed.split()) < 3:
        print("please enter more context")
        continue

    output = generate_text(seed_text=seed, num_word=20)

    print(output)
