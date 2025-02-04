import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


# load model

model = load_model('next_word_lstm.h5')

with open('tokenizer.pickle','rb') as f:
    tokenizer = pickle.load(f)



def predict_next_word(model, tokenizer, text, max_sequence_len):
    # Tokenize the input text
    token_list = tokenizer.texts_to_sequences([text])[0]
    
    # Pad the sequence
    if len(token_list)>=max_sequence_len:
        token_list = token_list[:max_sequence_len]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicated = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicated, axis=1)
    for word, index in tokenizer.word_index.items():
      if index==predicted_word_index:
        return word
    return None

print('Hello WOrld')
#### stremlit app

st.title('Nexxt Word Predictior')

input_text = st.text_input('Enter the text:','To be or not to be')

if st.button('Predict Next Word'):

    max_sequence_len = model.input_shape[1]+1
    next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len)
    st.write(f'The next word is: {next_word}')

