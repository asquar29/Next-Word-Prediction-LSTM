import numpy as np
import streamlit as st
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model


try:
    model = load_model('next_word_lstm_model.h5')
except Exception as e:
    st.error(f"Failed to load model: {e}")

try:
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
except Exception as e:
    st.error(f"Failed to load tokenizer: {e}")



# function to predict the next word
def predict_next_word(model, tokenizer, text,max_sequence_len):
  token_list = tokenizer.texts_to_sequences([text])[0]
  if (len(token_list)) >=max_sequence_len:
    token_list = token_list[-(max_sequence_len -1):]

  token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
  predicted = (model.predict(token_list,verbose=0))
  predicted_word_index = np.argmax(predicted,axis=1)

  for word, index in tokenizer.word_index.items():
    if index == predicted_word_index:
       return word

  return None





# streamlit app
st.title('Next word Prediction with LSTM Model')
input_text = st.text_input('Enter the sequences of words',"To be or to be")

st.button('Predict Next Word.')

if (st.button('Predict Next Word')):
  max_sequence_len = model.input_shape[1]+1
  next_word = predict_next_word(model,tokenizer,input_text,max_sequence_len)
  st.write(f'Next Word: {next_word}')