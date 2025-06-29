# 📘 Next Word Prediction Using LSTM (TensorFlow)

This project implements a **Next Word Prediction** model using **LSTM (Long Short-Term Memory)** networks built with **TensorFlow/Keras**. Given a sequence of words, the model predicts the most likely next word.

---

## 🚀 Features

- Uses **LSTM layers** for sequential learning
- Built with **TensorFlow/Keras**
- Trained on custom text corpus
- Predicts next word given a sentence prefix
- A RNN-based model
---

## 🧠 Model Architecture

- `Embedding` layer: Converts words to dense vector representations
- `LSTM` layer: Learns temporal dependencies in word sequences
- `Dense` layer: Outputs probability distribution over vocabulary

```python
model = Sequential()
model.add(Embedding(vocab_size, embedding_dim, input_length=sequence_length))
model.add(LSTM(128, return_sequences=False))
model.add(Dense(vocab_size, activation='softmax'))
