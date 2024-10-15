import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, GlobalAveragePooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# Данные для тренировки с ключевыми словами
reviews = [
    "The movie was fantastic, I really enjoyed it",        # Положительный
    "I hated the movie, it was awful",                     # Отрицательный
    "Amazing performance and great storyline",             # Положительный
    "Terrible movie, I would not recommend",               # Отрицательный
    "An excellent film with a superb plot",                # Положительный
    "Bad acting, very boring",                             # Отрицательный
    "Loved it, best movie I've seen in a while",           # Положительный
    "The plot was very weak and confusing",                # Отрицательный
    "Great cinematography and acting",                     # Положительный
    "The movie was too slow and dull",                     # Отрицательный
    "It was a good movie, but not the best",               # Положительный
    "I disliked the ending, it was bad",                   # Отрицательный
    "The film was amazing and I loved every second",       # Положительный
    "It was a horrible experience, I disapprove",          # Отрицательный
    "The direction was superb and the actors were great",  # Положительный
    "Awful script, I didn’t like it at all",               # Отрицательный
    "Good action scenes but the plot was mediocre",        # Нейтральный/Положительный
    "The movie was boring and bad",                        # Отрицательный
    "Great special effects and a good soundtrack",         # Положительный
    "I absolutely loved the movie, it was fantastic!",     # Положительный
    "I dislike the characters and the story was terrible", # Отрицательный
    "I enjoyed the movie but it wasn't great",             # Нейтральный/Положительный
    "The acting was superb but the story was bad",         # Нейтральный/Отрицательный
    "Fantastic movie with great performances",             # Положительный
    "The movie was okay, not great, but enjoyable",        # Нейтральный/Положительный
    "The movie was superb and I really liked it",          # Положительный
    "The movie was awful and I really disliked it",        # Отрицательный
    "I would not recommend",                               # Отрицательный
    "Just a waste of time",                                # Отрицательный
    "That was great!",                                     # Положительный
    "I like the movie plot and actions",                   # Положительный
    "If you want to seу something unique and intriguing, I woul advise", # Положительный
    "It was cool and fascinating",                         # Положительный
    "Disgusting",                                          # Отрицательный
    "Expected better...",                                  # Отрицательный
    "Regret seeing this",                                  # Отрицательный
    "The film producer is a genious!",                     # Положительный
    "It startled me in a good way",                        # Положительный
    "Such a great combination of plot, actions and music", # Положительный
    "Superb!",                                             # Положительный
    "It was just fine",                                    # Нейтральный/Положительный
    "Nothing special",                                     # Нейтральный/Положительный
    "Very emotional movie!",                               # Положительный
    "I was happy to see it",                               # Положительный
    "Wonderful!",                                          # Положительный
    "I wanna see it again!!",                              # Положительный
    "Someone, erase my memory(( ",                         # Положительный
    "Fantastic",                                           # Положительный
    "What a masterpiece!"                                  # Положительный


]

# Соответствующие рейтинги по 10-бальной шкале
ratings = [9, 1, 9, 1, 10, 2, 10, 3, 9, 2, 6, 3, 10, 1, 10, 1, 5, 2, 7, 10, 1, 6, 5, 10, 7, 10, 1, 2, 1, 9, 9, 9, 10, 1, 4, 1, 10, 9, 10, 8, 5, 5, 9, 7, 7, 9, 1, 8, 10]  # Добавлен недостающий рейтинг

# Параметры модели
vocab_size = 2000
embedding_dim = 16
max_length = 30
trunc_type = 'post'
padding_type = 'post'
oov_tok = "<OOV>"

# Токенизация текстов
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(reviews)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(reviews)
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

# нейронная сеть
model = Sequential([
    Embedding(vocab_size, embedding_dim, input_length=max_length),
    GlobalAveragePooling1D(),
    Dense(16, activation='relu'),
    Dense(1)  # Выходной слой, предсказывающий рейтинг от 0 до 10
])

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])

# Тренировка модели
model.fit(padded_sequences, np.array(ratings), epochs=500)

# Функция предсказания рейтинга
def predict_review(review):
    # Преобразование текста отзыва в формат, понятный модели
    seq = tokenizer.texts_to_sequences([review])
    padded = pad_sequences(seq, maxlen=max_length, padding=padding_type, truncating=trunc_type)

    # Предсказание рейтинга
    predicted_rating = model.predict(padded)[0][0]

    # Определение, положительный отзыв или отрицательный
    if predicted_rating >= 5:
        print(f"Positive Review. Predicted Rating: {predicted_rating:.2f}/10")
    else:
        print(f"Negative Review. Predicted Rating: {predicted_rating:.2f}/10")

# Пример
new_review = "The movie was superb and I really liked it"
predict_review(new_review)
