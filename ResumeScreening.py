import numpy as np 
import pandas as pd 
import nltk
import tensorflow as tf

nltk.download('stopwords')

from nltk.corpus import stopwords
stopwords = set(stopwords.words('english'))

data = pd.read_csv('UpdatedResumeDataSet.csv')
data.head()

print(data['Category'].value_counts())

import re

def cleanResume(Text):
    Text = re.sub('http\S+\s*', ' ', Text)
    Text = re.sub('RT|cc', ' ', Text)
    Text = re.sub('#\S+', '', Text)
    Text = re.sub('@\S+', '  ', Text)
    Text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', Text)
    Text = re.sub(r'[^\x00-\x7f]',r' ', Text) 
    Text = re.sub('\s+', ' ', Text)
    return Text

data['cleaned_resume'] = ''  
data['cleaned_resume'] = data.Resume.apply(lambda x: cleanResume(x))

for i in range(962):
  str = data.Category[i]
  for j in range(len(str)):
    str = str.replace(" ", "")
  data.Category[i] = str


sentences = list(data.cleaned_resume)

for idx in range(len(sentences)):
  sentence = sentences[idx]
  for word in stopwords:
    token = " " + word + " "
    sentence = sentence.replace(token, " ")
    sentence = sentence.replace("  ", " ")
  sentences[idx] = sentence

labels = list(data.Category)

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

vocab_size = 30000
embedding_dim = 32
max_length = 1024

import matplotlib.pyplot as plt


def plot_history(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_'+string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_'+string])
    plt.show()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(sentences, labels, train_size = 0.8, random_state = 1, shuffle = True)

tokenizer = Tokenizer(num_words = 15000, oov_token='<OOV>')
tokenizer.fit_on_texts(X_train)
word_index = tokenizer.word_index
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_train_padded = pad_sequences(X_train_seq, padding='post', maxlen=max_length)
X_test_seq = tokenizer.texts_to_sequences(X_test)
X_test_padded = pad_sequences(X_test_seq, padding='post', maxlen=max_length)

label_tokenizer = Tokenizer()
label_tokenizer.fit_on_texts(labels)
label_index = label_tokenizer.word_index

print(label_index)

y_train_label_seq = np.array(label_tokenizer.texts_to_sequences(y_train))
y_test_label_seq = np.array(label_tokenizer.texts_to_sequences(y_test))

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dense(256, activation='tanh'),
    tf.keras.layers.Dense(26, activation='softmax')
])
model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()
history = model.fit(X_train_padded, y_train_label_seq, epochs=15, validation_data=(X_test_padded, y_test_label_seq), verbose=2)

plot_history(history, 'accuracy')
plot_history(history, 'loss')

labels_pred = np.argmax(model.predict(X_test_padded),axis = -1) 
print(labels_pred[0:24])
print(y_test_label_seq[0:24].reshape( 1, -1))

from sklearn.metrics import accuracy_score
print("Accuracy Score")
print(accuracy_score(labels_pred,y_test_label_seq))



