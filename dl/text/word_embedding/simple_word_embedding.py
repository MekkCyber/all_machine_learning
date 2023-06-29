import io
import os
import re
import shutil
import string
import tensorflow as tf

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.layers import TextVectorization

## get the data
url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
dataset = tf.keras.utils.get_file("aclImdb_v1.tar.gz", url, untar=True, cache_dir='.', cache_subdir='')
print('Dataset downloaded successfully')
## set the paths
dataset_dir = os.path.join(os.path.dirname(dataset), 'aclImdb')
os.listdir(dataset_dir)
train_dir = os.path.join(dataset_dir,'train')
os.listdir(train_dir)

## remove uneeded folders
remove_dir = os.path.join(train_dir, 'unsup')
shutil.rmtree(remove_dir)

## create Dataset
batch_size = 1024
seed = 123
train_ds = tf.keras.utils.text_dataset_from_directory(
    'aclImdb/train', batch_size=batch_size, validation_split=0.2,
    subset='training', seed=seed)
val_ds = tf.keras.utils.text_dataset_from_directory(
    'aclImdb/train', batch_size=batch_size, validation_split=0.2,
    subset='validation', seed=seed)
'''
.cache() keeps data in memory after it's loaded off disk.
.prefetch() overlaps data preprocessing and model execution while training.
'''
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
print("Dataset configured successfully")
## create embedding layer
'''
For text or sequence problems, the Embedding layer takes a 2D tensor of integers, of shape (samples, sequence_length), 
where each entry is a sequence of integers. It can embed sequences of variable lengths.
The returned tensor has one more axis than the input, the embedding vectors are aligned along the new last axis.
Pass it a (2, 3) input batch and the output is (2, 3, N)
When given a batch of sequences as input, an embedding layer returns a 3D floating point tensor, of shape (samples, sequence_length, embedding_dimensionality). 
To convert from this sequence of variable length to a fixed representation there are a variety of standard approaches. You could use an RNN, Attention, 
or pooling layer before passing it to a Dense layer.
'''
embedding_layer = tf.keras.layers.Embedding(1000, 5)

## Data Preprocessing and Tokenization
def custom_standardization(input_data):
  lowercase = tf.strings.lower(input_data)
  stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
  return tf.strings.regex_replace(stripped_html, '[%s]' % re.escape(string.punctuation), '') #to remove in punctuation from the string, re.escape is used to escape any special characters present in string.punctuation
vocab_size = 10000
sequence_length = 100
vectorize_layer = TextVectorization(
    standardize=custom_standardization,
    max_tokens=vocab_size,
    output_mode='int',
    output_sequence_length=sequence_length)
text_ds = train_ds.map(lambda x, y: x) ## Make a text-only dataset (no labels)
vectorize_layer.adapt(text_ds) # call adapt to build the vocabulary.

## Create the model
embedding_dim=16

model = Sequential([
  vectorize_layer,
  Embedding(vocab_size, embedding_dim, name="embedding"),
  GlobalAveragePooling1D(), # calculates the average over the second dimension, which is the sequence length, and outputs (batch_size,embedding_dim)
  Dense(16, activation='relu'),
  Dense(1)
])
print("Model created")
## Compile model
model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])
print("Model compiled")
## Fit the model
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=15,
)

## Retrieve weights
weights = model.get_layer('embedding').get_weights()[0]
vocab = vectorize_layer.get_vocabulary()
out_v = io.open('vectors.tsv', 'w', encoding='utf-8')
out_m = io.open('metadata.tsv', 'w', encoding='utf-8')

for index, word in enumerate(vocab):
  if index == 0:
    continue  # skip 0, it's padding.
  vec = weights[index]
  out_v.write('\t'.join([str(x) for x in vec]) + "\n")
  out_m.write(word + "\n")
out_v.close()
out_m.close()




