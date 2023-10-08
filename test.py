import pickle


from gensim.models.word2vec import Word2Vec
import gensim

from janome.tokenizer import Tokenizer

t = Tokenizer()
malist = t.tokenize("私はたかよしです")
with open('gensim-kvecs.cc.ja.300.vec.pkl', mode='rb') as fp:
    model = pickle.load(fp)
for n in malist:
    vec = model[n.surface]
    print(model.most_similar( [ vec ], [], 2)[0])


"""
import tensorflow as tf
import tensorflow_datasets as tfds

builder = tfds.builder('huggingface:cc100/lang=ja')
builder.download_and_prepare()
ds = builder.as_dataset(split='train', shuffle_files=False)

for i, x in enumerate(ds.take(10)):
    text_tensor = x['text']
    text_str = tf.compat.as_text(text_tensor.numpy())
    print(f'{i} : {text_str}')"""

