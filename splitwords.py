import pickle


from gensim.models.word2vec import Word2Vec
import gensim

from wordvec import word_to_vec, vec_to_word

from janome.tokenizer import Tokenizer
"""
t = Tokenizer()
with open('gensim-kvecs.cc.ja.300.vec.pkl', mode='rb') as fp:
    model = pickle.load(fp)

import tensorflow as tf
import tensorflow_datasets as tfds

builder = tfds.builder('huggingface:cc100/lang=ja')
builder.download_and_prepare()
ds = builder.as_dataset(split='train', shuffle_files=False)
"""
"""
with open('splitwords.pkl', mode='rb') as fp:
    words = pickle.load(fp)

print(words[0])"""

"""
words = []
for i, x in enumerate(ds.take(100000)):
    text_tensor = x['text']
    text_str = tf.compat.as_text(text_tensor.numpy())
    malist = t.tokenize(text_str)
    a = []
    for n in malist:
        a.append(n.surface)
    words.append(a)
    if i % 1000 == 0:
        print(i)"""

"""
print("Ok")
with open('splitwords.pkl', mode='wb') as fp:
    pickle.dump(words, fp)"""

from collections import namedtuple, deque

Transition = namedtuple('Transition', ('sentence', 'nex_word', 'reward'))

memory = deque([])
#memory.append(Transition(*args))

with open('splitwords.pkl', mode='rb') as fp:
    words = pickle.load(fp)

memory = []
for i in range(1):
    if words[i] == "":
        continue
    for j, word in enumerate(words[i]):
        a = word_to_vec(''.join(words[i][:(j + 1)]), 30)
        b = word_to_vec(word, 300)
        if a == "notworderror" or b == "notworderror":
            continue
        else:
            memory.append(Transition(a, b, 1))
    if i % 100 == 0:
        print(i)
print(memory)
with open('learndata.pkl', mode='wb') as fp:
    pickle.dump(memory, fp)