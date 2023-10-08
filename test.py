import pickle


from gensim.models.word2vec import Word2Vec
import gensim

from janome.tokenizer import Tokenizer

t = Tokenizer()
malist = t.tokenize("私は田舎で育ちました。")
with open('gensim-kvecs.cc.ja.300.vec.pkl', mode='rb') as fp:
    model = pickle.load(fp)

print(model.most_similar( [ model["猫"] ], [], 10)[0][0])

