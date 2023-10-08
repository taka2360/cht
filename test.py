
from gensim.models.word2vec import Word2Vec
import gensim

model = gensim.models.KeyedVectors.load_word2vec_format('cc.ja.300.vec.gz', binary=False)
print(model)