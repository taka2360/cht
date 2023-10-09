import numpy as np

from janome.tokenizer import Tokenizer
t = Tokenizer()

import pickle
try:
    with open('gensim-kvecs.cc.ja.300.vec.pkl', mode='rb') as fp:
        model = pickle.load(fp)
except FileNotFoundError:
    print("ベクトル変換用のモデルを生成しています...\nこの処理には初回のみ10分ほどかかります")
    import gensim
    model = gensim.models.KeyedVectors.load_word2vec_format('cc.ja.300.vec.gz', binary=False)
    with open('gensim-kvecs.cc.ja.300.vec.pkl', mode='wb') as fp:
        pickle.dump(model, fp)

def word_to_vec(text:str, axis:int):
    if 300 % axis:
        raise ValueError("300で割り切れるaxisを設定してください!")
    malist = t.tokenize(text)
    vecs = []
    for i, n in enumerate(malist):
        if not i:
            vec = model[n.surface]
        else:
            vec = (vec + model[n.surface]) / 2
    for i in range(axis):
        vecs.append(np.median(vec[i * int(300 / axis) : (i + 1) * int(300 / axis)]))
    return np.asarray(vecs)

def vec_to_word(vec):
    if len(vec) != 300: raise ValueError("300次元のベクトルを指定してください!")
    return model.most_similar( [ vec ], [], 1)[0]

a = word_to_vec("こんにちは", 300)
print(vec_to_word(a))