from gensim.models import KeyedVectors
from gensim.corpora.wikicorpus import *
from gensim.models import FastText
from gensim.models.word2vec import Word2Vec
import multiprocessing


def w2v(vector_size, window, min_count, fname, fcorpus, sg=1, seed=1234):
    params = {'sg':sg, 
          'vector_size': vector_size, 
          'window': window, 
          'min_count': min_count, 
          'sample': 1e-3, 
          'hs': 0, # hierarchical softmax
          'negative': 5, # negative sampling
          'seed': seed,
#           'epochs': 20, # Number of iterations (epochs) over the corpus .
          'workers': max(1, multiprocessing.cpu_count() - 1)}
    word2vec = Word2Vec(corpus_file=fcorpus, **params)
    word2vec.wv.save_word2vec_format(fname)
    return True


def ft(vector_size, window, min_count, fname, fcorpus):
    ft = FastText(corpus_file=fcorpus, 
              vector_size = vector_size, 
              window = window, 
              min_count = min_count, 
              epochs=10)
    ft.wv.save_word2vec_format(fname)
    return True


def wiki2text(wiki_f, txt_f):
    tokens = []
    wiki = WikiCorpus(wiki_f, token_min_len=0, lower=True)
    print("number of articles:", len(wiki))
    with open(txt_f, 'w') as output:
        for text in wiki.get_texts():
            output.write(" ".join(text) + "\n")
    return True