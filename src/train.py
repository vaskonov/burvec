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
          'sample': 1e-3, # the threshold for configuring which higher-frequency words are randomly downsampled, useful range is (0, 1e-5).
          'hs': 0, # hierarchical softmax
          'negative': 5, #  If > 0, negative sampling will be used, the int for negative specifies how many “noise words” should be drawn (usually between 5-20). If set to 0, no negative sampling is used.
          'seed': seed,
          # 'epochs': 20, # Number of iterations (epochs) over the corpus .
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


def wiki2text(wiki_f, txt_f, **args):
    tokens = []
    wiki = WikiCorpus(wiki_f, **args)
    print("number of articles:", len(wiki))
    with open(txt_f, 'w') as output:
        for text in wiki.get_texts():
            tokens.extend(text)
            output.write(" ".join(text) + "\n")
            
    print('tokens in total', str(len(tokens)))
    print('unique tokens', str(len(list(set(tokens)))))
    return True