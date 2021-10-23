import json
import torch
import numpy as np
from torch import nn
from gensim.models import KeyedVectors
from gensim.corpora.wikicorpus import WikiCorpus
from gensim.models import FastText
from gensim.models.word2vec import Word2Vec
from collections import OrderedDict
import multiprocessing


# # based on text build windows n pairs of (word,context)
# def count_pairs(text, num):
#     ngs = list(ngrams(text.split(), num))
#     return dict(Counter(ngs))

def w2v(vector_size, window, min_count, fname, fcorpus, sg=1):
    params = {'sg':sg, 
          'vector_size': vector_size, 
          'window': window, 
          'min_count': min_count, 
          'sample': 1e-3, 
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
    wiki = WikiCorpus(wiki_f, token_min_len=0, lower=True)
    print("number of articles:", len(wiki))
    with open(txt_f, 'w') as output:
        for text in wiki.get_texts():
            output.write(" ".join(text) + "\n")


# # assert(build_pairs(['I like apples and alcohol'],2), )
# # print(build_pairs(['I like apples and alcohol is this OK'],3))
# assert(build_pairs(['I like apples and alcohol'],1), [('I', 'like'), ('like', 'I'), ('like', 'apples'), ('apples', 'like'), ('apples', 'and'), ('and', 'apples'), ('and', 'alcohol'), ('alcohol', 'and')])

           
def score(test_f, model_f, vocab=[]):
    with open(test_f, 'r') as f:
        test_d = json.load(f)   

    we = KeyedVectors.load_word2vec_format(model_f, binary=False)
    test_cd = OrderedDict(test_d)
    
    scores = {}
    
    for key, values in test_cd.items():
        for value in values:
            
            if len(vocab)>0:
                if value not in vocab or key not in vocab:
                    continue
            else:
                if key not in we.index_to_key or value not in we.index_to_key:
                    continue
            scores[(key, value)] = (we[key], we[value], 1)

    eval_values = list(test_cd.values())
    for idx, key in enumerate(list(test_cd.keys())):
        if idx+1<len(eval_values):
            for value in eval_values[idx+1]:
                
                if len(vocab)>0:
                    if value not in vocab or key not in vocab:
                        continue
                else:
                    if key not in we.index_to_key or value not in we.index_to_key:
                        continue
                
                scores[(key, value)] = (we[key], we[value], -1)
    return scores
    
def loss(scores):
    cos_loss = nn.CosineEmbeddingLoss()
    x = [i[0] for i in scores.values()]
    y = [i[1] for i in scores.values()]
    target = [i[2] for i in scores.values()]
    
    try:
        _loss = cos_loss(torch.tensor(x, dtype=torch.float64),
                    torch.tensor(y, dtype=torch.float64), 
                    torch.tensor(target))
    except:
        print(scores)
        print(x)
        print(y)
        print(target)
    
    return _loss

    
def get_vocab(wes_fs):
    vocabs = []
    for wes_f in wes_fs:
        vocab = KeyedVectors.load_word2vec_format(wes_f, binary=False).index_to_key
        vocabs.append(vocab)
        print(wes_f, len(vocab))
    
    joint_vocab = vocabs[0]
    for vocab in vocabs:
        joint_vocab = np.intersect1d(joint_vocab, vocab).tolist()
        
    return joint_vocab