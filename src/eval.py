import json
import torch

import random
import numpy as np
from torch import nn

from collections import OrderedDict
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity

def loss(scores):
    cos_loss = nn.CosineEmbeddingLoss()
    x = [i[0] for i in scores.values()]
    y = [i[1] for i in scores.values()]
    target = [i[2] for i in scores.values()]
    
    try:
        _loss = cos_loss(torch.tensor(x, dtype=torch.float64),
                    torch.tensor(y, dtype=torch.float64), 
                    torch.tensor(target))
        return _loss
    
    except Exception as e:
        print(e)
        print(scores)
        print(x)
        print(y)
        print(target)
    
    return False


# count common vocabulary between word embeddings files wes_fs
def get_vocab(wes_fs):
    vocabs = []
    for wes_f in wes_fs:
        
        if str(wes_f).split("/")[-1] in ['pmiWeighting', 'PlmiWeighting', 'PlogWeighting', 'PpmiWeighting', 'EpmiWeighting', 'ft']:
            continue
        
        print(wes_f)
        try:
            vocab = KeyedVectors.load_word2vec_format(wes_f, binary=False).index_to_key
        except:
            vocab = KeyedVectors.load_word2vec_format(wes_f, binary=False, no_header=True).index_to_key

        vocab = [x for x in vocab if x is not None]
        
        vocabs.append(vocab)
        print(wes_f, len(vocab))
    
    joint_vocab = vocabs[0]
    for vocab in vocabs:
        joint_vocab = np.intersect1d(joint_vocab, vocab).tolist()
        
    return joint_vocab


def score(test_f, model_f, vocab=[]):
    print("score: currently processing: ", test_f)
    with open(test_f, 'r') as f:
        test_d = json.load(f)   

    we = KeyedVectors.load_word2vec_format(model_f, binary=False)
    test_cd = OrderedDict(test_d)
    
    total = []
    oov = []
    
    positive = {}
    negative = {}
    
    for values in list(test_cd.values()):
        for idx, value1 in enumerate(values[:-1]):
            for value2 in values[idx+1:]:
                if value1 in vocab and value2 in vocab:
                    positive[(value1, value2)] = (we[value1], we[value2], 1)

    eval_values = list(test_cd.values())
    for idx, values1 in enumerate(eval_values[:-2]):
        for value2 in eval_values[idx+2]:
            for value1 in values1:
                if value1 in vocab and value2 in vocab:
                    if len(negative.keys()) < len(positive.keys()):
                        negative[(value1, value2)] = (we[value1], we[value2], -1)
        
    print('total words',len(list(set(total))))
    print('oov words', len(list(set(oov))))
    print('positive pairs', len([a for a in list(positive.values()) if a[2] == 1]))
    print('negative pairs', len([a for a in list(negative.values()) if a[2] == -1]))
          
    return {**positive, **negative}


def tabular(scores_d, language):
    for method, scores in scores_d[language].items():
        print(method)
        for vector_size, windows in scores.items():
            ws = [str(round(a,3)) for a in list(windows.values()) if type(a) is not list]
            print("&",vector_size," & "," & ".join(ws),"\\\\")
    return True