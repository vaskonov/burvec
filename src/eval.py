import json
import torch

import numpy as np
from torch import nn

from collections import OrderedDict
from gensim.models import KeyedVectors

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
        vocab = KeyedVectors.load_word2vec_format(wes_f, binary=False).index_to_key
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


def tabular(scores_d, language):
    for method, scores in scores_d[language].items():
        print(method)
        for vector_size, windows in scores.items():
            ws = [str(round(a,3)) for a in list(windows.values())]
            print("&",vector_size," & "," & ".join(ws),"\\\\")
    return True