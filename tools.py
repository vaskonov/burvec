import nltk
import json
import torch
from torch import nn
from nltk import word_tokenize
from nltk.util import ngrams
from gensim.models import KeyedVectors
from collections import OrderedDict
from collections import Counter

# # based on text build windows n pairs of (word,context)
# def count_pairs(text, num):
#     ngs = list(ngrams(text.split(), num))
#     return dict(Counter(ngs))

def build_pairs(post_lemma, ng):
    omit = ['BEG', 'END', 'UNK']
    word_context = []
    words = []

#     print('build_pairs: total articles', len(post_lemma))

    for idx, text in enumerate(post_lemma):
        
        text = text.split(" ")
        
        for i in range(ng):
            text = ['BEG'] + text
            text = text + ['END']

        for idx in range(ng,len(text)-ng):
            for i in range(idx-ng, idx):
                if text[i] not in omit:
                    word_context.append((text[idx], text[i]))
                
            for i in range(idx+1, idx+ng+1):
                if text[i] not in omit:
                    word_context.append((text[idx], text[i]))

#     print('build_pairs: ng:', ng, ' num of pairs:', len(word_context))
    return dict(Counter(word_context))

# # assert(build_pairs(['I like apples and alcohol'],2), )
# # print(build_pairs(['I like apples and alcohol is this OK'],3))
# assert(build_pairs(['I like apples and alcohol'],1), [('I', 'like'), ('like', 'I'), ('like', 'apples'), ('apples', 'like'), ('apples', 'and'), ('and', 'apples'), ('and', 'alcohol'), ('alcohol', 'and')])


def convert(fin,fout):
    x = []
    
    with open(fin) as f:
        for l in f:
            x.append(l.strip().split("\t"))
    
    with open(fout, 'w') as f:        
        f.write(str(len(x))+ " "+ str(len(x[0][1:]))+"\n")
        for l in x:
            f.write(" ".join(l)+"\n")
            
def prepare_pmi(pmi):
    rows = [i[0] for i in list(pmi.keys())]
    cols = [i[1] for i in list(pmi.keys())]
    
    with open('/tmp/data.pmi', 'w') as f:
        for v in zip(list(pmi.keys()), list(pmi.values())):
            f.write(" ".join(v[0])+" "+str(v[1])+"\n")
        
    with open('/tmp/rows.pmi', 'w') as f:
        for r in rows:
            f.write(r+"\n")
            
    with open('/tmp/cols.pmi', 'w') as f:
        for c in cols:
            f.write(c+"\n")
            
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
                if key in we.index_to_key and value in we.index_to_key:
                    scores[(key, value)] = (we[key], we[value], 1)

    eval_values = list(test_cd.values())
    for idx, key in enumerate(list(test_cd.keys())):
        if idx+1<len(eval_values):
            for value in eval_values[idx+1]:
                
                if len(vocab)>0:
                    if value not in vocab or key not in vocab:
                        continue
                else:
                    if key in we.index_to_key and value in we.index_to_key:
                        scores[(key, value)] = (we[key], we[value], 0)

    return scores
    
def loss(scores):
    cos_loss = nn.CosineEmbeddingLoss()
    x = [i[0] for i in scores.values()]
    y = [i[1] for i in scores.values()]
    target = [i[2] for i in scores.values()]
    return cos_loss(torch.tensor(x, dtype=torch.float64),torch.tensor(y, dtype=torch.float64), torch.tensor(target))

    
def get_vocab(wes_fs):
    vocabs = []
    for wes_f in wes_fs:
        vocabs.append(KeyedVectors.load_word2vec_format(wes_f, binary=False).index_to_key)
        
    joint_vocab = vocabs[0]
    for vocab in vocabs:
        joint_vocab = numpy.intersect1d(joint_vocab, vocab).tolist()
        
    return joint_vocab