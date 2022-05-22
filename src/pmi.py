from composes.semantic_space.space import Space
from composes.transformation.scaling.ppmi_weighting import PpmiWeighting
from composes.transformation.scaling.epmi_weighting import EpmiWeighting
from composes.transformation.scaling.plmi_weighting import PlmiWeighting
from composes.transformation.scaling.plog_weighting import PlogWeighting

from composes.transformation.dim_reduction.svd import Svd
from composes.matrix.sparse_matrix import SparseMatrix
from composes.similarity.cos import CosSimilarity
from collections import Counter

#data = './wiki-word2vec/pmi_no_suf/average-1.0-data'
#rows = './wiki-word2vec/pmi_no_suf/average-1.0-rows'
#cols = './wiki-word2vec/pmi_no_suf/average-1.0-cols'

data='/tmp/data.pmi'
rows='/tmp/rows.pmi'
cols='/tmp/cols.pmi'

def build_pairs(post_lemma, ng, min_count=10):
    print("build_pairs: "+str(ng))
    omit = ['BEG', 'END', 'UNK']
    word_context = []
    words = []
    vocabs = []

    for idx, text in enumerate(post_lemma):
        vocabs.extend(text.split(" "))
    
    vocabs_c = dict(Counter(vocabs))

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

    grams = dict(Counter(word_context))
    print("before filtering", len(grams.keys()))
    
    filter_grams = { pair:idx for pair, idx in grams.items() if vocabs_c[pair[0]]>=min_count and vocabs_c[pair[1]]>=min_count}
    print("before filtering", len(filter_grams.keys()))
    
    return filter_grams


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
    print("prepare_pmi")
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

            
def pmi(method='PpmiWeighting', dim=0):
    print("pmi: "+str(dim))
    my_space = Space.build(data = data, rows = rows, cols = cols, format = "sm")
#     Positive Point-wise Mutual Information
    print("apply", method)
    my_space = my_space.apply(eval(method + "()"))

    if dim != 0:
        my_space = my_space.apply(Svd(dim))
    
    print("exporting pmi...")
    my_space.export("/tmp/space", format='dm')
    print("done")
    return True

#from sklearn.decomposition import TruncatedSVD

#def svd(space, dim):
#    svdm = TruncatedSVD(n_components=dim, random_state=42)
#    B = svdm.fit_transform(space.cooccurrence_matrix.mat)
#    return Space(SparseMatrix(B), list(space.id2row), [],
#                     space.row2id.copy(), {}, [])

# print my_space.cooccurrence_matrix
# print my_space.id2column

#pdb.set_trace()
#get_sim(key, value, CosSimilarity())
#space.id2row

# # assert(build_pairs(['I like apples and alcohol'],2), )
# # print(build_pairs(['I like apples and alcohol is this OK'],3))
# assert(build_pairs(['I like apples and alcohol'],1), [('I', 'like'), ('like', 'I'), ('like', 'apples'), ('apples', 'like'), ('apples', 'and'), ('and', 'apples'), ('and', 'alcohol'), ('alcohol', 'and')])