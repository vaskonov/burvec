from composes.semantic_space.space import Space
from composes.transformation.scaling.ppmi_weighting import PpmiWeighting
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

def build_pairs(post_lemma, ng):
    print("build_pairs: "+str(ng))
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

            
def pmi(dim=0):
    print("pmi: "+str(dim))
    my_space = Space.build(data = data, rows = rows, cols = cols, format = "sm")
#     Positive Point-wise Mutual Information
    my_space = my_space.apply(PpmiWeighting())

    if dim!=0:
        my_space = my_space.apply(Svd(dim))


    my_space.export("/tmp/space", format='dm')
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
