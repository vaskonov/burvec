from composes.semantic_space.space import Space
from composes.transformation.scaling.ppmi_weighting import PpmiWeighting
from composes.transformation.dim_reduction.svd import Svd
from composes.matrix.sparse_matrix import SparseMatrix
from composes.similarity.cos import CosSimilarity
import pdb

#data = './wiki-word2vec/pmi_no_suf/average-1.0-data'
#rows = './wiki-word2vec/pmi_no_suf/average-1.0-rows'
#cols = './wiki-word2vec/pmi_no_suf/average-1.0-cols'

data='/tmp/data.pmi'
rows='/tmp/rows.pmi'
cols='/tmp/cols.pmi'

my_space = Space.build(data = data, rows = rows, cols = cols, format = "sm")
my_space = my_space.apply(PpmiWeighting())

#pdb.set_trace()


my_space = my_space.apply(Svd(200))


my_space.export("/tmp/space", format='dm')
#from sklearn.decomposition import TruncatedSVD

#def svd(space, dim):
#    svdm = TruncatedSVD(n_components=dim, random_state=42)
#    B = svdm.fit_transform(space.cooccurrence_matrix.mat)
#    return Space(SparseMatrix(B), list(space.id2row), [],
#                     space.row2id.copy(), {}, [])

# print my_space.cooccurrence_matrix
# print my_space.id2column

pdb.set_trace()
#get_sim(key, value, CosSimilarity())
#space.id2row
