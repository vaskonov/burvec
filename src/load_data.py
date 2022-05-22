import numpy as np
from torchtext import data, datasets
from sklearn.model_selection import KFold


class load_data(object):
    def __init__(self, language, SEED=1234):

        NUM = data.Field(lower = True)
        TEXT = data.Field(lower = True)
        LEMMA = data.Field(lower = True)
        UD_TAGS = data.Field(unk_token=None)
        
        fields = (("num", NUM), ("text", TEXT), ("lemma", LEMMA), ("udtags", UD_TAGS))
        
        self.train_data, self.val_data, self.test_data = datasets.UDPOS.splits(
                                                            fields,
                                                            root='./extrinsic/'+language,
                                                            train='train.txt',
                                                            validation='train.txt',
                                                            test='train.txt',
                                                         )
        self.SEED = SEED


    def get_fold_data(self, num_folds=10):
        
        NUM = data.Field(lower = True)
        TEXT = data.Field(lower = True)
        LEMMA = data.Field(lower = True)
        UD_TAGS = data.Field(unk_token=None)
        
        fields = (("num", NUM), ("text", TEXT), ("lemma", LEMMA), ("udtags", UD_TAGS))
        
        kf = KFold(n_splits=num_folds, shuffle=True, random_state=self.SEED)
        train_data_arr = np.array(self.train_data.examples)

        for train_index, val_index in kf.split(train_data_arr):
            yield(
                NUM,
                TEXT,
                LEMMA,
                UD_TAGS,
                data.Dataset(train_data_arr[train_index], fields=fields),
                data.Dataset(train_data_arr[val_index], fields=fields),
            )
    
    def get_test_data(self):
        return self.test_data