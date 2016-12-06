from glob import glob
from scipy.stats import rankdata
import pandas as pd
import numpy as np
default = pd.read_csv('./csv_files/sample_submission.csv', index_col=0)
base = './submissions/'

weights = {'Alex_Gilberto_all_flat_dataset_XGB.csv': 1.,
           'Alex_Gilberto_autocorrmat_TS_XGB.csv': 1.,
           'Alex_Gilberto_coherences_transposed_TS_XGB.csv': 1.,
           'Alex_Gilberto_relative_log_power_XGB.csv': 1.,
           'Feng_6bands.csv': 1.,
           'Feng_knn.csv': 1.,
           'Feng_knnmorefeature.csv': 1.,
           'Feng_morefeatureslasso.csv': 1.,
           'Andriy_submission5_7_SVM.csv': 1.,
           'Andriy_submissionLR5_3_glmnet.csv': 1.,
           'Andriy_submissionXGB7_5mean.csv': 1.}

fnames = weights.keys()
normalize = np.float64(np.sum(weights.values()))

for w in weights:
    weights[w] /= normalize

res = []
out = []
for ii, fn in enumerate(fnames):
    res = pd.read_csv(base + fn, index_col='File')
    res.loc[res.index, 'Class'] = rankdata(res.Class) / len(res)
    default.loc[default.index, 'Class'] += weights[fn] * res.loc[default.index, 'Class']

default.to_csv('./submissions/Winning_submission.csv)
