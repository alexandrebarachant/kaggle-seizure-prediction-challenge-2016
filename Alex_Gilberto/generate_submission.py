import os
import yaml
import numpy as np
import pandas as pd
from copy import deepcopy
from sklearn.metrics import roc_auc_score

from progressbar import Bar, ETA, Percentage, ProgressBar
from joblib import Parallel, delayed
from optparse import OptionParser
from time import time
from sklearn.pipeline import make_pipeline


def from_yaml_to_func(method, params):
    """Convert yaml to function"""
    prm = dict()
    if params is not None:
        for key, val in params.iteritems():
            prm[key] = eval(str(val))
    return eval(method)(**prm)

parser = OptionParser()

parser.add_option("-c", "--config",
                  dest="config", default="config.yml",
                  help="The config file")
parser.add_option("-p", "--preds",
                  dest="preds_only", default=False, action="store_true",
                  help="Compute only predictions")
parser.add_option("-v", "--validation",
                  dest="val_only", default=False, action="store_true",
                  help="Compute only validation")

(options, args) = parser.parse_args()

# load yaml file
yml = yaml.load(open(options.config))


# imports
for pkg, functions in yml['imports'].iteritems():
    stri = 'from ' + pkg + ' import ' + ','.join(functions)
    exec(stri)

# parse pipe function from parameters
pipe = []
for item in yml['model']:
    for method, params in item.iteritems():
        pipe.append(from_yaml_to_func(method, params))

# create pipeline
model = make_pipeline(*pipe)

datasets = yml['datasets']
n_jobs = yml['n_jobs']
safe_old = yml['safe_old']

if 'ignore_na' in yml.keys():
    ignore_na = yml['ignore_na']
else:
    ignore_na = True

def classify(split, ii, model):
    """classify one split"""

    # clean test features
    numSlices = features.shape[1]

    # train
    feat_tr_fold = np.concatenate([features[split['indices_train']],
                                   old_features[split['indices_old_train']]])
    feat_tr_fold = np.concatenate(feat_tr_fold, 0)

    y_tr_fold = np.concatenate([split['labels_train'],
                                split['labels_old_train']])
    y_tr_fold = y_tr_fold.repeat(numSlices)

    fnames_tr_fold = np.concatenate([split['fnames_train'],
                                     split['fnames_old_train']])
    fnames_tr_fold = fnames_tr_fold.repeat(numSlices)

    index_tr_fold = np.arange(len(y_tr_fold))


    # test
    feat_te_fold = np.concatenate([features[split['indices_test']],
                                   old_features[split['indices_old_test']]])
    feat_te_fold = np.concatenate(feat_te_fold, 0)

    fnames_te_fold = np.concatenate([split['fnames_test'],
                                     split['fnames_old_test']])
    fnames_te_fold = fnames_te_fold.repeat(numSlices)

    y_te_fold = np.concatenate([split['labels_test'],
                                split['labels_old_test']])
    y_te_fold = y_te_fold.repeat(numSlices)

    index_te_fold = np.arange(len(y_te_fold))

    cv_fold1 = pd.DataFrame(0, index=index_te_fold, columns=['Preds'])
    cv_fold1['CV'] = ii
    cv_fold1['Labels'] = y_te_fold
    cv_fold1['File'] = fnames_te_fold

    cv_fold2 = pd.DataFrame(0, index=index_tr_fold, columns=['Preds'])
    cv_fold2['CV'] = ii
    cv_fold2['Labels'] = y_tr_fold
    cv_fold2['File'] = fnames_tr_fold

    if ignore_na:
        ix_good = np.array([np.sum(np.isnan(f)) == 0 for f in feat_tr_fold])
        feat_tr_fold = feat_tr_fold[ix_good]
        y_tr_fold = y_tr_fold[ix_good]
        fnames_tr_fold = fnames_tr_fold[ix_good]
        index_tr_fold = index_tr_fold[ix_good]

        ix_good = np.array([np.sum(np.isnan(f)) == 0 for f in feat_te_fold])
        y_te_fold = y_te_fold[ix_good]
        feat_te_fold = feat_te_fold[ix_good]
        fnames_te_fold = fnames_te_fold[ix_good]
        index_te_fold = index_te_fold[ix_good]

    if not options.preds_only:
        clf = deepcopy(model)
        clf.fit(feat_tr_fold, y_tr_fold)
        if getattr(clf, 'predict_proba', None):
            cv_fold1.loc[index_te_fold, 'Preds'] = clf.predict_proba(feat_te_fold)[:, 1]
        else:
            cv_fold1.loc[index_te_fold, 'Preds'] = clf.predict(feat_te_fold)
        pbar.update(1)

        clf = deepcopy(model)
        clf.fit(feat_te_fold, y_te_fold)
        if getattr(clf, 'predict_proba', None):
            cv_fold2.loc[index_tr_fold, 'Preds'] = clf.predict_proba(feat_tr_fold)[:, 1]
        else:
            cv_fold2.loc[index_tr_fold, 'Preds'] = clf.predict(feat_tr_fold)
        pbar.update(2)

    if not options.val_only:
        feat_tr = np.concatenate([feat_tr_fold, feat_te_fold])
        y_tr = np.concatenate([y_tr_fold, y_te_fold])

        del(feat_tr_fold)
        del(feat_te_fold)

        clf = deepcopy(model)
        clf.fit(feat_tr, y_tr)

    return cv_fold1, cv_fold2, clf


sub = pd.read_csv('../csv_files/sample_submission.csv', index_col=0)
sub.Class = 0.0


tot_preds = []
tot_cv_fold1 = []
tot_cv_fold2 = []
t_init = time()

for subject in [1, 2, 3]:

    cv = np.load('./cv_splits/splits_alex_subject%d.npy' % subject)
    print len(cv)
    NCV = len(cv)

    pbar = ProgressBar(widgets=[Percentage(), Bar(), ETA()], maxval=3).start()

    features = []
    old_features = []
    for jj, dataset in enumerate(datasets):
        a = np.load('./features/%s/train%d.npz' % (dataset, subject))
        features.append(a['features'])

        a = np.load('./features/%s/test%d.npz' % (dataset, subject))
        old_features.append(a['features'])

    del(a)
    features = np.concatenate(features, -1)
    old_features = np.concatenate(old_features, -1)

    print features.shape

    cv_fold1, cv_fold2, clf = classify(cv[0], 0, model)
    # average prediction accross CV splits
    del(features)
    del(old_features)

    if not options.val_only:
        features_te = []
        for jj, dataset in enumerate(datasets):
            a = np.load('./features/%s/new_test%d.npz' % (dataset, subject))
            indices = a['fnames']
            features_te.append(a['features'])

        del(a)
        features_te = np.concatenate(features_te, -1)
        numSlices = features_te.shape[1]

        feat_te = np.concatenate(features_te, 0)
        index_te = np.arange(len(feat_te))

        indices_te = indices.repeat(numSlices)
        preds = pd.DataFrame(0, index=index_te, columns=['Class'])
        preds['File'] = indices_te

        if ignore_na:
            ix_good = np.array([np.sum(np.isnan(f)) == 0 for f in feat_te])
            indices_te = indices_te[ix_good]
            feat_te = feat_te[ix_good]
            index_te = index_te[ix_good]

        preds.loc[index_te, 'Class'] = clf.predict_proba(feat_te)[:, 1]
        pbar.update(3)

        del(features_te)
        res = preds
        tot_preds.append(res)

if not options.val_only:
    preds = pd.concat(tot_preds).groupby('File').max()
    output = '../submissions/%s.csv' % yml['output']
    preds.to_csv(output)

print time() - t_init
