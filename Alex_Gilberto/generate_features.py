import re
import os
import yaml
import numpy as np
import pandas as pd

from scipy.io import loadmat
from glob import glob

from progressbar import Bar, ETA, Percentage, ProgressBar
from joblib import Parallel, delayed
from optparse import OptionParser

from sklearn.pipeline import make_pipeline


def from_yaml_to_func(method, params):
    """Convert yaml to function"""
    prm = dict()
    if params is not None:
        for key, val in params.iteritems():
            prm[key] = eval(str(val))
    return eval(method)(**prm)


parser = OptionParser()

parser.add_option("-s", "--subject",
                  dest="subject", default=1,
                  help="The subject")
parser.add_option("-c", "--config",
                  dest="config", default="config.yml",
                  help="The config file")
parser.add_option("-o", "--old",
                  dest="old", default=False, action="store_true",
                  help="process the old test set")
parser.add_option("-n", "--njobs",
                  dest="njobs", default=8,
                  help="the number of jobs")

(options, args) = parser.parse_args()

subject = int(options.subject)
njobs = int(options.njobs)

# load yaml file
yml = yaml.load(open(options.config))

# output of the script
output = './features/%s' % yml['output']
# create forlder if it does not exist
if not os.path.exists(output):
    os.makedirs(output)

# imports
for pkg, functions in yml['imports'].iteritems():
    stri = 'from ' + pkg + ' import ' + ','.join(functions)
    exec(stri)

# parse pipe function from parameters
pipe = []
for item in yml['preproc']:
    for method, params in item.iteritems():
        pipe.append(from_yaml_to_func(method, params))

# create pipeline
preproc = make_pipeline(*pipe)

# parse pipe function from parameters
if 'postproc' in yml.keys():
    pipe = []
    for item in yml['postproc']:
        for method, params in item.iteritems():
            pipe.append(from_yaml_to_func(method, params))

    # create pipeline
    postproc = make_pipeline(*pipe)


reg = re.compile('.*(\d)_(\d*)_(\d).mat')
reg_test = re.compile('.*(new_%s_\d*.mat)' % subject)
reg_old_test = re.compile('.*(%s_\d*.mat)' % subject)
reg_fname = re.compile('.*(%s_\d*_\d.mat)' % subject)


def process_data_train(fname, ii):
    subj, indice, label = reg.findall(fname)[0]
    fn = reg_fname.findall(fname)[0]
    pbar.update(ii)
    data = loadmat(fname, squeeze_me=True, struct_as_record=False,
                   verify_compressed_data_integrity=False)['dataStruct']

    out = preproc.fit_transform(np.array([data.data.T]))
    if len(out) == 1:
        out = out[0]
    val = np.sum(np.isnan(out)) == 0
    return out, val, int(label), int(indice), (int(indice) - 1) / 6, data.sequence, fn


def process_data_test(fname, ii, reg_test=reg_test):
    idx = reg_test.findall(fname)[0]
    pbar.update(ii)
    data = loadmat(fname, squeeze_me=True, struct_as_record=False,
                   verify_compressed_data_integrity=False)['dataStruct']

    out = preproc.fit_transform(np.array([data.data.T]))
    if len(out) == 1:
        out = out[0]
    val = np.sum(np.isnan(out)) == 0

    return out, val, idx

base = '../data/train_%d/%d_' % (subject, subject)
fnames = (sorted(glob(base + '*_0.mat'),
                 key=lambda x: int(x.replace(base, '')[:-6])) +
          sorted(glob(base + '*_1.mat'),
                 key=lambda x: int(x.replace(base, '')[:-6])))

# ignore file not safe
ignore = pd.read_csv('../csv_files/train_and_test_data_labels_safe.csv', index_col=0)

fnames_finals = []
for fname in fnames:
    ba = '../data/train_%d/' % subject
    fn = fname.replace(ba, '')
    if ignore.loc[fn, 'safe'] == 1:
        fnames_finals.append(fname)
fnames = fnames_finals

pbar = ProgressBar(widgets=[Percentage(), Bar(), ETA()], maxval=len(fnames)).start()


res = Parallel(n_jobs=njobs)(delayed(process_data_train)(fname=fname, ii=ii)
                             for ii, fname in enumerate(fnames))

features, valid, y, idx, clips, sequence, fnames = zip(*res)

features = np.array(features)
sequence = np.array(sequence)
idx = np.array(idx)
y = np.array(y)
clips = np.array(clips)
valid = np.array(valid)
fnames = np.array(fnames)

if 'postproc' in yml.keys():
    print("\npost process training data")
    features = postproc.fit_transform(features[valid], y[valid])
    out_shape = list(features.shape)
    out_shape[0] = len(valid)
    features_final = np.ones(tuple(out_shape)) * np.nan
    features_final[valid] = features
else:
    features_final = features

np.savez('%s/train%d.npz' % (output, subject), features=features_final,
         y=y, sequence=sequence, idx=idx, clips=clips, valid=valid,
         fnames=fnames)
# clear memory
res = []
features = []

print('Done Training !!!')

if options.old:
    base = '../data/test_%d/%d_' % (subject, subject)
    fnames = sorted(glob(base + '*.mat'),
                    key=lambda x: int(x.replace(base, '')[:-4]))

    ignore = pd.read_csv('../csv_files/train_and_test_data_labels_safe.csv', index_col=0)

    fnames_finals = []
    for fname in fnames:
        ba = '../data/test_%d/' % subject
        fn = fname.replace(ba, '')
        if fn in ignore.index.values:
            fnames_finals.append(fname)

    #fnames = fnames_finals

    pbar = ProgressBar(widgets=[Percentage(), Bar(), ETA()],
                       maxval=len(fnames)).start()

    res = Parallel(n_jobs=njobs)(delayed(process_data_test)(fname=fname, ii=ii, reg_test=reg_old_test)
                                 for ii, fname in enumerate(fnames))

    features, valid, idx = zip(*res)

    features = np.array(features)
    idx = np.array(idx)
    valid = np.array(valid)
    if 'postproc' in yml.keys():
        print("\npost process test data")
        features = postproc.transform(features[valid])
        out_shape = list(features.shape)
        out_shape[0] = len(valid)
        features_final = np.ones(tuple(out_shape)) * np.nan
        features_final[valid] = features
    else:
        features_final = features

    np.savez('%s/test%d.npz' % (output, subject), features=features_final,
             fnames=idx, valid=valid)
    print('Done Old Test !!!')
    # clear memory
    res = []
    features = []
base = '../data/test_%d_new/new_%d_' % (subject, subject)
fnames = sorted(glob(base + '*.mat'),
                key=lambda x: int(x.replace(base, '')[:-4]))

pbar = ProgressBar(widgets=[Percentage(), Bar(), ETA()],
                   maxval=len(fnames)).start()

res = Parallel(n_jobs=njobs)(delayed(process_data_test)(fname=fname, ii=ii, reg_test=reg_test)
                             for ii, fname in enumerate(fnames))

features, valid, idx = zip(*res)

features = np.array(features)
idx = np.array(idx)
valid = np.array(valid)
if 'postproc' in yml.keys():
    print("\npost process test data")
    features = postproc.transform(features[valid])
    out_shape = list(features.shape)
    out_shape[0] = len(valid)
    features_final = np.ones(tuple(out_shape)) * np.nan
    features_final[valid] = features
else:
    features_final = features

np.savez('%s/new_test%d.npz' % (output, subject), features=features_final,
         fnames=idx, valid=valid)
print('Done New Test!!!')
