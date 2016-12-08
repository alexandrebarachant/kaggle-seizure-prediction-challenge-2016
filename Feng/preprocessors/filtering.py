import scipy as sc
import scipy.signal
import json
import os
import shutil
import numpy as np
from functools import partial
from multiprocessing import Pool
from scipy.io import loadmat, savemat
from utils.config_name_creator import create_time_data_name
import pandas as pd


def get_files_paths_filters(directory, extension='.mat'):
    filenames = sorted(os.listdir(directory))
    files_with_extension = [directory + '/' + f for f in filenames if f.endswith(extension) and not f.startswith('.')]
    return files_with_extension


def filter(x, new_sampling_frequency, data_length_sec, lowcut, highcut):
    x1 = scipy.signal.resample(x, new_sampling_frequency * data_length_sec,
                               axis=0)
    nyq = 0.5 * new_sampling_frequency
    b, a = sc.signal.butter(5, np.array([lowcut, highcut]) / nyq, btype='band')
    x_filt = sc.signal.lfilter(b, a, x1, axis=0)
    return np.float32(x_filt)


def process_file_filter(lowcut, highcut, raw_file_path):
    preprocessed_file_path = raw_file_path[1]

    d = loadmat(raw_file_path[0], verify_compressed_data_integrity=False)
    sample = ''
    for key in d.keys():
        if 'dataStruct' in key:
            sample = key
            break
    x = np.array(d[sample][0][0][0], dtype='float32')
    data_length_sec = 600
    new_sampling_frequency = 400
    new_x = filter(x, new_sampling_frequency, data_length_sec, lowcut, highcut)
    data_dict = {'data': new_x, 'data_length_sec': data_length_sec,
                 'sampling_frequency': new_sampling_frequency}
    savemat(preprocessed_file_path, data_dict, do_compression=True)


def run_filter_preprocessor():
    with open('kaggle_SETTINGS.json') as f:
        settings_dict = json.load(f)

    raw_data_path = settings_dict['path']['raw_data_path']
    processed_data_path = (settings_dict['path']['processed_data_path'] + '/' +
                           create_time_data_name(settings_dict))

    if not os.path.exists(processed_data_path):
        os.makedirs(processed_data_path)
    shutil.copy2('kaggle_SETTINGS.json', processed_data_path +
                 '/kaggle_SETTINGS.json')

    highcut = settings_dict['preprocessor']['highcut']
    lowcut = settings_dict['preprocessor']['lowcut']

    safe = pd.read_csv('../csv_files/train_and_test_data_labels_safe.csv',
                       index_col=0)
    # select safe files
    safe = safe[safe.safe == 1]

    for subject in [1, 2, 3]:
        print '>> filtering', subject
        write_dir = processed_data_path + '/train_' + str(subject)
        train_dir = raw_data_path + '/train_' + str(subject)
        test_dir = raw_data_path + '/test_' + str(subject)

        if not os.path.exists(write_dir):
            os.makedirs(write_dir)

        su = [int(fn[0]) == subject for fn in safe.index]
        raw_safe_files = list(safe[su].index)
        raw_files = []
        k = 151
        for fn in raw_safe_files:
            if os.path.exists(train_dir + '/' + fn):
                tmp = [train_dir + '/' + fn,  write_dir + '/' + fn]
            else:
                new_filename = '%d_%d_1.mat' % (subject, k)
                tmp = [test_dir + '/' + fn, write_dir + '/' + new_filename]
                k += 1
            raw_files.append(tmp)

        pool = Pool(8)
        part_f = partial(process_file_filter, lowcut, highcut)
        pool.map(part_f, raw_files)


    # process test data
    subjects = ['test_1_new', 'test_2_new', 'test_3_new']
    for subject in subjects:
        print '>> filtering', subject
        read_dir = raw_data_path + '/' + subject
        write_dir = processed_data_path + '/' + subject

        if not os.path.exists(write_dir):
            os.makedirs(write_dir)

        raw_files = get_files_paths_filters(read_dir)
        raw_files = [[fn, fn.replace(read_dir, write_dir)] for fn in raw_files]
        pool = Pool(8)
        part_f = partial(process_file_filter, lowcut, highcut)
        pool.map(part_f, raw_files)
