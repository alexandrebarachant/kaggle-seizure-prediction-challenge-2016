import json
import os
import shutil
from preprocessors.filtering import run_filter_preprocessor
from functools import partial
from multiprocessing import Pool
from utils.config_name_creator import create_time_data_name, create_fft_data_name
from preprocessors.fft import process_file, process_file_more


def get_files_paths(directory, extension='.mat'):
    filenames = sorted(os.listdir(directory))
    files_with_extension = [directory + '/' + f for f in filenames if f.endswith(extension) and not f.startswith('.')]
    return files_with_extension

def run_fft_preprocessor():
    with open('kaggle_SETTINGS.json') as f:
        settings_dict = json.load(f)
    # path
    input_data_path = settings_dict['path']['processed_data_path'] + '/' + create_time_data_name(settings_dict)
    output_data_path = settings_dict['path']['processed_data_path'] + '/' + create_fft_data_name(settings_dict)

    # params
    nfreq_bands = settings_dict['preprocessor']['nfreq_bands']
    win_length_sec = settings_dict['preprocessor']['win_length_sec']
    stride_sec = settings_dict['preprocessor']['stride_sec']
    features = settings_dict['preprocessor']['features']

    if not os.path.exists(input_data_path):
        run_filter_preprocessor()

    if not os.path.exists(output_data_path):
        os.makedirs(output_data_path)
    shutil.copy2('kaggle_SETTINGS.json', output_data_path + '/kaggle_SETTINGS.json')

    subjects = ['train_1', 'train_2', 'train_3',
                'test_1_new', 'test_2_new', 'test_3_new']
    for subject in subjects:
        print '>> fft', subject
        read_dir = input_data_path + '/' + subject
        write_dir = output_data_path + '/' + subject

        if not os.path.exists(write_dir):
            os.makedirs(write_dir)

        raw_files = get_files_paths(read_dir)
        pool = Pool(8)
        part_f = partial(process_file, read_dir, write_dir, nfreq_bands,
                         win_length_sec, stride_sec, features)
        pool.map(part_f, raw_files)


def run_fft_more_preprocessor():
    with open('kaggle_SETTINGS_more.json') as f:
        settings_dict = json.load(f)
    # path
    input_data_path = settings_dict['path']['processed_data_path'] + '/' + create_time_data_name(settings_dict)
    output_data_path = settings_dict['path']['processed_data_path'] + '/' + 'combine' + create_fft_data_name(settings_dict)

    # params
    nfreq_bands = settings_dict['preprocessor']['nfreq_bands']
    win_length_sec = settings_dict['preprocessor']['win_length_sec']
    stride_sec = settings_dict['preprocessor']['stride_sec']
    features = settings_dict['preprocessor']['features']

    if not os.path.exists(input_data_path):
        run_filter_preprocessor()

    if not os.path.exists(output_data_path):
        os.makedirs(output_data_path)
    shutil.copy2('kaggle_SETTINGS_more.json', output_data_path +
                 '/kaggle_SETTINGS.json')

    subjects = ['train_1', 'train_2', 'train_3',
                'test_1_new', 'test_2_new', 'test_3_new']
    for subject in subjects:
        print '>> fft', subject
        read_dir = input_data_path + '/' + subject
        write_dir = output_data_path + '/' + subject

        if not os.path.exists(write_dir):
            os.makedirs(write_dir)

        raw_files = get_files_paths(read_dir)
        pool = Pool(8)
        part_f = partial(process_file_more, read_dir, write_dir, nfreq_bands,
                         win_length_sec, stride_sec, features)
        pool.map(part_f, raw_files)

if __name__ == '__main__':
    run_fft_preprocessor()
    run_fft_more_preprocessor()
