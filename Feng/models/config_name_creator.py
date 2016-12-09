def create_time_data_name(settings_dict):
    highcut = settings_dict['preprocessor']['highcut']
    lowcut = settings_dict['preprocessor']['lowcut']
    config_name = 'lowcut' + str(lowcut) + 'highcut' + str(highcut)
    return config_name


def create_fft_data_name(settings_dict):
    return 'fft_' + settings_dict['preprocessor']['features'] + '_' + create_time_data_name(settings_dict) \
           + 'nfreq_bands' + str(settings_dict['preprocessor']['nfreq_bands']) \
           + 'win_length_sec' + str(settings_dict['preprocessor']['win_length_sec']) \
           + 'stride_sec' + str(settings_dict['preprocessor']['stride_sec'])


