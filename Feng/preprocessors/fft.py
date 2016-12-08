import numpy as np
from scipy.io import savemat, loadmat
from pandas import DataFrame
from sklearn import preprocessing


class CorrelationMatrix:
    """
    Calculate correlation coefficients matrix across all EEG channels.
    """
    def get_name(self):
        return 'corr-mat'

    def apply(self, data):

        for ch in data:
            if np.alltrue(ch == 0.0):
                ch[-1] += 0.00001
        return np.corrcoef(data)


class Eigenvalues:
    """
    Take eigenvalues of a matrix, and sort them by magnitude in order to
    make them useful as features (as they have no inherent order).
    """
    def get_name(self):
        return 'eigenvalues'

    def apply(self, data):
        w, v = np.linalg.eig(data)
        w = np.absolute(w)
        w.sort()
        return w


def upper_right_triangle(matrix):
    accum = []
    for i in range(matrix.shape[0]):
        for j in range(i+1, matrix.shape[1]):
            accum.append(matrix[i, j])

    return np.array(accum)


def group_into_bands(fft, fft_freq, nfreq_bands):
    if nfreq_bands == 178:
        bands = range(1, 180)
    elif nfreq_bands == 4:
        bands = [0.1, 4, 8, 12, 30]
    elif nfreq_bands == 6:
        bands = [0.1, 4, 8, 12, 30, 70, 180]
    # http://onlinelibrary.wiley.com/doi/10.1111/j.1528-1167.2011.03138.x/pdf
    elif nfreq_bands == 8:
        bands = [0.1, 4, 8, 12, 30, 50, 70, 100, 180]
    elif nfreq_bands == 12:
        bands = [0.1, 4, 8, 12, 30, 40, 50, 60, 70, 85, 100, 140, 180]
    elif nfreq_bands == 9:
        bands = [0.1, 4, 8, 12, 21, 30, 50, 70, 100, 180]
    else:
        raise ValueError('wrong number of frequency bands')
    freq_bands = np.digitize(fft_freq, bands)
    df = DataFrame({'fft': fft, 'band': freq_bands})
    df = df.groupby('band').mean()
    return df.fft[1:-1]


def compute_fft(x, data_length_sec, sampling_frequency, nfreq_bands, win_length_sec, stride_sec, features):
    n_channels = x.shape[1]
    n_timesteps = (data_length_sec - win_length_sec) / stride_sec + 1
    n_fbins = nfreq_bands + 1 if 'std' in features else nfreq_bands

    x2 = np.zeros((n_channels, n_fbins, n_timesteps))
    for i in range(n_channels):
        xc = np.zeros((n_fbins, n_timesteps))
        for frame_num, w in enumerate(range(0, data_length_sec - win_length_sec + 1, stride_sec)):
            xw = x[w * sampling_frequency: (w + win_length_sec) * sampling_frequency,i]
            fft = np.log10(np.absolute(np.fft.rfft(xw)))
            fft_freq = np.fft.rfftfreq(n=xw.shape[-1], d=1.0 / sampling_frequency)
            xc[:nfreq_bands, frame_num] = group_into_bands(fft, fft_freq, nfreq_bands)
            if 'std' in features:
                xc[-1, frame_num] = np.std(xw)
        x2[i, :, :] = xc
    return x2


def compute_frequencydomaincoef(x, data_length_sec, sampling_frequency, nfreq_bands, win_length_sec, stride_sec):
    n_channels = x.shape[1]
    n_timesteps = (data_length_sec - win_length_sec) / stride_sec + 1
    n_fbins = nfreq_bands
    xfreq = np.zeros((n_timesteps, 136))
    x2 = np.zeros((n_channels, n_fbins, n_timesteps))
    for i in range(n_channels):
        xc = np.zeros((n_fbins, n_timesteps))
        for frame_num, w in enumerate(range(0, data_length_sec - win_length_sec + 1, stride_sec)):
            xw = x[w * sampling_frequency: (w + win_length_sec) * sampling_frequency,i]#window
            fft = np.log10(np.absolute(np.fft.rfft(xw)))
            fft_freq = np.fft.rfftfreq(n=xw.shape[-1], d=1.0 / sampling_frequency) ## return the FFT sample  frequency
            xc[:nfreq_bands, frame_num] = group_into_bands(fft, fft_freq, nfreq_bands)
        x2[i, :, :] = xc
    for j in range(n_timesteps):

        x2[:, :, j][np.isneginf(x2[:, :, j])] = 0
        scaled = preprocessing.scale(x2[:, :, j], axis=0)

        matrix = CorrelationMatrix().apply(scaled)
        matrix[np.isneginf(matrix)] = 0
        matrix[np.isnan(matrix)] = 0

        eigenvalue = Eigenvalues().apply(matrix)

        freqdomaincor = upper_right_triangle(matrix)
        xfreq[j, :] = np.concatenate((freqdomaincor, eigenvalue))
    xfreq[np.isneginf(xfreq)] = 0
    xfreq[np.isnan(xfreq)] = 0
    return xfreq


def compute_timedomaincoef(x,data_length_sec,sampling_frequency, win_length_sec, stride_sec):
    n_channels = x.shape[1]
    n_timesteps = (data_length_sec - win_length_sec) / stride_sec + 1
    n_windowlength = x.shape[0] / n_timesteps
    timedomaincorrelation = np.zeros((n_timesteps, 120 + 16))

    for frame_num, w in enumerate(range(0, data_length_sec - win_length_sec + 1, stride_sec)):
        xcor = np.zeros((n_channels, n_windowlength))
        for i in range(n_channels):
            xw = x[w * sampling_frequency: (w + win_length_sec) * sampling_frequency, i]
            xcor[i, :] = xw
        xcor[np.isneginf(xcor)] = 0
        xcor = preprocessing.scale(xcor, axis=0)
        cormatrix = CorrelationMatrix().apply(xcor)
        cormatrix[np.isneginf(cormatrix)] = 0
        cormatrix[np.isnan(cormatrix)] = 0
        eigenvalue = Eigenvalues().apply(cormatrix)
        timedomaincorrelation[frame_num, :] = np.concatenate((upper_right_triangle(cormatrix), eigenvalue))
    return timedomaincorrelation


def compute_fft_more(x, data_length_sec, sampling_frequency, nfreq_bands, win_length_sec, stride_sec, features):
    n_channels = x.shape[1]
    n_timesteps = (data_length_sec - win_length_sec) / stride_sec + 1
    n_fbins = nfreq_bands + 1 if 'std' in features else nfreq_bands

    x2 = np.zeros((n_channels, n_fbins, n_timesteps))
    for i in range(n_channels):
        xc = np.zeros((n_fbins, n_timesteps))
        for frame_num, w in enumerate(range(0, data_length_sec - win_length_sec + 1, stride_sec)):
            xw = x[w * sampling_frequency: (w + win_length_sec) * sampling_frequency,i]#window
            fft = np.log10(np.absolute(np.fft.rfft(xw)))
            fft_freq = np.fft.rfftfreq(n=xw.shape[-1], d=1.0 / sampling_frequency) ## return the FFT sample  frequency
            xc[:nfreq_bands, frame_num] = group_into_bands(fft, fft_freq, nfreq_bands)
            if 'std' in features:
                xc[-1, frame_num] = np.std(xw)
        x2[i, :, :] = xc
    x2 = x2.reshape(n_channels * n_fbins, n_timesteps)
    return x2.transpose()


def process_file(read_dir, write_dir, nfreq_bands, win_length_sec, stride_sec,
                 features, raw_file_path):
    preprocessed_file_path = raw_file_path.replace(read_dir, write_dir)

    d = loadmat(raw_file_path, squeeze_me=True,
                verify_compressed_data_integrity=False)
    x = d['data']
    data_length_sec = d['data_length_sec']
    sampling_frequency = d['sampling_frequency']

    new_x = compute_fft(x, data_length_sec, sampling_frequency, nfreq_bands,
                        win_length_sec, stride_sec, features)

    data_dict = {'data': new_x, 'data_length_sec': data_length_sec,
                 'sampling_frequency': sampling_frequency}
    savemat(preprocessed_file_path, data_dict)


def process_file_more(read_dir, write_dir, nfreq_bands, win_length_sec,
                      stride_sec, features, raw_file_path):
    preprocessed_file_path = raw_file_path.replace(read_dir, write_dir)

    d = loadmat(raw_file_path, squeeze_me=True,
                verify_compressed_data_integrity=False)
    x = d['data']
    data_length_sec = d['data_length_sec']
    sampling_frequency = d['sampling_frequency']

    new_x_fft = compute_fft_more(x, data_length_sec, sampling_frequency,
                                 nfreq_bands, win_length_sec, stride_sec,
                                 features)

    new_x_corr = compute_timedomaincoef(x, data_length_sec, sampling_frequency,
                                        win_length_sec, stride_sec)

    new_x_frecorr = compute_frequencydomaincoef(x, data_length_sec,
                                                sampling_frequency,
                                                nfreq_bands, win_length_sec,
                                                stride_sec)

    new_x = np.concatenate((new_x_fft, new_x_corr, new_x_frecorr), axis=1)
    data_dict = {'data': new_x, 'data_length_sec': data_length_sec,
                 'sampling_frequency': sampling_frequency}
    savemat(preprocessed_file_path, data_dict)
