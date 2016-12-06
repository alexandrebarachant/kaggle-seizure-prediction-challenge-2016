import numpy as np
import pandas as pd

class Hurst:
    """
    Hurst exponent per-channel, see http://en.wikipedia.org/wiki/Hurst_exponent
    Another description can be found here: http://www.ijetch.org/papers/698-W10024.pdf
    Kavya Devarajan, S. Bagyaraj, Vinitha Balasampath, Jyostna. E. and Jayasri. K.,
    "EEG-Based Epilepsy Detection and Prediction," International Journal of Engineering
    and Technology vol. 6, no. 3, pp. 212-216, 2014.
    """
    def get_name(self):
        return 'hurst'

    def apply(self, X):
        def apply_one(x):
            x -= x.mean()
            z = np.cumsum(x)
            r = (np.maximum.accumulate(z) - np.minimum.accumulate(z))[1:]
            s = pd.expanding_std(x)[1:]

            # prevent division by 0
            s[np.where(s == 0)] = 1e-12
            r += 1e-12

            y_axis = np.log(r / s)
            x_axis = np.log(np.arange(1, len(y_axis) + 1))
            x_axis = np.vstack([x_axis, np.ones(len(x_axis))]).T

            m, b = np.linalg.lstsq(x_axis, y_axis)[0]
            return m

        return np.apply_along_axis(apply_one, -1, X)

class PFD():
    """
    Petrosian fractal dimension per-channel
    Implementation derived from reading:
    http://arxiv.org/pdf/0804.3361.pdf
    F.S. Bao, D.Y.Lie,Y.Zhang,"A new approach to automated epileptic diagnosis using EEG
    and probabilistic neural network",ICTAI'08, pp. 482-486, 2008.
    """
    def get_name(self):
        return 'pfd'

    def pfd_for_ch(self, ch):
        diff = np.diff(ch, n=1, axis=0)

        asign = np.sign(diff)
        sign_changes = ((np.roll(asign, 1) - asign) != 0).astype(int)
        N_delta = np.count_nonzero(sign_changes)

        n = len(ch)
        log10n = np.log10(n)
        return log10n / (log10n + np.log10(n / (n + 0.4 * N_delta)))

    def apply(self, X):
        return np.array([self.pfd_for_ch(ch) for ch in X])


class HFD():
    """
    Higuchi fractal dimension per-channel
    Implementation derived from reading:
    http://arxiv.org/pdf/0804.3361.pdf
    F.S. Bao, D.Y.Lie,Y.Zhang,"A new approach to automated epileptic diagnosis using EEG
    and probabilistic neural network",ICTAI'08, pp. 482-486, 2008.
    """
    def __init__(self, kmax=2):
        self.kmax = kmax

    def hfd(self, X):
        N = len(X)
        Nm1 = float(N - 1)
        L = np.empty((self.kmax,))
        L[0] = np.sum(abs(np.diff(X, n=1))) # shortcut :)
        for k in xrange(2, self.kmax + 1):
            Lmks = np.empty((k,))
            for m in xrange(1, k + 1):
                i_end = (N - m) / k # int
                Lmk_sum = np.sum(abs(np.diff(X[np.arange(m - 1, m + (i_end + 1) * k - 1, k)], n=1)))
                Lmk = Lmk_sum * Nm1 / (i_end * k)
                Lmks[m-1] = Lmk

            L[k - 1] = np.mean(Lmks)

        a = np.empty((self.kmax, 2))
        a[:, 0] = np.log(1.0 / np.arange(1.0, self.kmax + 1.0))
        a[:, 1] = 1.0

        b = np.log(L)

        # find x by solving for ax = b
        x, residues, rank, s = np.linalg.lstsq(a, b)
        return x[0]



    def get_name(self):
        return 'hfd-%d' % self.kmax

    def apply(self, data, meta=None):
        return np.array([self.hfd(ch) for ch in data])
