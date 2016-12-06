import numpy as np
import scipy as sp
from copy import deepcopy
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline
from pyriemann.estimation import Shrinkage
from pyriemann.tangentspace import TangentSpace
from joblib import Parallel, delayed

class CoherenceToEigen(BaseEstimator, TransformerMixin):
    """Remove dropped packet."""

    def __init__(self, metric='logeuclid', tsupdate=True):
        """Init."""
        self.tsupdate = tsupdate
        self.metric = metric

    def fit(self, X, y=None):
        """
        Fit, do nothing
        """
        return self

    def transform(self, X):
        """
        Detect and remove dropped.
        """
        features = []
        for f in range(X.shape[-1]):
            tmp = []
            for x in X[:, :, :, f]:
                tmp.append(sp.linalg.eigvalsh(x))
            features.append(np.array(tmp))
        features = np.concatenate(features, 1)
        return features


def fit_one_ts(X, metric, tsupdate):
    ts = make_pipeline(Shrinkage(1e-9),
                       TangentSpace(metric=metric,
                       tsupdate=tsupdate))
    return ts.fit(X)

def apply_one_ts(X, ts):
    return ts.transform(X)

class CoherenceToTangent(BaseEstimator, TransformerMixin):
    """Remove dropped packet."""

    def __init__(self, metric='logeuclid', tsupdate=True, n_jobs=1):
        """Init."""
        self.tsupdate = tsupdate
        self.metric = metric
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        """
        Fit, do nothing
        """
        if self.n_jobs == 1:
            self.ts_ = []
            for f in range(X.shape[-1]):
                self.ts_.append(fit_one_ts(X=X[:, :, :, f],
                                           metric=self.metric,
                                           tsupdate=self.tsupdate))
        else:
            self.ts_ = Parallel(n_jobs=self.n_jobs)(delayed(fit_one_ts)(X=X[:, :, :, f], metric=self.metric, tsupdate=self.tsupdate)
                                          for f in range(X.shape[-1]))
        return self

    def transform(self, X):
        """
        Detect and remove dropped.
        """
        features = []
        if self.n_jobs == 1:
            for f in range(X.shape[-1]):
                features.append(apply_one_ts(self.ts_[f], X[:, :, :, f]))
        else:
            features = Parallel(n_jobs=self.n_jobs)(delayed(apply_one_ts)(X=X[:, :, :, f], ts=self.ts_[f])
                                          for f in range(X.shape[-1]))
        features = np.concatenate(features, 1)
        return features


class CoherenceToTangent2(BaseEstimator, TransformerMixin):
    """Remove dropped packet."""

    def __init__(self, metric='logeuclid'):
        """Init."""
        self.metric = metric

    def fit(self, X, y=None):
        """
        Fit, do nothing
        """
        return self

    def transform(self, X):
        """
        Detect and remove dropped.
        """
        features = []
        for x in X:
            ts = TangentSpace(metric=self.metric)
            tmp = ts.fit_transform(x.transpose(2, 0, 1))
            features.append(tmp.ravel())
        features = np.array(features)
        return features


class ApplyOnLastAxis(BaseEstimator, TransformerMixin):
    """Remove dropped packet."""

    def __init__(self, model=TangentSpace()):
        """Init."""
        self.model = model

    def fit(self, X, y=None):
        """
        Fit, do nothing
        """
        models = []
        for f in range(X.shape[-1]):
            m = deepcopy(self.model)
            m.fit(X[..., f], y)
            models.append(m)
        self.models = models
        return self

    def transform(self, X):
        """
        Detect and remove dropped.
        """
        features = []
        for f in range(X.shape[-1]):
            tmp = self.models[f].transform(X[..., f])
            features.append(tmp)
        features = np.array(features)
        features = np.moveaxis(features, 0, -1)
        return features


class BagElectrodes(BaseEstimator, TransformerMixin):
    """Remove dropped packet."""

    def __init__(self, model, size=4, n_bags=10, random_state=42):
        """Init."""
        self.size = size
        self.n_bags = n_bags
        self.model = model
        self.random_state = random_state

    def fit(self, X, y=None):
        """
        Fit, do nothing
        """
        rs = np.random.RandomState(self.random_state)
        Nelec = X.shape[1]
        self.bags = []
        for ii in range(self.n_bags):
            self.bags.append(rs.permutation(Nelec)[0:self.size])
        self.models = []
        for bag in self.bags:
            clf = deepcopy(self.model)
            clf.fit(X[:, bag][:, :, bag], y)
            self.models.append(clf)
        return self

    def predict_proba(self, X):
        """
        Detect and remove dropped.
        """
        preds = []
        for ii, bag in enumerate(self.bags):
            preds.append(self.models[ii].predict_proba(X[:, bag][:, :, bag]))
        preds = np.mean(preds, 0)
        return preds
