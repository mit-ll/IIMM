import os
from typing import Union

import torch
import numpy as np
from scipy import linalg
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from transformers import AutoTokenizer, BertModel, CLIPModel, CLIPProcessor, GPT2Tokenizer, GPT2Model

from utils import iterative_A


### GBC Score

# Code adapted from https://github.com/google-research/google-research/blob/8e752fede8468efb5b72f2d7f5fc3dccdc9e33bd/stable_transfer/transferability/gbc.py

def compute_bhattacharyya_distance(mu1, mu2, sigma1, sigma2):
  """Compute Bhattacharyya distance between diagonal or spherical Gaussians."""
  avg_sigma = (sigma1 + sigma2) / 2
  first_part = np.sum((mu1 - mu2)**2 / avg_sigma) / 8
  second_part = np.sum(np.log(avg_sigma))
  second_part -= 0.5 * (np.sum(np.log(sigma1)))
  second_part -= 0.5 * (np.sum(np.log(sigma2)))
  return first_part + 0.5 * second_part


def get_bhattacharyya_distance(per_class_stats, c1, c2, gaussian_type):
    """Return Bhattacharyya distance between 2 diagonal or spherical gaussians."""
    mu1 = per_class_stats[c1]['mean']
    mu2 = per_class_stats[c2]['mean']
    sigma1 = per_class_stats[c1]['variance']
    sigma2 = per_class_stats[c2]['variance']
    if gaussian_type == 'spherical':
        sigma1 = np.mean(sigma1)
        sigma2 = np.mean(sigma2)
    return compute_bhattacharyya_distance(mu1, mu2, sigma1, sigma2)


def compute_per_class_mean_and_variance(features, target_labels, unique_labels):
    """Compute features mean and variance for each class."""
    per_class_stats = {}
    for label in unique_labels:
        label = int(label)  # For correct indexing
        per_class_stats[label] = {}
        class_ids = target_labels == label
        class_features = features[class_ids]
        mean = np.mean(class_features, axis=0)
        variance = np.var(class_features, axis=0)
        per_class_stats[label]['mean'] = mean
        # Avoid 0 variance in cases of constant features with tf.maximum
        per_class_stats[label]['variance'] = np.maximum(variance, 1e-4)
    return per_class_stats


def GBC(features, target_labels, gaussian_type):
    """Compute Gaussian Bhattacharyya Coefficient (GBC).

    Args:
    features: source features from the target data.
    target_labels: ground truth labels in the target label space.
    gaussian_type: type of gaussian used to represent class features. The
        possibilities are spherical (default) or diagonal.

    Returns:
    gbc: transferability metric score.
    """

    assert gaussian_type in ('diagonal', 'spherical')
    unique_labels = np.unique(target_labels)
    per_class_stats = compute_per_class_mean_and_variance(
        features, target_labels, unique_labels)

    per_class_bhattacharyya_distance = []
    for c1 in unique_labels:
        temp_metric = []
        for c2 in unique_labels:
            if c1 != c2:
                bhattacharyya_distance = get_bhattacharyya_distance(
                    per_class_stats, int(c1), int(c2), gaussian_type)
                temp_metric.append(np.exp(-bhattacharyya_distance))
        per_class_bhattacharyya_distance.append(np.sum(temp_metric))
    gbc = -np.sum(per_class_bhattacharyya_distance)

    return gbc

# Code borrowed from https://github.com/OpenGVLab/Multitask-Model-Selector

### SFDA ###

def _cov(X, shrinkage=-1):
    emp_cov = np.cov(np.asarray(X).T, bias=1)
    if shrinkage < 0:
        return emp_cov
    n_features = emp_cov.shape[0]
    mu = np.trace(emp_cov) / n_features
    shrunk_cov = (1.0 - shrinkage) * emp_cov
    shrunk_cov.flat[:: n_features + 1] += shrinkage * mu
    return shrunk_cov


def softmax(X, copy=True):
    if copy:
        X = np.copy(X)
    max_prob = np.max(X, axis=1).reshape((-1, 1))
    X -= max_prob
    np.exp(X, X)
    sum_prob = np.sum(X, axis=1).reshape((-1, 1))
    X /= sum_prob
    return X


def _class_means(X, y):
    """Compute class means.
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input data.
    y : array-like of shape (n_samples,) or (n_samples, n_targets)
        Target values.
    Returns
    -------
    means : array-like of shape (n_classes, n_features)
        Class means.
    means : array-like of shape (n_classes, n_features)
        Outer classes means.
    """
    classes, y = np.unique(y, return_inverse=True)
    cnt = np.bincount(y)
    means = np.zeros(shape=(len(classes), X.shape[1]))
    np.add.at(means, y, X)
    means /= cnt[:, None]

    means_ = np.zeros(shape=(len(classes), X.shape[1]))
    for i in range(len(classes)):
        means_[i] = (np.sum(means, axis=0) - means[i]) / (len(classes) - 1)    
    return means, means_


class SFDA():
    def __init__(self, shrinkage=None, priors=None, n_components=None):
        self.shrinkage = shrinkage
        self.priors = priors
        self.n_components = n_components
        
    def _solve_eigen(self, X, y, shrinkage):
        classes, y = np.unique(y, return_inverse=True)
        cnt = np.bincount(y)
        means = np.zeros(shape=(len(classes), X.shape[1]))
        np.add.at(means, y, X)
        means /= cnt[:, None]
        self.means_ = means
                
        cov = np.zeros(shape=(X.shape[1], X.shape[1]))
        for idx, group in enumerate(classes):
            Xg = X[y == group, :]
            cov += self.priors_[idx] * np.atleast_2d(_cov(Xg))
        self.covariance_ = cov

        Sw = self.covariance_  # within scatter
        if self.shrinkage is None:
            # adaptive regularization strength
            largest_evals_w = iterative_A(Sw, max_iterations=3)
            shrinkage = max(np.exp(-5 * largest_evals_w), 1e-10)
            self.shrinkage = shrinkage
        else:
            # given regularization strength
            shrinkage = self.shrinkage
        # print("Shrinkage: {}".format(shrinkage))
        # between scatter
        St = _cov(X, shrinkage=self.shrinkage) 

        # add regularization on within scatter   
        n_features = Sw.shape[0]
        mu = np.trace(Sw) / n_features
        shrunk_Sw = (1.0 - self.shrinkage) * Sw
        shrunk_Sw.flat[:: n_features + 1] += self.shrinkage * mu

        Sb = St - shrunk_Sw  # between scatter

        evals, evecs = linalg.eigh(Sb, shrunk_Sw)
        evecs = evecs[:, np.argsort(evals)[::-1]]  # sort eigenvectors

        self.scalings_ = evecs
        self.coef_ = np.dot(self.means_, evecs).dot(evecs.T)
        self.intercept_ = -0.5 * np.diag(np.dot(self.means_, self.coef_.T)) + np.log(
            self.priors_
        )

    def fit(self, X, y):
        '''
        X: input features, N x D
        y: labels, N

        '''
        self.classes_ = np.unique(y)
        #n_samples, _ = X.shape
        n_classes = len(self.classes_)

        max_components = min(len(self.classes_) - 1, X.shape[1])

        if self.n_components is None:
            self._max_components = max_components
        else:
            if self.n_components > max_components:
                raise ValueError(
                    "n_components cannot be larger than min(n_features, n_classes - 1)."
                )
            self._max_components = self.n_components

        _, y_t = np.unique(y, return_inverse=True)  # non-negative ints
        self.priors_ = np.bincount(y_t) / float(len(y))
        self._solve_eigen(X, y, shrinkage=self.shrinkage,)

        return self
    
    def transform(self, X):
        # project X onto Fisher Space
        X_new = np.dot(X, self.scalings_)
        return X_new[:, : self._max_components]

    def predict_proba(self, X):
        scores = np.dot(X, self.coef_.T) + self.intercept_   ##计算分数
        return softmax(scores)


def SFDA_Score(X, y):

    n = len(y)
    num_classes = len(np.unique(y))
    SFDA_first = SFDA()
    prob = SFDA_first.fit(X, y).predict_proba(X)  # p(y|x)
    
    # soften the probability using softmax for meaningful confidential mixture
    prob = np.exp(prob) / np.exp(prob).sum(axis=1, keepdims=True) 
    means, means_ = _class_means(X, y)  # class means, outer classes means
    
    # ConfMix
    for y_ in range(num_classes):
        indices = np.where(y == y_)[0]
        y_prob = np.take(prob, indices, axis=0)
        y_prob = y_prob[:, y_]  # probability of correctly classifying x with label y        
        X[indices] = y_prob.reshape(len(y_prob), 1) * X[indices] + \
                            (1 - y_prob.reshape(len(y_prob), 1)) * means_[y_]
    
    SFDA_second = SFDA(shrinkage=SFDA_first.shrinkage)
    prob = SFDA_second.fit(X, y).predict_proba(X)   # n * num_cls

    # leep = E[p(y|x)]. Note: the log function is ignored in case of instability.
    sfda_score = np.sum(prob[np.arange(n), y]) / n
    return sfda_score


### TransRate ###

def discretize_vector(vec, num_buckets=47):
    num_bins = num_buckets
    bin_size = len(vec) // num_bins
    sorted_vec = vec

    result = [0] * len(vec)
    
    for i in range(num_bins):
        start_idx = i * bin_size
        end_idx = (i + 1) * bin_size
        
        if i < num_bins - 1:
            if len(vec) - end_idx < bin_size:
                end_idx = len(vec)
        else:
            end_index = len(vec)

        for j in range(start_idx, end_idx):
            result[j] = i
    
    return np.array(result)


def coding_rate(Z, eps=1e-4): 
    n, d = Z.shape
    (_, rate) = np.linalg.slogdet((np.eye(d) + 1 / (n*eps)*Z.transpose()@Z))
    return 0.5*rate


def Transrate(Z, y, eps=1e-4): 
    Z = Z - np.mean(Z, axis=0, keepdims=True)
    RZ = coding_rate(Z, eps) 
    RZY = 0.
    K = int(y.max() + 1) 
    for i in range(K):
        tmp_Z = Z[(y == i).flatten()]
        RZY += coding_rate(tmp_Z, eps) 
    return (RZ - RZY / K)


### EMMS ###

def emms_clip_label_embeddings(
        categories,
        load_path: Union[str, os.PathLike]='laion/CLIP-ViT-g-14-laion2B-s12B-b42K',
        device='cuda',
):
    model = CLIPModel.from_pretrained(load_path).to(device=device)
    processor = CLIPProcessor.from_pretrained(load_path)
    
    inputs = processor(categories, padding=True, return_tensors="pt").to(device=device)
    with torch.no_grad():
        label_embeddings = model.get_text_features(**inputs)
    
    return label_embeddings.cpu().numpy()


def emms_gpt2_label_embeddings(
        categories,
        load_path: Union[str, os.PathLike]='gpt2-medium',
        device='cuda',
):
    model = GPT2Model.from_pretrained(load_path).to(device=device)
    tokenizer = GPT2Tokenizer.from_pretrained(load_path)
    tokenizer.pad_token = tokenizer.eos_token

    encoded_input = tokenizer(categories, return_tensors='pt', padding=True, truncation=True).to(device=device)
    with torch.no_grad():
        outputs = model(**encoded_input)
    last_hidden_states = outputs.last_hidden_state
    label_embeddings = last_hidden_states.mean(dim=1)
    
    return label_embeddings.cpu().numpy()


def emms_bert_label_embeddings(
        categories,
        load_path: Union[str, os.PathLike]='bert-large-uncased',
        device='cuda',
):
    model = BertModel.from_pretrained(load_path).to(device=device)
    tokenizer = AutoTokenizer.from_pretrained(load_path)

    encoded_input = tokenizer(categories, return_tensors='pt', padding=True).to(device=device)
    with torch.no_grad():
        outputs = model(**encoded_input)
    last_hidden_states = outputs.last_hidden_state
    label_embeddings = last_hidden_states.mean(dim=1)

    return label_embeddings.cpu().numpy()


def sparsemax(z):
    """forward pass for sparsemax
    this will process a 2d-array $z$, where axis 1 (each row) is assumed to be
    the the z-vector.
    """

    # sort z
    z_sorted = np.sort(z, axis=1)[:, ::-1]

    # calculate k(z)
    z_cumsum = np.cumsum(z_sorted, axis=1)
    k = np.arange(1, z.shape[1] + 1)
    z_check = 1 + k * z_sorted > z_cumsum
    # use argmax to get the index by row as .nonzero() doesn't
    # take an axis argument. np.argmax return the first index, but the last
    # index is required here, use np.flip to get the last index and
    # `z.shape[axis]` to compensate for np.flip afterwards.
    k_z = z.shape[1] - np.argmax(z_check[:, ::-1], axis=1)

    # calculate tau(z)
    tau_sum = z_cumsum[np.arange(0, z.shape[0]), k_z - 1]
    tau_z = ((tau_sum - 1) / k_z).reshape(-1, 1)

    return np.maximum(0, z - tau_z)


def EMMS(Z,Y):
    x = Z
    y = Y
    N, D2, K = y.shape
    for i in range(K):
        y_mean = np.mean(y[:,:,i] , axis=0)
        y_std = np.std(y[:,:,i], axis=0)
        epsilon = 1e-8
        y[:,:,i] = (y[:,:,i] - y_mean) / (y_std + epsilon)
    N, D1 = x.shape
    x_mean = np.mean(x , axis=0)
    x_std = np.std(x, axis=0)
    epsilon = 1e-8
    x = (x - x_mean) / (x_std + epsilon)
    lam = np.array([1/K] * K)
    w1 = 0
    lam1 = 0
    T = 0
    b = np.dot(y, lam)
    T = b
    for k in range(1):
        a = x
        w = np.linalg.lstsq(a, b, rcond=None)[0]
        w1 = w
        a = y.reshape(N*D2, K)
        b = np.dot(x, w).reshape(N*D2)
        lam = np.linalg.lstsq(a, b, rcond=None)[0]
        lam = lam.reshape(1,K)
        lam = sparsemax(lam)
        lam = lam.reshape(K,1)
        lam1 = lam
        b = np.dot(y, lam)
        b = b.reshape(N,D2)
        T = b
    y_pred = np.dot(x,w1)
    res = np.sum((y_pred - T)**2) / N*D2

    return -res


# Code borrowed from https://github.com/BUserName/NCTI

### NCTI ###

def NCTI_Score(X, y):
    C = np.unique(y).shape[0]
    pca = PCA(n_components=64)
    X = pca.fit_transform(X, y)
    temp = max(np.exp(-pca.explained_variance_[:32].sum()), 1e-10)

    if temp == 1e-10:
        clf = LinearDiscriminantAnalysis(solver='svd')
    else:
        clf = LinearDiscriminantAnalysis(solver='eigen', shrinkage=float(temp))
    
    low_feat = clf.fit_transform(X, y)
    low_feat = low_feat - np.mean(low_feat, axis=0, keepdims=True)
    seli_score = np.linalg.norm(low_feat, ord='nuc')

    low_pred = clf.predict_proba(X)
    ncc_score = np.sum(low_pred[np.arange(X.shape[0]), y]) / X.shape[0]

    class_pred_nuc = 0
    for c in range(C):
        c_pred = low_pred[(y==c).flatten()]
        c_pred_nuc = np.linalg.norm(c_pred, ord='nuc')
        class_pred_nuc += c_pred_nuc
    vc_score = np.log(class_pred_nuc)
    
    return seli_score, ncc_score, vc_score

# code from https://github.com/YaojieBao/An-Information-theoretic-Metric-of-Transferability
def getCov(X):
    X_mean = X - np.mean(X, axis=0, keepdims=True)
    cov = np.divide(np.dot(X_mean.T, X_mean), len(X)-1)
    return cov


def H_Score(X, Y):
    '''
    X: feature representations
    Y: vector of corresponding feature labels
    '''
    CovX = getCov(X)
    alphabetY = list(set(Y))
    g = np.zeros_like(X)
    for y in alphabetY:
        Ef_y = np.mean(X[Y==y, :], axis=0)
        g[Y==y] = Ef_y
    
    CovG = getCov(g)
    score = np.trace(np.dot(np.linalg.pinv(CovX, rcond=1e-15), CovG))
    return score
