from __future__ import division

import numpy as np
from scipy.spatial.distance import cdist

from adapt.base import BaseAdaptEstimator
from adapt.feature_based._sca import convert_to_cell_data


def compute_width(dist_mat):
    n1, n2 = dist_mat.shape
    if n1 == n2:
        if np.allclose(dist_mat, dist_mat.T):
            id_tril = np.tril_indices(dist_mat.shape[0], -1)
            bandwidth = dist_mat[id_tril]
            return np.sqrt(0.5 * np.median(bandwidth))
        else:
            return np.sqrt(0.5 * np.median(dist_mat))
    else:
        return np.sqrt(0.5 * np.median(dist_mat))


def _calc_k(Xs, X):
    dist_s = cdist(Xs, X)
    dist_s = dist_s**2
    sgm = compute_width(dist_s)
    return np.exp(-dist_s / (2 * sgm ** 2))


class MDA(BaseAdaptEstimator):
    """
    Python implementation of Multidomain discriminant analysis (MDA)
    (tested on Anaconda 5.3.0 64-bit for python 2.7.15 on Windows 10)

    Shoubo Hu (shoubo.sub AT gmail.com)
    2019-08-13
    """

    def __init__(self,
                 estimator=None,
                 Xt=None,
                 copy=True,
                 verbose=1,
                 random_state=None,
                 beta=None,
                 gamm=None,
                 alph=None,
                 eig_ratio=None,
                 mda_terms=None,
                 n_class=None,
                 n_total=None,
                 n_domain=None,
                 test=False,
                 **params):

        names = self._get_param_names()
        kwargs = {k: v for k, v in locals().items() if k in names}
        kwargs.update(params)
        super().__init__(**kwargs)


    def fit_transform(self, Xs, ys, Xt, yt, domains):
        self.mda_terms = {'X_s': Xs}
        temp, self.y_s_list  = convert_to_cell_data(Xs, ys, domains)
        self._get_helpful_numbers(ys, domains)
        K_s = _calc_k(Xs, Xs)
        if self.test:
            self.test_hyperparams(K_s)
        else:
            # Obtain the transformation and its corresponding eigenvalues
            self._get_trans_matrix_and_eigenvalues(K_s)

        return self.transform(Xs)

    def transform(self, X, domains=None):
        K_bar = self._calc_K_bar(X)

        # Project the instances
        Z = np.dot(np.dot(K_bar.T, self.mda_terms['B1']), self.mda_terms['A1'])

        return Z

    def _get_helpful_numbers(self, Y, domains):
        self.n_domain = max(domains)+1
        self.n_total = len(Y)
        self.n_class = len(np.unique(Y))
        self.H_s = np.eye(self.n_total, dtype=float) - np.ones((self.n_total, self.n_total), dtype=float) / self.n_total


    def _get_trans_matrix_and_eigenvalues(self, K_s):
        F, P, G, Q, K_bar = self._quantities(K_s)
        B, A = self._transformation(F, P, G, Q, K_bar)
        B1, A1 = self._trim(B, A)
        self.mda_terms.update({'B1': B1, 'A1': A1})

    def _trim(self, B, A):
        vals = np.diag(A)
        vals_sqrt = np.zeros((A.shape[0],), dtype=float)
        ratio = []
        count = 0
        for i in range(0, len(vals)):
            if vals[i] < 0:
                break

            count = count + vals[i]
            ratio.append(count)
            vals_sqrt[i] = 1 / np.sqrt(vals[i])

        A_sqrt = np.diag(vals_sqrt)
        ratio = [i / count for i in ratio]

        if self.eig_ratio <= 1:
            element = [x for x in ratio if x > self.eig_ratio][0]
            idx = ratio.index(element)
            n_eigs = idx
        else:
            n_eigs = self.eig_ratio

        B1 = B[:, 0:n_eigs]
        A1 = A_sqrt[0:n_eigs, 0:n_eigs]

        return B1, A1

    def _quantities(self, K):
        """
        Compute quantities in Multidomain Discriminant Analysis (MDA)

        INPUT
            K 		 kernel matrix of data of all source domains

        OUTPUT
            F 		 F in avearge class discrepancy, Eq.(11)
            P 		 P in multi-domain between class scatter, Eq.(15)
            G 		 G in average domain discrepancy, Eq.(6)
            Q 		 Q in multi-domain within class scatter, Eq.(18)
            K_bar 	 the kernel matrix (K)
        """

        # save class and domain index of all obs into two row vectors
        class_index = np.zeros((self.n_total,), dtype=float)
        domain_index = np.zeros((self.n_total,), dtype=float)
        count = 0
        for s in range(0, self.n_domain):
            for i in range(0, self.y_s_list[s].shape[0]):
                temp_c = self.y_s_list[s][i]
                class_index[count] = temp_c
                domain_index[count] = s
                count = count + 1

        # prepare count and proportion matrix
        # cnt_mat_{sj} is the number of pts in domain s class j
        cnt_mat = np.zeros((self.n_domain, self.n_class), dtype=float)
        domain_index_list = []
        class_index_list = []
        for s in range(0, self.n_domain):
            idx_s = np.where(domain_index == s)[0]
            domain_index_list.append(idx_s)

            for j in range(0, self.n_class):
                if s == 0:
                    idx_j = np.where(class_index == j)[0]
                    class_index_list.append(idx_j)
                else:
                    idx_j = class_index_list[j]

                idx = np.intersect1d(idx_s, idx_j, assume_unique=True)
                cnt_mat[s, j] = len(idx)

        # [prpt_vec]_{sj} is n^{s}_{j} / n^{s}
        prpt_vec = cnt_mat * np.reciprocal(np.tile(np.sum(cnt_mat, axis=1).reshape(-1, 1), (1, self.n_class)))
        sum_over_dm_vec = np.sum(prpt_vec, axis=0)
        nj_vec = np.sum(cnt_mat, axis=0)

        class_domain_mean = [None for _ in range(self.n_domain)]
        for s in range(0, self.n_domain):
            idx_s = domain_index_list[s]
            domain_mean = np.zeros((self.n_total, self.n_class), dtype=float)

            for j in range(0, self.n_class):
                idx_j = class_index_list[j]
                idx = np.intersect1d(idx_s, idx_j, assume_unique=True)
                domain_mean[:, j] = np.mean(K[:, idx], axis=1)

            class_domain_mean[s] = domain_mean

        u_j_mat = np.zeros((self.n_total, self.n_class), dtype=float)
        for j in range(0, self.n_class):
            u_j = np.zeros((self.n_total, 1), dtype=float)
            for s in range(0, self.n_domain):
                u_j = u_j + class_domain_mean[s][:, j].reshape(-1, 1) * prpt_vec[s, j]
            u_j_mat[:, j] = u_j[:, 0] / sum_over_dm_vec[j]

        # compute matrix P
        u_bar = np.zeros((self.n_total,), dtype=float)
        for j in range(0, self.n_class):
            u_bar = u_bar + u_j_mat[:, j] * (sum_over_dm_vec[j] / self.n_domain)

        pre_P = np.zeros((self.n_total, self.n_total), dtype=float)
        for j in range(0, self.n_class):
            diff = (u_j_mat[:, j] - u_bar).reshape(-1, 1)
            pre_P = pre_P + nj_vec[j] * np.dot(diff, diff.T)

        P = pre_P / self.n_total

        # compute matrix F
        F = np.zeros((self.n_total, self.n_total), dtype=float)
        for j1 in range(0, self.n_class - 1):
            for j2 in range(j1 + 1, self.n_class):
                temp = u_j_mat[:, j1].reshape(-1, 1) - u_j_mat[:, j2].reshape(-1, 1)
                F = F + np.dot(temp, temp.T)

        F = F / (self.n_class * (self.n_class - 1) * 0.5)

        # compute matrix Q
        Q = np.zeros((self.n_total, self.n_total), dtype=float)
        for j in range(0, self.n_class):
            idx_j = class_index_list[j]
            G_j = u_j_mat[:, j].reshape(-1, 1)

            G_ij = K[:, idx_j]
            Q_i = G_ij - np.tile(G_j, (1, len(idx_j)))

            Q = Q + np.dot(Q_i, Q_i.T)

        Q = Q / self.n_total

        # compute matrix G
        G = np.zeros((self.n_total, self.n_total), dtype=float)
        for j in range(0, self.n_class):
            for s1 in range(0, self.n_domain - 1):
                idx = np.intersect1d(domain_index_list[s1], class_index_list[j], assume_unique=True)
                left = np.mean(K[:, idx], axis=1).reshape(-1, 1)

                for s2 in range(s1 + 1, self.n_domain):
                    idx = np.intersect1d(domain_index_list[s2], class_index_list[j], assume_unique=True)
                    right = np.mean(K[:, idx], axis=1).reshape(-1, 1)
                    temp = left - right
                    G = G + np.dot(temp, temp.T)

        G = G / (self.n_class * self.n_domain * (self.n_domain - 1) * 0.5)

        J = np.ones((self.n_total, self.n_total), dtype=float) / self.n_total
        K_bar = K - np.dot(J, K) - np.dot(K, J) + np.dot(np.dot(J, K), J)
        return F, P, G, Q, K_bar

    def _transformation(self, F, P, G, Q, K_bar, eps=1e-5):
        """
        compute the transformation in Multidomain Discriminant Analysis (MDA)

        INPUT
            beta, alph, gamm    - trade-off parameters in Eq.(20)
            eps                 - coefficient of the identity matrix (footnote in page 5)

        OUTPUT
            B                   - matrix of projection
            A                   - corresponding eigenvalues
        """

        I_0 = np.eye(self.n_total)
        F1 = self.beta * F + (1 - self.beta) * P
        F2 = (self.gamm * G + self.alph * Q + K_bar + eps * I_0)
        F2_inv_F1 = np.linalg.solve(F2, F1)

        [A, B] = np.linalg.eig(F2_inv_F1)
        B, A = np.real(B), np.real(A)
        idx = np.argsort(A)
        idx = np.flip(idx, axis=0)
        A = A[idx]
        B = B[:, idx]
        A = np.diag(A)

        return B, A


    def _calc_K_bar(self, X):
        K_s_v = _calc_k(self.mda_terms['X_s'], X)
        n_v = X.shape[0]
        H_v = np.eye(n_v, dtype=float) - np.ones((n_v, n_v), dtype=float) / n_v
        K_s_v_bar = np.dot(np.dot(self.H_s, K_s_v), H_v)
        return K_s_v_bar

    def test_hyperparams(self, K_s):
        all_beta = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        all_alph= [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6]
        all_gamm = [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6]

        F, P, G, Q, K_bar = self._quantities(K_s)

        for beta in all_beta:
            for alph in all_alph:
                for gamm in all_gamm:
                    self.beta = beta
                    self.alph = alph
                    self.gamm = gamm

                    B, A = self._transformation(F, P, G, Q, K_bar)
                    A_ratio = A[0,0]/A[1,1]

