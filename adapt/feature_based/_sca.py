import numpy as np
import scipy
from scipy.spatial.distance import cdist
from scipy.sparse import identity
import tensorflow as tf

from adapt.base import BaseAdaptEstimator, make_insert_doc
from adapt.utils import set_random_seed


def convert_to_cell_data(X, y=None, domains=None) -> (list, list):
    """
    separate X and Y to a list of dataframes by column.
    Args:
        X: ndarray N*M
        Y: ndarray N*1

    Returns:
        chunks: list of dataframes
    """
    if not domains:
        return [X], [y]

    unique_domains = np.unique(domains)
    X_chunks = [X[domains == domain] for domain in unique_domains]
    if y is not None:
        y_chunks = [y[domains == domain] for domain in unique_domains]
    else:
        y_chunks = None

    return X_chunks, y_chunks


class SCA(BaseAdaptEstimator):
    """
    Scatter Component Analysis (SCA). this is a fast representation learning algorithm that can be applied to both
    domain adaptation and domain generalization. SCA is based on a simple geometrical measure, i.e., scatter, which
    operates on reproducing kernel Hilbert space.

    Parameters
    ----------
    BETA: float
        Trade-off parameter.
    DELTA: float
        Trade-off parameter.
    EIG_RATIO: int.
        number of output features

    Attributes
    ----------
    estimator_ : object
        Fitted estimator.

    """

    def __init__(self,
                 estimator=None,
                 Xt=None,
                 BETA=None,
                 DELTA=None,
                 EIG_RATIO=None,
                 copy=True,
                 verbose=1,
                 random_state=None,
                 **params):
        self.sca_terms = None
        names = self._get_param_names()
        kwargs = {k: v for k, v in locals().items() if k in names}
        kwargs.update(params)
        super().__init__(**kwargs)

    def fit_transform(self, Xs, ys, Xt, yt, domains=None, **kwargs):
        """
        Fit embeddings.

        Parameters
        ----------
        X : array
            Input source data.

        Y : array

        domains: list of domains

        kwargs : key, value argument

        Returns
        -------
        Zs : embedded source data
        """
        X_cell, Y_cell = convert_to_cell_data(Xs, ys, domains)
        self._get_helpful_numbers(X_cell, Y_cell)
        K_s = self._calc_k(X_cell, X_cell)

        # Obtain the transformation and its corresponding eigenvalues
        B1, A1 = self._get_trans_matrix_and_eigenvalues(K_s, X_cell, Y_cell)

        self.sca_terms = {'X_s_cell': X_cell, 'B1': B1, 'A1': A1}

        return self.transform(Xs, domains)

    def transform(self, X, domains=None):
        """
        Apply the transformation on the target domain instances.

        Args:
            data: dict of features to transform
            data_type: str: 'test', 'train' or 'val'
        :return: data with a new column Z - projected domain instances

        """
        X_cell, temp = convert_to_cell_data(X, domains=domains)

        K = self._calc_k(X_cell, self.sca_terms['X_s_cell'])

        K_bar = self._calc_K_bar(self.sca_terms['X_s_cell'], X_cell, K)

        # Project the instances
        Z = np.dot(K_bar.T, self.sca_terms['B1'])
        # Normalize the projected instances
        Z = Z / np.sqrt(self.sca_terms['A1'].diagonal())

        return Z
    @staticmethod
    def _compute_width(dist_mat):
        # Use np.nan to ignore the lower triangle and the diagonal
        triu_only = np.triu(dist_mat, k=1)
        triu_only[triu_only == 0] = np.nan

        # Calculate the median of the non-zero elements directly
        md = np.nanmedian(triu_only)

        return np.sqrt(0.5 * md)

    def _calc_X_c(self, X, Y):
        """Calculate X_c, num_labeled, class_index, and domain_index from given X and Y.
                Args:
                    X (list): List of arrays representing data from different domains.
                    Y (list): List of arrays representing class labels for the data.
                Returns:
                    X_c: List of arrays, each containing data for one class.
                    num_labeled: Number of labeled samples.
                    class_index: Array indicating the class of each sample.
                    domain_index: Array indicating the domain of each sample.
                """
        # Find the samples of class k and put them into cell k
        X_c = [[] for _ in range(self.num_class)]
        class_index = np.zeros(self.n_total, dtype=int)
        domain_index = np.zeros(self.n_total, dtype=int)
        count = 0
        num_labeled = 0
        for i, Yi in enumerate(Y):
            for j, temp_c in enumerate(Yi):
                class_index[count] = temp_c
                domain_index[count] = i
                count += 1
                if temp_c != 0:
                    X_c[int(temp_c - 1)].append(X[i][j])
                    num_labeled += 1

        # Convert list of lists to list of arrays
        X_c = [np.array(Xi) for Xi in X_c]
        return X_c, num_labeled, class_index, domain_index

    def _calc_P(self, X_c, class_index, K):
        # creates a list of class index for each column in X_c
        class_idx = np.repeat(np.arange(len(X_c)), [len(X_ci) for X_ci in X_c])
        # mean of each class
        P_mean = np.mean(K[:, class_index != 0], axis=1)
        unique_classes = np.unique(class_index)
        #
        temp_k = np.array([np.mean(K[:, class_index == u_class], axis=1).T for u_class in unique_classes]).T
        # Broadcasting subtraction
        P = temp_k[:, class_idx] - P_mean[:, None]
        P = P @ P.T

        return P

    def _calc_Q(self, class_index, K):
        n_total = K.shape[0]
        Q = np.zeros((n_total, n_total))  # Start with an all-zero matrix with appropriate dimensions

        for i in range(self.num_class):
            # Boolean mask for selecting the indices within the class
            mask = (class_index == i + 1)
            G_ij = K[:, mask]
            # Compute the corrected G_ij by subtracting the mean
            Q_i = G_ij - G_ij.mean(axis=1, keepdims=True)
            # dot product update function
            Q += Q_i @ Q_i.T

        return Q

    def _calc_D(self, domain_index, K):
        # Create domain indicator matrix
        domain_indicator = (np.arange(self.n_domain) == domain_index[:, None]).astype(int)

        # Number of elements in each domain
        n_per_domain = np.count_nonzero(domain_indicator, axis=0)

        # Calculate the domain means
        domain_means = K @ domain_indicator / n_per_domain

        # Calculate the overall mean
        overall_mean = np.mean(domain_means, axis=1, keepdims=True)

        # Calculate the difference matrix (each domain mean - overall mean)
        D = domain_means - overall_mean

        # Calculate the scatter matrix D@D^T / n_domain
        D = D @ D.T / self.n_domain

        return D

    def _center_K(self, K):
        n_total = K.shape[0]

        # Calculate the row mean and column mean vectors
        row_mean = np.nanmean(K, axis=1, keepdims=True)
        col_mean = np.nanmean(K, axis=0, keepdims=True)

        # Calculate the total mean
        total_mean = np.nanmean(K)
        # Center the matrix K
        K_centered = K - row_mean - col_mean + total_mean

        return K_centered

    def _SCA_trans(self, P, T, D, Q, K_bar, beta, delta, epsilon):
        # Solve the generalized eigenvalue problem for the matrix F
        F1 = beta * P + (1 - beta) * T
        F2 = delta * D + Q + K_bar + epsilon * identity(F1.shape[0], format='csr')  # epsilon * np.eye(K_bar.shape[0])

        A1, B1 = scipy.sparse.linalg.eigs(F1, M=F2, which='LR',
                                          k=self.EIG_RATIO)  # this line can replace lines 126-130.
        # It returns B with vectors which are not normalized. use B = B / np.linalg.norm(B, axis=0) to normalize
        B1 = B1 / np.linalg.norm(B1, axis=0)

        # Ensure the eigenvalues and eigenvectors are real
        A1 = np.real(A1)
        B1 = np.real(B1)

        # Sort the eigenvalues in descending order and reorder the eigenvectors accordingly
        idx = A1.argsort()[::-1]
        A1 = np.diagflat(A1[idx])
        B1 = B1[:, idx]

        return B1, A1

    def _get_trans_matrix_and_eigenvalues(self, K, X, Y, epsilon=1e-5):
        X_c, num_labeled, class_index, domain_index = self._calc_X_c(X, Y)
        # Compute matrix P
        P = self._calc_P(X_c, class_index, K)
        # Compute matrix Q
        Q = self._calc_Q(class_index, K)
        # Compute matrix D (Domain Scatter)
        D = self._calc_D(domain_index, K)
        # Center the kernel matrix K
        K_bar = self._center_K(K)
        # Compute matrix T (Total Scatter)
        T = K_bar @ K_bar / self.n_total
        # Obtain the transformation and its corresponding eigenvalues
        B1, A1 = self._SCA_trans(P, T, D, Q, K_bar, self.BETA, self.DELTA, epsilon)
        # Select the top eig_ratio eigenvalues and corresponding eigenvectors
        # A1 = A[:self._p.SCA['EIG_RATIO'], :self._p.SCA['EIG_RATIO']]
        # B1 = B[:, :self._p.SCA['EIG_RATIO']]

        return B1, A1

    @tf.function
    def get_numpy_from_tensor(tensors):
        return [tensor for tensor in tensors]

    def _calc_k(self, X_cell, X_s_cell):
        X_s = np.vstack(X_s_cell)
        X = np.vstack(X_cell)
        # Check if inputs contain NaNs or infinite values
        if np.any(np.isnan(X)) or np.any(np.isnan(X_s)):
            raise ValueError('Input contains NaNs.')
        if np.any(np.isinf(X)) or np.any(np.isinf(X_s)):
            raise ValueError('Input contains infinite values.')

        dist_s = cdist(X_s, X, 'sqeuclidean')
        sgm = self._compute_width(dist_s)
        return np.exp(-dist_s / (2 * sgm ** 2))

    def _calc_K_bar(self, X_s_cell, X_v_cell, K_s_v):
        # We do not need to create X_s and X_v because we only need their sizes
        n_s = sum(cell.shape[0] for cell in X_s_cell)
        n_v = sum(cell.shape[0] for cell in X_v_cell)
        # Row means and column means
        row_means = K_s_v.mean(axis=1).reshape(n_s, 1)
        col_means = K_s_v.mean(axis=0).reshape(1, n_v)
        # Grand mean
        grand_mean = K_s_v.mean()
        # Applying the centering
        new_K_s_v = K_s_v - row_means - col_means + grand_mean

        return new_K_s_v

    def _get_helpful_numbers(self, X, Y):
        self.n_domain = len(X)
        self.n_total = sum(len(Xi) for Xi in X)
        Y_all = np.concatenate(Y)
        # Check the label of the first class
        # if np.any(Y_all == 0):
        #     self.num_class = len(np.unique(Y_all)) - 1
        # else:
        self.num_class = len(np.unique(Y_all))





