from sklearn.decomposition import PCA
from sklearn.base import check_array

from adapt.base import BaseAdaptEstimator, make_insert_doc
from adapt.utils import set_random_seed


@make_insert_doc()
class SA(BaseAdaptEstimator):
    """
    SA : Subspace Alignment
    
    Parameters
    ----------
    n_components : int (default=None)
        Number of components of the PCA
        transformation. If ``None`` the
        number of components is equal
        to the input dimension of ``X``
    
    Attributes
    ----------    
    estimator_ : object
        Fitted estimator.
        
    pca_src_ : sklearn PCA
        Source PCA
    
    pca_tgt_ : sklearn PCA
        Target PCA
        
    M_ : numpy array
        Alignment matrix
    """
    
    def __init__(self,
                 estimator=None,
                 Xt=None,
                 yt=None,
                 n_components=None,
                 copy=True,
                 verbose=1,
                 random_state=None,
                 **params):
        
        names = self._get_param_names()
        kwargs = {k: v for k, v in locals().items() if k in names}
        kwargs.update(params)
        super().__init__(**kwargs)
        
    
    def fit_transform(self, Xs, Xt, **fit_params):
        Xs = check_array(Xs)
        Xt = check_array(Xt)
        set_random_seed(self.random_state)
        
        self.pca_src_ = PCA(self.n_components)
        self.pca_tgt_ = PCA(self.n_components)
        
        self.pca_src_.fit(Xs)
        self.pca_tgt_.fit(Xt)
        
        self.M_  = self.pca_src_.components_.dot(
            self.pca_tgt_.components_.transpose())
        
        return self.pca_src_.transform(Xs).dot(self.M_)


    def transform(self, X, domain="tgt"):
        X = check_array(X)
        
        if domain in ["tgt", "target"]:
            return self.pca_tgt_.transform(X)
        elif domain in ["src", "source"]:
            return self.pca_src_.transform(X).dot(self.M_)
        else:
            raise ValueError("`domain `argument "
                             "should be `tgt` or `src`, "
                             "got, %s"%domain)