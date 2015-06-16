from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomTreesEmbedding
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import KernelPCA, PCA
from sklearn.gaussian_process import GaussianProcess

class Regressor(BaseEstimator):
    def __init__(self):
        rte = RandomTreesEmbedding(n_estimators=100, n_jobs=-1)
        kpca = KernelPCA(n_components=100)
        rf = RandomForestRegressor(n_estimators=100)

        self.reg = make_pipeline(rte, kpca, rf)
 

    def fit(self, X, y):
        self.reg.fit(X, y)
 
    def predict(self, X):
        return self.reg.predict(X)