from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomTreesEmbedding
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import KernelPCA, PCA
from sklearn.gaussian_process import GaussianProcess

class Regressor(BaseEstimator):
    def __init__(self):
        kpca = PCA(n_components=500)
        gbr = BaggingRegressor(base_estimator=GradientBoostingRegressor(n_estimators=200), n_jobs=-1, n_estimators=50)
        self.reg = make_pipeline(kpca, gbr)

    def fit(self, X, y):
        self.reg.fit(X, y)
 
    def predict(self, X):
        return self.reg.predict(X)