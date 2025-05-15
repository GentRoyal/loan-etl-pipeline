import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_selection import chi2
import pandas as pd

class Statistics:
    def __init__(self):
        pass

    def pca_ranking(self, X, y):
        pca = PCA(n_components = min(X.shape[1], X.shape[0]))
        pca.fit(X)
        
        feature_importance = np.sum(np.abs(pca.components_), axis=0)
        feature_ranks = dict(zip(X.columns, feature_importance))
        
        sorted_features = sorted(feature_ranks.items(), key=lambda x: x[1], reverse=True)
        
        return [f[0] for f in sorted_features]  # Return feature names only
    
    def dispersion_ratio(self, X, y):
        variances = X.var()
    
        class_means = {}
        for cls in y.unique():
            class_means[cls] = X[y == cls].mean()
        
        overall_mean = X.mean()
        between_variance = pd.Series(0.0, index=X.columns)
        
        for cls, means in class_means.items():
            class_size = sum(y == cls)
            between_variance += class_size * ((means - overall_mean) ** 2)
        
        between_variance /= len(y)
        
        dispersion_ratios = between_variance / variances
        sorted_ratios = dispersion_ratios.sort_values(ascending=False)
        
        return sorted_ratios.index.tolist()
    
    def chi_square_ranking(self, X, y):
        chi_values, p_values = chi2(X, y)
    
        feature_scores = dict(zip(X.columns, chi_values))
        
        sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
        
        return [f[0] for f in sorted_features]
    
    def pearson_correlation(self, X, y):
        y_numeric = pd.factorize(y)[0]
        
        correlations = {}
        for col in X.columns:
            corr = abs(np.corrcoef(X[col], y_numeric)[0, 1])
            correlations[col] = corr
        
        sorted_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
        
        return [f[0] for f in sorted_features]
    
    def mean_absolute_difference(self, X, y):
        classes = y.unique()
        
        mad_scores = {}
        
        for col in X.columns:
            overall_mean = X[col].mean()
    
            mad = 0
            for cls in classes:
                class_mean = X.loc[y == cls, col].mean()
                mad += abs(class_mean - overall_mean) * sum(y == cls) / len(y)
            
            mad_scores[col] = mad
    
        sorted_features = sorted(mad_scores.items(), key=lambda x: x[1], reverse=True)
        
        return [f[0] for f in sorted_features]
    
    def low_variance(self, X, y):
        variances = X.var()
        sorted_vars = variances.sort_values()
        
        return sorted_vars.index.tolist()