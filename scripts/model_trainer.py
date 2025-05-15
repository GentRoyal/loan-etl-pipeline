from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from collections import Counter
from statistical_measures import Statistics
import numpy as np
import time

class ModelPipeline:
    def __init__(self, model = None, X = None, y = None):
        self.model = model if model else MLPClassifier(activation = 'logistic', solver = 'lbfgs', batch_size = 'auto', learning_rate = 'adaptive', \
                                                        learning_rate_init = 0.03, max_iter = 5000, \
                                                        momentum = 0.2, \
                                                        random_state = np.random.get_state()[1][0], \
                                                        early_stopping=False)
        self.X = X
        self.y = y

    def vertical_split(self, df, target):
        self.X = df.drop(columns = [target])
        self.X = (self.X - self.X.min())/(self.X.max() - self.X.min())
        #self.X = pd.DataFrame(X_normalized, columns = self.X.columns)
        self.y = df[target]

    def horizontal_split(self, X, y, test_size = 0.2, random_state = 42, stratify = True):
        X_normalized = (X - X.min())/(X.max() - X.min())
        
        return train_test_split(
            X, y,
            test_size = test_size,
            random_state = random_state,
            stratify = y if stratify else None
        )

    def train_with_kfold(self):
        sta = Statistics()
        statistical_methods = {
            'PCA': sta.pca_ranking,
            'Dispersion_Ratio': sta.dispersion_ratio,
            'Chi_Square': sta.chi_square_ranking,
            'Pearson_Correlation': sta.pearson_correlation,
            'Mean_Absolute_Difference' : sta.mean_absolute_difference, 
            'Low_Variance' : sta.low_variance
            }
        
        # Apply each statistical method and get top quartile features
        top_features_all_methods = []
        method_specific_results = {}
        
        for method_name, method in statistical_methods.items():
            print(f"Applying {method_name} feature selection...")
            
            ranked_features = method(self.X, self.y)
            
            top_quartile_count = max(1, len(ranked_features) // 4) # Get top quartile (25%) of features
            top_features = ranked_features[:top_quartile_count]
            
            top_features_all_methods.append(set(top_features))
        
        # Calculate selected feature sets for different n values
        feature_counts = Counter(f for s in top_features_all_methods for f in s)
        
        # For each n (1 to number of methods), get features appearing in at least n methods
        selected_sets = {}
        for n in range(1, len(statistical_methods) + 1):
            selected_sets[n] = {f for f, count in feature_counts.items() if count >= n}
        
        # Steps 13-19: Perform Cross Validation
        results = {}
        
        for n in range(1, len(statistical_methods) + 1):
            selected_features = list(selected_sets[n])
            
            if not selected_features:
                results[n] = {
                    'num_features': 0,
                    'features': [],
                    'accuracy': None
                }
                print(f"No features selected for threshold n={n}")
                continue
            
            print(f"Evaluating features selected by at least {n} methods...")
            
            X_selected = self.X[selected_features]
            accuracies = []
            epochs_used = []
    
            cv_generator = StratifiedKFold(n_splits = 10, shuffle = False)
    
            t1 = time.time()
            for train_idx, test_idx in cv_generator.split(X_selected, self.y):
                X_train, X_test = X_selected.iloc[train_idx], X_selected.iloc[test_idx]
                y_train, y_test = self.y.iloc[train_idx], self.y.iloc[test_idx]
    
    
                hidden_layer_sizes = int((X_train.shape[1] + len(np.unique(y_train)))/2)
                self.model.hidden_layer_sizes = hidden_layer_sizes
                
                self.model.fit(X_train, y_train)
                y_pred = self.model.predict(X_test)
                
                accuracies.append(accuracy_score(y_test, y_pred))
                epochs_used.append(self.model.n_iter_)
    
            duration = time.time() - t1
            
            avg_accuracy = np.median(accuracies)
            avg_epochs = int(np.median(epochs_used))
            
            # Store results
            results[n] = {
                'num_features': len(selected_features),
                'features': selected_features,
                'accuracy': avg_accuracy,
                'time_taken' : duration,
                'epochs': avg_epochs
    
            }
            
            print(f"Threshold n={n}: {len(selected_features)} features, accuracy = {avg_accuracy:.4f}")
        
        return results
