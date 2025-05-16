import numpy as np
import time
from datetime import datetime
import pandas as pd
import pickle as pkl

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from collections import Counter
from statistical_measures import Statistics

mlflow.set_tracking_uri("http://127.0.0.1:5000")  # Use your server address
mlflow.set_experiment("Feature_Selection_Experiments")

class ModelPipeline:
    """
    A pipeline for feature selection and model evaluation using k-fold cross-validation.

    This class applies multiple statistical feature ranking techniques, evaluates
    model performance with varying thresholds of selected features, and logs results
    using MLflow.
    """
    def __init__(self, model = None, X = None, y = None):
        self.model = MLPClassifier(activation = 'logistic', solver = 'lbfgs', batch_size = 'auto', learning_rate = 'adaptive', \
                                    learning_rate_init = 0.03, max_iter = 5000, \
                                    momentum = 0.2, \
                                    random_state = np.random.get_state()[1][0], \
                                    early_stopping = False)
        self.X = X
        self.y = y
        self.results_path = '../data/results'
        self.model_path = '../models'

    def vertical_split(self, df, target):
        """
        Splits a DataFrame into normalized features and target variable.
        
        Parameters:
        - df (DataFrame): Input data containing features and target column.
        - target (str): The name of the target column.
        """

        self.X = df.drop(columns = [target])
        self.X = (self.X - self.X.min())/(self.X.max() - self.X.min())
        #self.X = pd.DataFrame(X_normalized, columns = self.X.columns)
        self.y = df[target]

    def train_with_kfold(self): 
        """
        Trains the model using 10-fold stratified cross-validation with different feature
        subsets selected by multiple statistical methods.
    
        Applies several feature ranking methods, evaluates the model on top-ranked features,
        logs the process and results with MLflow, and saves the best model.
    
        Returns:
        - DataFrame: A summary table of evaluation metrics for each threshold level.
        """
        
        sta = Statistics()
        statistical_methods = {
            'PCA': sta.pca_ranking,
            'Dispersion_Ratio': sta.dispersion_ratio,
            'Chi_Square': sta.chi_square_ranking,
            'Pearson_Correlation': sta.pearson_correlation,
            'Mean_Absolute_Difference': sta.mean_absolute_difference,
            'Low_Variance': sta.low_variance
        }
    
        top_features_all_methods = []
        
        for method_name, method in statistical_methods.items():
            print(f"Applying {method_name} feature selection...")
            ranked_features = method(self.X, self.y)
            top_quartile_count = max(1, len(ranked_features) // 4)
            top_features = ranked_features[:top_quartile_count]
            top_features_all_methods.append(set(top_features))
    
        feature_counts = Counter(f for s in top_features_all_methods for f in s)
    
        selected_sets = {
            n: {f for f, count in feature_counts.items() if count >= n}
            for n in range(1, len(statistical_methods) + 1)
        }
    
        results = {}
        models = {}
    
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        for n in range(0, len(statistical_methods) + 1):
            if n == 0:
                selected_features = self.X.columns.tolist()
            else:
                selected_features = list(selected_sets[n])
                if not selected_features:
                    results[n] = {
                        'num_features': 0,
                        'features': [],
                        'accuracy': None
                    }
                    print(f"No Features Selected for Threshold n = {n}")
                    continue
    
            print(f"Evaluating Model on {'All Features' if n == 0 else f'Features Selected by ≥ {n} Methods'}...")
            X_selected = self.X[selected_features]
    
            accuracies = []
            epochs_used = []
            cv_generator = StratifiedKFold(n_splits=10, shuffle=False)
            t1 = time.time()
    
            for train_idx, test_idx in cv_generator.split(X_selected, self.y):
                X_train, X_test = X_selected.iloc[train_idx], X_selected.iloc[test_idx]
                y_train, y_test = self.y.iloc[train_idx], self.y.iloc[test_idx]
    
                hidden_layer_sizes = int((X_train.shape[1] + len(np.unique(y_train))) / 2)
                self.model.hidden_layer_sizes = hidden_layer_sizes
    
                self.model.fit(X_train, y_train)
                
                signature = infer_signature(X_train, self.model.predict(X_train))
                mlflow.sklearn.log_model(
                    sk_model = self.model,
                    artifact_path = "model",
                    input_example = X_train[:5],
                    signature = signature
                )

                y_pred = self.model.predict(X_test)
    
                accuracies.append(accuracy_score(y_test, y_pred))
                epochs_used.append(self.model.n_iter_)
    
            duration = time.time() - t1
            avg_accuracy = np.median(accuracies)
            avg_epochs = int(np.median(epochs_used))
    
            results[n] = {
                'num_features': len(selected_features),
                'features': selected_features,
                'accuracy': avg_accuracy,
                'time_taken': duration,
                'epochs': avg_epochs
            }
    
            models[n] = self.model
    
            print(f"Threshold n = {n}: {len(selected_features)} features, accuracy = {avg_accuracy:.4f}")
    
            # Log to MLflow
            with mlflow.start_run(run_name = f"KFold_n_{n}_{timestamp}", nested = True):
                mlflow.log_param("threshold_n", n)
                mlflow.log_param("num_features", len(selected_features))
                mlflow.log_metric("accuracy", avg_accuracy)
                mlflow.log_metric("training_time_sec", duration)
                mlflow.log_metric("epochs", avg_epochs)
    
                mlflow.set_tag("developer", "Ridwan")
    
                # Log the model
                mlflow.sklearn.log_model(self.model, artifact_path = "model", input_example = X_train[:5])
    
        result_df = pd.DataFrame(results).T
        result_df = result_df.query('num_features > 0').reset_index(drop=True)
        result_csv_path = f"{self.results_path}/results_{timestamp}.csv"
        result_df.to_csv(result_csv_path, index=False)
    
        # Log CSV once (not for each run)
        with mlflow.start_run(run_name = f"Final_Results_{timestamp}", nested = True):
            mlflow.log_artifact(result_csv_path, artifact_path="results")
    
        best_n = max(
            (k for k in results if results[k]['accuracy'] is not None),
            key=lambda x: results[x]['accuracy']
        )
    
        best_model = models[best_n]
        model_file_path = f"{self.model_path}/model_{timestamp}.pkl"
        with open(model_file_path, 'wb') as f:
            pkl.dump(best_model, f)
    
        # Log final best model separately
        with mlflow.start_run(run_name = f"Best_Model_n_{best_n}_{timestamp}", nested = True):
            mlflow.log_param("best_n", best_n)
            mlflow.log_metric("best_accuracy", results[best_n]["accuracy"])
            mlflow.log_artifact(model_file_path, artifact_path="best_model")
    
        return result_df
