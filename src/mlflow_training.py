import mlflow
import mlflow.sklearn
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
from scipy.stats import randint, uniform
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class MLflowSpamTrainer:
    def __init__(self, experiment_name="spam_detection"):
        mlflow.set_experiment(experiment_name)
        self.param_distributions = {
            'n_estimators': randint(50, 200),
            'max_depth': randint(10, 50),
            'min_samples_split': randint(2, 20),
            'min_samples_leaf': randint(1, 10),
            'max_features': uniform(0.1, 0.9)
        }
    
    def preprocess_text_data(self, X):
        if not isinstance(X, pd.Series):
            X = pd.Series(X)
        X = X.fillna('')
        X = X.astype(str)
        return X
        
    def evaluate_model(self, model, X, y):
        """
        Evaluate model and return metrics
        """
        y_pred = model.predict(X)
        return {
            "accuracy": accuracy_score(y, y_pred),
            "precision": precision_score(y, y_pred),
            "recall": recall_score(y, y_pred),
            "f1": f1_score(y, y_pred)
        }

    def custom_cv_iterator(self, X, y, cv_splits):
        for i, (train_idx, val_idx) in enumerate(cv_splits):
            with mlflow.start_run(nested=True):
                yield train_idx, val_idx

    def optimize_model(self, X_train, y_train, X_test, y_test, n_iter=20):
        # Preprocess text data
        X_train = self.preprocess_text_data(X_train)
        X_test = self.preprocess_text_data(X_test)
        
        # Create TF-IDF features
        vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words='english'
        )
        X_train_features = vectorizer.fit_transform(X_train)
        X_test_features = vectorizer.transform(X_test)

        # Start parent MLflow run
        with mlflow.start_run(run_name="hyperparameter_optimization") as parent_run:
            mlflow.log_param("n_iterations", n_iter)
            mlflow.log_param("vectorizer_max_features", 5000)
            
            # Initialize base model
            base_model = RandomForestClassifier(random_state=42, n_jobs=-1)
            
            # Custom scorer that logs metrics to MLflow
            def custom_scorer(estimator, X, y):
                with mlflow.start_run(nested=True) as child_run:
                    # Log parameters of current model
                    params = estimator.get_params()
                    mlflow.log_params(params)
                    
                    # Calculate and log metrics
                    metrics = self.evaluate_model(estimator, X, y)
                    mlflow.log_metrics(metrics)
                    
                    # Return f1 score as the optimization metric
                    return metrics['f1']
            
            # Random search with cross-validation
            random_search = RandomizedSearchCV(
                base_model,
                param_distributions=self.param_distributions,
                n_iter=n_iter,
                cv=5,
                random_state=42,
                n_jobs=1,  # Set to 1 to ensure sequential processing for proper MLflow tracking
                scoring=custom_scorer,
                return_train_score=True
            )
            
            # Fit the random search
            random_search.fit(X_train_features, y_train)
            
            # Log best model details
            mlflow.log_params(random_search.best_params_)
            
            # Evaluate best model on test set
            best_model = random_search.best_estimator_
            test_metrics = self.evaluate_model(best_model, X_test_features, y_test)
            
            # Log test metrics with 'test_' prefix
            for metric_name, value in test_metrics.items():
                mlflow.log_metric(f"test_{metric_name}", value)
            
            # Log validation results for all iterations
            for i, results in enumerate(random_search.cv_results_['mean_test_score']):
                iter_params = {f"iter_{i}_{k}": v for k, v in 
                             zip(random_search.param_distributions.keys(),
                                 random_search.cv_results_['params'][i].values())}
                mlflow.log_params(iter_params)
                mlflow.log_metric(f"iter_{i}_mean_cv_score", results)
                mlflow.log_metric(f"iter_{i}_std_cv_score", 
                                random_search.cv_results_['std_test_score'][i])
            
            # Log best model and vectorizer
            mlflow.sklearn.log_model(best_model, "model")
            mlflow.sklearn.log_model(vectorizer, "vectorizer")
            
            # Log feature importance plot
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 6))
            feature_importance = pd.DataFrame({
                'feature': vectorizer.get_feature_names_out(),
                'importance': best_model.feature_importances_
            }).sort_values('importance', ascending=False).head(20)
            
            plt.bar(feature_importance['feature'], feature_importance['importance'])
            plt.xticks(rotation=45, ha='right')
            plt.title('Top 20 Most Important Features')
            plt.tight_layout()
            mlflow.log_figure(plt.gcf(), "feature_importance.png")
            plt.close()
            
            return best_model, vectorizer
    
    @staticmethod
    def load_best_model(run_id):
        model_uri = f"runs:/{run_id}/model"
        vectorizer_uri = f"runs:/{run_id}/vectorizer"
        
        model = mlflow.sklearn.load_model(model_uri)
        vectorizer = mlflow.sklearn.load_model(vectorizer_uri)
        
        return model, vectorizer

if __name__ == "__main__":
    # Load processed data
    data = pd.read_csv('../data/processed_spam_data.csv')
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        data['processed_text'],
        data['label'],
        test_size=0.2,
        random_state=42
    )
    
    # Initialize trainer and optimize model
    trainer = MLflowSpamTrainer(experiment_name="spam_detection_detailed")
    best_model, vectorizer = trainer.optimize_model(
        X_train, y_train, X_test, y_test
    )
    
    print("Training completed. Check MLflow UI for detailed results.")
