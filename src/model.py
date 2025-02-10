from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class SpamDetectionModel:
    """
    Class for training and evaluating spam detection model with improved NaN handling
    """
    
    def __init__(self):
        """Initialize model components"""
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words='english'
        )
        self.classifier = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
    
    def _clean_texts(self, texts):
        """
        Clean text data by handling NaN values
        
        Args:
            texts (array-like): Array of text messages
            
        Returns:
            array: Cleaned text messages
        """
        # Convert to pandas Series if not already
        texts = pd.Series(texts)
        
        # Replace NaN values with empty string
        texts = texts.fillna('')
        
        # Convert to list for vectorizer
        return texts.tolist()
    
    def prepare_features(self, texts, training=True):
        """
        Convert text to TF-IDF features with NaN handling
        
        Args:
            texts (array-like): Array of text messages
            training (bool): Whether this is for training or prediction
            
        Returns:
            scipy.sparse.csr_matrix: TF-IDF features
        """
        # Clean texts before vectorization
        cleaned_texts = self._clean_texts(texts)
        
        if training:
            return self.vectorizer.fit_transform(cleaned_texts)
        return self.vectorizer.transform(cleaned_texts)
    
    def train(self, X_train, y_train):
        """
        Train the spam detection model
        
        Args:
            X_train (array-like): Training features
            y_train (array-like): Training labels
        """
        # Convert text to features
        X_train_features = self.prepare_features(X_train)
        
        # Train the model
        self.classifier.fit(X_train_features, y_train)
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance
        
        Args:
            X_test (array-like): Test features
            y_test (array-like): Test labels
            
        Returns:
            dict: Dictionary containing evaluation metrics
        """
        # Convert text to features
        X_test_features = self.prepare_features(X_test, training=False)
        
        # Make predictions
        y_pred = self.classifier.predict(X_test_features)
        
        # Calculate metrics
        report = classification_report(y_test, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        return {
            'classification_report': report,
            'confusion_matrix': conf_matrix
        }
    
    def predict(self, texts):
        """
        Make predictions on new texts
        
        Args:
            texts (array-like): Array of text messages
            
        Returns:
            array: Predicted labels (0 for ham, 1 for spam)
        """
        # Convert text to features
        features = self.prepare_features(texts, training=False)
        
        # Make predictions
        return self.classifier.predict(features)
    
    def save_model(self, model_path, vectorizer_path):
        """
        Save the trained model and vectorizer
        
        Args:
            model_path (str): Path to save the model
            vectorizer_path (str): Path to save the vectorizer
        """
        joblib.dump(self.classifier, model_path)
        joblib.dump(self.vectorizer, vectorizer_path)
    
    @classmethod
    def load_model(cls, model_path, vectorizer_path):
        """
        Load a trained model and vectorizer
        
        Args:
            model_path (str): Path to the saved model
            vectorizer_path (str): Path to the saved vectorizer
            
        Returns:
            SpamDetectionModel: Loaded model instance
        """
        instance = cls()
        instance.classifier = joblib.load(model_path)
        instance.vectorizer = joblib.load(vectorizer_path)
        return instance

if __name__ == "__main__":
    # Load processed data
    data = pd.read_csv(r'C:\vnit\MLOps\mlops_assignment_2_v1\data\processed_spam_data.csv')
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        data['processed_text'],
        data['label'],
        test_size=0.2,
        random_state=42
    )
    
    # Train and evaluate model
    model = SpamDetectionModel()
    model.train(X_train, y_train)
    metrics = model.evaluate(X_test, y_test)
    
    # Save model
    model.save_model(r'C:\vnit\MLOps\mlops_assignment_2_v1\models\spam_model.joblib', 
                     r'C:\vnit\MLOps\mlops_assignment_2_v1\models\vectorizer.joblib')
    
    # Print evaluation results
    print("Model Performance:")
    print("\nClassification Report:")
    print(pd.DataFrame(metrics['classification_report']).transpose())