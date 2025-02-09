import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import string

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Tokenization of text
    tokens = word_tokenize(text)
    # Remove stopwords from text
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    # Lemmatization 
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    # Join tokens back into text
    return ' '.join(tokens)

def main():
    # Read the data
    df = pd.read_csv('../data/raw/spam_data.csv')
    
    # Display initial class distribution
    print("Initial class distribution:")
    print(df['label'].value_counts())
    
    # Preprocess the text data
    print("\nPreprocessing text data...")
    df['processed_text'] = df['sms'].apply(preprocess_text)
    
    # Split features and target
    X = df['processed_text']
    y = df['label']
    
    # Apply SMOTE to handle class imbalance
    print("\nApplying SMOTE to balance data...")
    smote = SMOTE(random_state=42)
    
    # Convert text to dummy features for SMOTE
    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer(max_features=1000)
    X_vec = vectorizer.fit_transform(X)
    
    # Apply SMOTE
    X_balanced, y_balanced = smote.fit_resample(X_vec, y)
    
    # Convert back to DataFrame
    X_balanced_df = pd.DataFrame(
        X_balanced.toarray(),
        columns=vectorizer.get_feature_names_out()
    )
    
    # Create balanced dataset
    balanced_df = pd.DataFrame({
        'processed_text': [' '.join([word for word, count in zip(vectorizer.get_feature_names_out(), row) if count > 0])
                          for row in X_balanced.toarray()],
        'label': y_balanced,
        'original_text': [df['sms'].iloc[i] if i < len(df) else 'SMOTE_generated' 
                         for i in range(len(y_balanced))]
    })
    
    # Save processed dataset
    balanced_df.to_csv('../data/processed_spam_data.csv', index=False)
    
    # Display final class distribution
    print("\nFinal class distribution:")
    print(balanced_df['label'].value_counts())
    
    print("\nProcessed file saved as 'processed_spam_data.csv'")
    print(f"Total number of samples: {len(balanced_df)}")

if __name__ == "__main__":
    main()