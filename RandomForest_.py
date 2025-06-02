import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

def load_and_prepare_data(filename, selected_features, chunk_size=50000):
    """Load data in chunks and prepare for training"""
    print("Loading and preparing data...")
    
    # Add 'is_attack' to features as it's our target
    features = selected_features + ['is_attack']
    
    # Specify dtypes
    dtype_dict = {
        'TotPkts': 'float32',
        'TotBytes': 'float32',
        'TotAppByte': 'float32',
        'Loss': 'float32',
        'Rate': 'float32',
        'SrcRate': 'float32',
        'DstRate': 'float32',
        'Protocol_std': 'str',
        'Flags_std': 'str',
        'Direction_std': 'str',
        'State_std': 'str',
        'Sport_category': 'str',
        'Dport_category': 'str',
        'SrcAddr_type': 'str',
        'DstAddr_type': 'str',
        'bytes_per_packet': 'float32',
        'duration_seconds': 'float32',
        'traffic_direction': 'str',
        'traffic_type': 'str',
        'is_tcp': 'str',
        'is_udp': 'str',
        'is_icmp': 'str',
        'has_well_known_port': 'str',
        'is_internal': 'str',
        'is_outbound': 'str',
        'is_attack': 'str'
    }
    
    # Initialize empty list to store chunks
    chunks = []
    
    # Read and process data in chunks
    for chunk in tqdm(pd.read_csv(filename, usecols=features, chunksize=chunk_size, dtype=dtype_dict)):
        # Convert boolean-like strings to int
        bool_cols = ['is_attack', 'is_tcp', 'is_udp', 'is_icmp', 
                    'has_well_known_port', 'is_internal', 'is_outbound']
        for col in bool_cols:
            if col in chunk.columns:
                # Convert various forms of True/False to integers
                chunk[col] = (chunk[col].str.upper() == 'TRUE').astype(np.int8)
        
        chunks.append(chunk)
    
    # Combine all chunks
    data = pd.concat(chunks, axis=0)
    
    # Convert categorical columns to numeric using LabelEncoder
    categorical_cols = [col for col in data.columns if data[col].dtype == 'object']
    encoders = {}
    
    for col in categorical_cols:
        encoders[col] = LabelEncoder()
        data[col] = encoders[col].fit_transform(data[col].astype(str))
    
    return data

def train_random_forest(X_train, X_test, y_train, y_test):
    """Train and evaluate Random Forest model"""
    print("Training Random Forest model...")
    
    # Initialize and train the model
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        n_jobs=-1,
        random_state=42
    )
    
    rf.fit(X_train, y_train)
    
    # Make predictions
    y_pred = rf.predict(X_test)
    
    # Calculate and print metrics
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    
    return rf, y_pred

def plot_feature_importance(rf, feature_names):
    """Plot feature importance from the trained model"""
    importance = pd.DataFrame({
        'feature': feature_names,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(data=importance, x='importance', y='feature')
    plt.title('Feature Importance from Trained Model')
    plt.tight_layout()
    plt.show()
    
    return importance

def main():
    # Load selected features from file
    with open('selected_features.txt', 'r') as f:
        selected_features = [line.strip() for line in f]
    
    print("Selected features:", selected_features)
    
    # Load and prepare data
    data = load_and_prepare_data("netflow_data.csv", selected_features)
    
    # Split features and target
    X = data.drop('is_attack', axis=1)
    y = data['is_attack']
    
    # Print class distribution
    print("\nClass distribution:")
    print(y.value_counts(normalize=True))
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    
    # Train model and get predictions
    rf_model, predictions = train_random_forest(X_train, X_test, y_train, y_test)
    
    # Plot feature importance from the trained model
    importance = plot_feature_importance(rf_model, X.columns)
    
    # Save the model
    from joblib import dump
    dump(rf_model, 'random_forest_model.joblib')
    
    return rf_model, importance

if __name__ == "__main__":
    try:
        print("Starting Random Forest classification...")
        rf_model, importance = main()
        print("\nModel training completed successfully!")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()