import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

def load_and_prepare_data(filename, selected_features, chunk_size=50000):
    """Load data in chunks and prepare for training"""
    print("Loading and preparing data...")
    
    features = selected_features + ['is_attack']
    
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
    
    chunks = []
    
    for chunk in tqdm(pd.read_csv(filename, usecols=features, chunksize=chunk_size, dtype=dtype_dict)):
        # Convert boolean-like strings to int
        bool_cols = ['is_attack', 'is_tcp', 'is_udp', 'is_icmp', 
                    'has_well_known_port', 'is_internal', 'is_outbound']
        for col in bool_cols:
            if col in chunk.columns:
                chunk[col] = (chunk[col].str.upper() == 'TRUE').astype(np.int8)
        
        chunks.append(chunk)
    
    data = pd.concat(chunks, axis=0)
    
    # Convert categorical columns to numeric
    categorical_cols = [col for col in data.columns if data[col].dtype == 'object']
    encoders = {}
    
    for col in categorical_cols:
        encoders[col] = LabelEncoder()
        data[col] = encoders[col].fit_transform(data[col].astype(str))
    
    return data

def train_xgboost(X_train, X_test, y_train, y_test):
    """Train and evaluate XGBoost model"""
    print("Training XGBoost model...")
    
    # Calculate scale_pos_weight
    scale_pos_weight = len(y_train[y_train==0]) / len(y_train[y_train==1])
    
    # Initialize XGBoost classifier
    xgb_model = xgb.XGBClassifier(
        max_depth=6,
        learning_rate=0.1,
        n_estimators=100,
        objective='binary:logistic',
        scale_pos_weight=scale_pos_weight,
        tree_method='hist',
        n_jobs=-1,
        random_state=42,
        eval_metric=['error', 'auc'],
        enable_categorical=True  # Enable categorical feature support
    )
    
    # Train model
    xgb_model.fit(
        X_train, 
        y_train,
        verbose=True
    )
    
    # Make predictions
    y_pred = xgb_model.predict(X_test)
    
    # Print metrics
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
    
    # Calculate FAR and FRR
    tn, fp, fn, tp = cm.ravel()
    far = fp / (fp + tn)
    frr = fn / (fn + tp)
    
    print("\nAdditional Metrics:")
    print(f"False Acceptance Rate (FAR): {far:.4f}")
    print(f"False Rejection Rate (FRR): {frr:.4f}")
    
    return xgb_model, y_pred

def plot_feature_importance(model, feature_names):
    """Plot feature importance from the trained model"""
    importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(data=importance, x='importance', y='feature')
    plt.title('Feature Importance from XGBoost Model')
    plt.tight_layout()
    plt.show()
    
    return importance

def main():
    # Load selected features
    with open('selected_features.txt', 'r') as f:
        selected_features = [line.strip() for line in f]
    
    print("Selected features:", selected_features)
    
    # Load and prepare data
    data = load_and_prepare_data("netflow_data.csv", selected_features)
    
    # Split features and target
    X = data.drop('is_attack', axis=1)
    y = data['is_attack']
    
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
    xgb_model, predictions = train_xgboost(X_train, X_test, y_train, y_test)
    
    # Plot feature importance
    importance = plot_feature_importance(xgb_model, X.columns)
    
    # Save the model
    xgb_model.save_model('xgboost_model.json')
    
    return xgb_model, importance

if __name__ == "__main__":
    try:
        print("Starting XGBoost classification...")
        xgb_model, importance = main()
        print("\nModel training completed successfully!")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()