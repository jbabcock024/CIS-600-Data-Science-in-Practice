import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

# Gaussian NB because we have both categorical and numerical features

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

def train_naive_bayes(X_train, X_test, y_train, y_test):
    """Train and evaluate Naive Bayes model"""
    print("Training Naive Bayes model...")
    
    # Initialize and train the model
    nb_model = GaussianNB()
    nb_model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = nb_model.predict(X_test)
    
    # Analyze predictions
    print("\nPrediction distribution:")
    pred_dist = pd.Series(y_pred).value_counts()
    print(pred_dist)
    
    print("\nActual distribution in test set:")
    actual_dist = pd.Series(y_test).value_counts()
    print(actual_dist)
    
    # Print metrics with zero_division parameter
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
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
    far = fp / (fp + tn) if (fp + tn) > 0 else 0
    frr = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    print("\nDetailed Metrics:")
    print(f"True Negatives (TN): {tn}")
    print(f"False Positives (FP): {fp}")
    print(f"False Negatives (FN): {fn}")
    print(f"True Positives (TP): {tp}")
    print(f"False Acceptance Rate (FAR): {far:.4f}")
    print(f"False Rejection Rate (FRR): {frr:.4f}")
    
    # Calculate and show probability distributions
    y_prob = nb_model.predict_proba(X_test)
    print("\nProbability distribution statistics:")
    prob_df = pd.DataFrame(y_prob, columns=['Prob_Class_0', 'Prob_Class_1'])
    print(prob_df.describe())
    
    # Plot probability distributions
    plt.figure(figsize=(10, 6))
    plt.hist(y_prob[:, 1], bins=50, alpha=0.5, label='Class 1 probabilities')
    plt.hist(y_prob[:, 0], bins=50, alpha=0.5, label='Class 0 probabilities')
    plt.title('Probability Distributions')
    plt.xlabel('Probability')
    plt.ylabel('Count')
    plt.legend()
    plt.show()
    
    return nb_model, y_pred

def analyze_feature_importance(nb_model, feature_names, X_test, y_test):
    """Analyze feature importance for Naive Bayes"""
    # Calculate feature importance using variance of gaussian distributions
    var_per_class = nb_model.var_
    mean_per_class = nb_model.theta_
    
    # Calculate overall importance as the difference in means weighted by inverse variance
    importance = np.abs(mean_per_class[1] - mean_per_class[0]) / np.sqrt(var_per_class[0] + var_per_class[1])
    
    # Create importance DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance,
        'mean_diff': np.abs(mean_per_class[1] - mean_per_class[0]),
        'variance_class_0': var_per_class[0],
        'variance_class_1': var_per_class[1]
    }).sort_values('importance', ascending=False)
    
    # Print detailed feature statistics
    print("\nFeature Statistics:")
    print(importance_df)
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    sns.barplot(data=importance_df, x='importance', y='feature')
    plt.title('Feature Importance in Naive Bayes Model')
    plt.tight_layout()
    plt.show()
    
    # Plot mean differences between classes
    plt.figure(figsize=(12, 8))
    sns.barplot(data=importance_df, x='mean_diff', y='feature')
    plt.title('Mean Difference Between Classes')
    plt.tight_layout()
    plt.show()
    
    return importance_df

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
    nb_model, predictions = train_naive_bayes(X_train, X_test, y_train, y_test)
    
    # Analyze feature importance
    importance = analyze_feature_importance(nb_model, X.columns, X_test, y_test)
    
    # Save the model
    from joblib import dump
    dump(nb_model, 'naive_bayes_model.joblib')
    
    return nb_model, importance

if __name__ == "__main__":
    try:
        print("Starting Naive Bayes classification...")
        nb_model, importance = main()
        print("\nModel training completed successfully!")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()