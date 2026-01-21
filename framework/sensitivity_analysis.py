import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')

def perform_sensitivity_analysis(model, X, y, hyperparameters):
    """
    Perform sensitivity analysis on a model
    
    Parameters:
    - model: sklearn model instance
    - X: feature data
    - y: target data
    - hyperparameters: dict of hyperparameters
    
    Returns:
    - dict with f1_score, auc, and accuracy
    """
    try:
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Get probability predictions if available
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, 'decision_function'):
            # For LinearSVC and similar models
            y_pred_proba = model.decision_function(X_test)
        else:
            y_pred_proba = y_pred
        
        # Calculate metrics
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        # Handle auc calculation for different model types
        try:
            auc = roc_auc_score(y_test, y_pred_proba)
        except:
            auc = 0.0
            
        accuracy = accuracy_score(y_test, y_pred)
        
        return {
            'f1_score': f1,
            'auc': auc,
            'accuracy': accuracy
        }
    except Exception as e:
        print(f"    Error: {str(e)}")
        return {
            'f1_score': 0,
            'auc': 0,
            'accuracy': 0
        }