import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def train_ml_models():
    print("Loading extracted features...")
    features = np.load('features/features.npy')
    labels = np.load('features/labels.npy')

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    print(f"Training on {X_train.shape[0]} samples, Testing on {X_test.shape[0]} samples.")

    # Initialize models
    models = {
        "SVM": SVC(kernel='linear', C=1.0),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000)
    }

    os.makedirs('models', exist_ok=True)

    print("\n--- Model Evaluation ---")
    print(f"{'Model Name':<20} | {'Test Accuracy':<15}")
    print("-" * 40)

    for name, model in models.items():
        # Train
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Evaluate
        accuracy = accuracy_score(y_test, y_pred)
        print(f"{name:<20} | {accuracy:.4f}")

        # Save model
        with open(f'models/{name.replace(" ", "_").lower()}_model.pkl', 'wb') as f:
            pickle.dump(model, f)

    print("-" * 40)
    print("All ML models trained and saved in 'models/' directory.")

if __name__ == "__main__":
    train_ml_models()
