# train_ml.py

import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score

import config
from preprocess import get_ml_datasets

def compute_metrics(y_true, y_pred, model_name, le, num_classes):
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    print(f"  Accuracy: {accuracy*100:.3f}% | F1 Score (Macro): {f1*100:.3f}%")
    
    mlflow.log_metrics({"test_accuracy": accuracy, "test_f1_macro": f1})
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(max(6, num_classes // 2), max(4, num_classes // 2.5)))
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', 
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted Label'); plt.ylabel('True Label')
    plt.tight_layout()
    
    cm_path = f"confusion_matrix_{model_name.replace(' ', '_')}.png"
    plt.savefig(cm_path)
    mlflow.log_artifact(cm_path, "plots")
    plt.close()

def train_ml_models(df, le, num_classes):
    print("\n--- Bắt đầu Huấn luyện các Mô hình ML ---")
    (X_train_stm, X_test_stm, y_train_stm, y_test_stm), \
    (X_train_lem, X_test_lem, y_train_lem, y_test_lem) = get_ml_datasets(df)
    
    # --- Huấn luyện Logistic Regression ---
    run_name_lr = "LogisticRegression_Tfidf_Stm"
    with mlflow.start_run(run_name=run_name_lr):
        print(f"\n[MLflow Run] {run_name_lr}")
        mlflow.log_params({"model_type": "LogisticRegression", "vectorizer": "Tfidf", "preprocessing": "Stemming"})
        mlflow.log_params(config.LR_PARAMS)
        
        lr_model = LogisticRegression(**config.LR_PARAMS)
        lr_model.fit(X_train_stm, y_train_stm)
        
        y_pred = lr_model.predict(X_test_stm)
        compute_metrics(y_test_stm, y_pred, run_name_lr, le, num_classes)
        mlflow.sklearn.log_model(lr_model, "model")
        print(f"Đã hoàn tất run: {run_name_lr}")

    # --- Huấn luyện LinearSVC ---
    run_name_lsvc = "LinearSVC_Tfidf_Lem"
    with mlflow.start_run(run_name=run_name_lsvc):
        print(f"\n[MLflow Run] {run_name_lsvc}")
        mlflow.log_params({"model_type": "LinearSVC", "vectorizer": "Tfidf", "preprocessing": "Lemmatization"})
        mlflow.log_params(config.LSVC_PARAMS)
        
        lsvc_model = LinearSVC(**config.LSVC_PARAMS)
        lsvc_model.fit(X_train_lem, y_train_lem)
        
        y_pred = lsvc_model.predict(X_test_lem)
        compute_metrics(y_test_lem, y_pred, run_name_lsvc, le, num_classes)
        mlflow.sklearn.log_model(lsvc_model, "model")
        print(f"Đã hoàn tất run: {run_name_lsvc}")