# main.py

import mlflow
import os

import config
from preprocess import load_and_preprocess_data
from train_ml import train_ml_models
from train_cnn import train_cnn_model

def main():
    # Thiết lập MLflow
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)
    print(f"MLflow experiment được đặt thành: '{config.MLFLOW_EXPERIMENT_NAME}'")
    
    # Tạo thư mục lưu model nếu chưa có
    if not os.path.exists(config.MODEL_SAVE_DIR):
        os.makedirs(config.MODEL_SAVE_DIR)
        
    # 1. Tải và tiền xử lý dữ liệu
    df, le, num_classes = load_and_preprocess_data()
    
    # 2. Huấn luyện các mô hình ML
    train_ml_models(df, le, num_classes)
    
    # 3. Huấn luyện mô hình CNN
    train_cnn_model(df, le, num_classes)
    
    print("\n--- Pipeline đã hoàn tất! ---")
    print(f"Kiểm tra kết quả tại MLflow UI bằng cách chạy 'mlflow ui' trong terminal.")

if __name__ == "__main__":
    main()  