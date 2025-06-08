import torch

# --- Đường dẫn và Cấu hình chung ---
DATA_PATH = './data/legal_text_classification.csv'
MODEL_SAVE_DIR = "saved_models"
MLFLOW_EXPERIMENT_NAME = "Legal_Text_Classification_Pipeline_v2"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Cấu hình Tiền xử lý ---
RANDOM_STATE = 42
TEST_SIZE = 0.2

# --- Cấu hình Vectorizer ---
MAX_FEATURES_ML = 3000

# --- Cấu hình Word Embedding (FastText) ---
W2V_SIZE = 300
W2V_WINDOW = 5
W2V_MIN_COUNT = 3
W2V_WORKERS = 4 # Số core CPU để sử dụng

# --- Cấu hình Mô hình ML ---
# LinearSVC
LSVC_PARAMS = {
    'class_weight': 'balanced',
    'dual': False, # An toàn cho n_samples > n_features
    'C': 1.0,
    'max_iter': 2000
}

# Logistic Regression
LR_PARAMS = {
    'solver': 'liblinear',
    'class_weight': 'balanced',
    'C': 1.0,
    'max_iter': 1000
}

# --- Cấu hình Mô hình CNN ---
CNN_EMBEDDING_DIM = 300 # Phải khớp với W2V_SIZE của FastText
CNN_HIDDEN_SIZE = 128
CNN_MAX_SEQ_LENGTH = 250
CNN_DROPOUT_RATE = 0.5
CNN_EPOCHS = 10 # Giảm bớt để chạy nhanh hơn, có thể tăng lại
CNN_BATCH_SIZE = 32
CNN_LEARNING_RATE = 0.001