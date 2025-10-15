import pandas as pd
import numpy as np
import re
import os
import requests
from PIL import Image
from tqdm import tqdm
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import gc
import warnings
warnings.filterwarnings('ignore')
import torch
from sentence_transformers import SentenceTransformer
from torchvision import models, transforms

DATA_DIR = 'dataset/'
IMAGE_DIR = 'images/'
TRAIN_IMAGE_DIR = os.path.join(IMAGE_DIR, 'train')
TEST_IMAGE_DIR = os.path.join(IMAGE_DIR, 'test')
os.makedirs(TRAIN_IMAGE_DIR, exist_ok=True)
os.makedirs(TEST_IMAGE_DIR, exist_ok=True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")

def smape(y_true, y_pred):
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    denominator[denominator == 0] = 1e-6
    return np.mean(numerator / denominator) * 100

def download_image(sample_id, url, save_dir):
    filepath = os.path.join(save_dir, f"{sample_id}.jpg")
    if os.path.exists(filepath):
        return filepath
    try:
        response = requests.get(url, stream=True, timeout=10)
        if response.status_code == 200:
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            return filepath
    except Exception:
        return None
    return None

def extract_ipq(text):
    text = str(text).lower()
    patterns = [
        r'ipq:\s*(\d+)',
        r'pack of\s*(\d+)',
        r'(\d+)\s*pack',
        r'(\d+)\s*count',
        r'(\d+)\s*ct'
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return int(match.group(1))
    return 1

def generate_text_embeddings(df):
    print("Generating text embeddings...")
    model = SentenceTransformer('all-MiniLM-L6-v2', device=DEVICE)
    cleaned_content = df['catalog_content'].str.lower().str.replace(r'ipq:\s*\d+|pack of\s*\d+|\d+\s*pack|\d+\s*count|\d+\s*ct', '', regex=True)
    embeddings = model.encode(cleaned_content.tolist(), show_progress_bar=True, batch_size=128)
    return embeddings

def generate_image_embeddings(df, image_dir):
    print("Generating image embeddings...")
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    model.classifier = torch.nn.Identity()
    model.to(DEVICE)
    model.eval()
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    all_embeddings = []
    print("Downloading images (this may take a while)...")
    df.apply(lambda row: download_image(row['sample_id'], row['image_link'], image_dir), axis=1)
    for sample_id in tqdm(df['sample_id'], desc="Processing images"):
        image_path = os.path.join(image_dir, f"{sample_id}.jpg")
        try:
            with Image.open(image_path).convert('RGB') as img:
                img_t = preprocess(img)
                batch_t = torch.unsqueeze(img_t, 0).to(DEVICE)
                with torch.no_grad():
                    embedding = model(batch_t).squeeze().cpu().numpy()
                all_embeddings.append(embedding)
        except Exception:
            all_embeddings.append(np.zeros(1280))
    return np.array(all_embeddings)

if __name__ == '__main__':
    print("Loading data...")
    try:
        train_df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
        test_df = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure the dataset folder is in the correct location.")
        exit()
    print("Starting feature engineering...")
    train_df['ipq'] = train_df['catalog_content'].apply(extract_ipq)
    test_df['ipq'] = test_df['catalog_content'].apply(extract_ipq)
    train_text_embeddings = generate_text_embeddings(train_df)
    test_text_embeddings = generate_text_embeddings(test_df)
    train_image_embeddings = generate_image_embeddings(train_df, TRAIN_IMAGE_DIR)
    test_image_embeddings = generate_image_embeddings(test_df, TEST_IMAGE_DIR)
    print("Combining features...")
    X_train_ipq = train_df[['ipq']].values
    X_train = np.concatenate([X_train_ipq, train_text_embeddings, train_image_embeddings], axis=1)
    y_train = train_df['price'].values
    y_train_log = np.log1p(y_train)
    X_test_ipq = test_df[['ipq']].values
    X_test = np.concatenate([X_test_ipq, test_text_embeddings, test_image_embeddings], axis=1)
    del train_text_embeddings, test_text_embeddings, train_image_embeddings, test_image_embeddings
    gc.collect()
    print("Training LightGBM model...")
    lgb_params = {
        'objective': 'regression_l1',
        'metric': 'rmse',
        'n_estimators': 2000,
        'learning_rate': 0.02,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'lambda_l1': 0.1,
        'lambda_l2': 0.1,
        'num_leaves': 31,
        'verbose': -1,
        'n_jobs': -1,
        'seed': 42,
        'boosting_type': 'gbdt',
    }
    model = lgb.LGBMRegressor(**lgb_params)
    model.fit(X_train, y_train_log)
    print("Generating predictions...")
    log_predictions = model.predict(X_test)
    predictions = np.expm1(log_predictions)
    predictions[predictions < 0] = 0
    submission_df = pd.DataFrame({'sample_id': test_df['sample_id'], 'price': predictions})
    submission_df.to_csv('test_out.csv', index=False)
    print("Submission file 'test_out.csv' created successfully!")
    print("File shape:", submission_df.shape)
    print("Sample predictions:")
    print(submission_df.head())
