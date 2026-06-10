import os
import pandas as pd
import numpy as np
import warnings
import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

warnings.filterwarnings('ignore')

try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except:
    SCRIPT_DIR = os.getcwd()

DATASET_PATH = os.path.join(SCRIPT_DIR, 'datasetkepuasan', 'train.csv')

RANDOM_STATE = 42
TEST_SIZE = 0.2
K = 5
DEMO_SIZE = 1000  


def load_data(filepath):
    df = pd.read_csv(filepath, sep=';')
    print(f"[OK] Dataset: {df.shape[0]} baris, {df.shape[1]} kolom")
    return df

def preprocess_data(df):
    df = df.copy()
    
    if 'id' in df.columns:
        df = df.drop(columns=['id'])
    
    initial_rows = len(df)
    df = df.dropna()
    rows_dropped = initial_rows - len(df)
    if rows_dropped > 0:
        print(f"[OK] Menghapus {rows_dropped} baris dengan missing values")
    
    y = df['satisfaction'].copy()
    
    print("\nMengahapus fitur yang tidak diperlukan seperti : Age, Gender, id, numb")
    X = df.drop(columns=['satisfaction', 'Age', 'Gender', 'id', 'numb'], errors='ignore').copy()
    print(f"Fitur sudah diperbarui menjadi : {X.columns.tolist()}\n")
    
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = X[col].fillna('Unknown')
    
    X = pd.get_dummies(X, columns=X.select_dtypes(include=['object']).columns, 
                       drop_first=False, dtype='int64')
    
    print(f"[OK] Preprocessing selesai")
    print(f"     Fitur setelah OneHotEncoding: {X.shape[1]}")
    print(f"     Total data: {len(X)}")
    
    le_target = LabelEncoder()
    y_encoded = le_target.fit_transform(y)
    print(f"     Kelas: {list(le_target.classes_)}\n")
    
    return X.values, y_encoded, le_target



def main():
    print("\n" + "="*70)
    print("KNN MANHATTAN DISTANCE (sklearn)")
    print("Analisis Kepuasan Pelanggan Penerbangan")
    print("="*70 + "\n")
    
    start_time = time.time()
    
    df = load_data(DATASET_PATH)
    X, y, le_target = preprocess_data(df)
    
    print(f"Train-test split (80-20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"[OK] Train: {len(X_train)}, Test: {len(X_test)}")
    
    if DEMO_SIZE is not None and DEMO_SIZE < len(X_test):
        print(f"[INFO] Demo mode: {DEMO_SIZE} samples")
        idx = np.random.choice(len(X_test), DEMO_SIZE, replace=False)
        X_test, y_test = X_test[idx], y_test[idx]
        print(f"[OK] Test (demo): {len(X_test)}\n")
    else:
        print()
    
    print(f"Training KNN (k={K}, metric='manhattan')...")
    train_start = time.time()
    knn = KNeighborsClassifier(n_neighbors=K, metric='manhattan', n_jobs=-1)
    knn.fit(X_train, y_train)
    train_time = time.time() - train_start
    print(f"[OK] Training: {train_time:.3f}s\n")
    
    print(f"Prediksi...")
    pred_start = time.time()
    y_pred = knn.predict(X_test)
    pred_time = time.time() - pred_start
    print(f"[OK] Prediksi: {pred_time:.3f}s\n")
    
    acc = accuracy_score(y_test, y_pred)
    
    print("="*70)
    print("HASIL EVALUASI - KNN MANHATTAN DISTANCE")
    print("="*70)
    print(f"Akurasi: {acc:.4f} ({acc*100:.2f}%)")
    print(f"Benar: {np.sum(y_test == y_pred)}/{len(y_test)}")
    print(f"Salah: {np.sum(y_test != y_pred)}/{len(y_test)}")
    
    print(f"\nPer Kelas:")
    for i, cls in enumerate(le_target.classes_):
        mask = y_test == i
        if np.sum(mask) > 0:
            cls_acc = np.mean(y_pred[mask] == y_test[mask])
            correct = np.sum(y_pred[mask] == y_test[mask])
            print(f"  [{i}] {cls:<25}: {cls_acc:.4f} ({correct}/{np.sum(mask)})")
    
    print("\n" + "="*70)
    print("CLASSIFICATION REPORT")
    print("="*70)
    print(f"{'Kelas':<30} {'Precision':<15} {'Recall':<15} {'F1-Score':<15} {'Support':<10}")
    print("-"*70)
    report = classification_report(y_test, y_pred, target_names=le_target.classes_, output_dict=True)
    for cls in le_target.classes_:
        p = report[cls]['precision']
        r = report[cls]['recall']
        f = report[cls]['f1-score']
        s = report[cls]['support']
        print(f"{cls:<30} {p:<15.4f} {r:<15.4f} {f:<15.4f} {int(s):<10}")
    print("-"*70)
    print(f"{'Macro Avg':<30} {report['macro avg']['precision']:<15.4f} {report['macro avg']['recall']:<15.4f} {report['macro avg']['f1-score']:<15.4f} {int(report['macro avg']['support']):<10}")
    print(f"{'Weighted Avg':<30} {report['weighted avg']['precision']:<15.4f} {report['weighted avg']['recall']:<15.4f} {report['weighted avg']['f1-score']:<15.4f} {int(report['weighted avg']['support']):<10}")
    print("="*70)
    print("CONFUSION MATRIX")
    print("="*70)
    cm = confusion_matrix(y_test, y_pred)
    print("\n         Predicted")
    print("Actual   ", le_target.classes_)
    for i, cls in enumerate(le_target.classes_):
        print(f"{cls:<10} {cm[i]}")
    
    print("\n" + "="*70)
    print("PERFORMANCE")
    print("="*70)
    print(f"Training:  {train_time:.3f}s")
    print(f"Prediction: {pred_time:.3f}s ({(pred_time/len(X_test))*1000:.2f}ms/sample)")
    print(f"Total:      {time.time()-start_time:.3f}s")
    
    print("\n" + "="*70)
    print("MANHATTAN DISTANCE: d(p,q) = sum(|p_i - q_i|)")
    print("="*70)
    print("Implementasi: scikit-learn (optimized C/Cython)")
    print("Fitur: n_jobs=-1 (parallel processing)")
    print("="*70 + "\n")
    
    return {
        'model': knn,
        'accuracy': acc,
        'le_target': le_target,
        'cm': cm
    }

if __name__ == "__main__":
    main()