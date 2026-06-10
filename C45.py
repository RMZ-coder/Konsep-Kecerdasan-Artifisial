import os
import sys
import pandas as pd
import numpy as np
from collections import Counter
import warnings

warnings.filterwarnings('ignore')

try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except:
    SCRIPT_DIR = os.getcwd()

DATASET_PATH = os.path.join(SCRIPT_DIR, 'datasetkepuasan', 'train.csv')

RANDOM_STATE = 42 
TEST_SIZE = 0.2   

def train_test_split_manual(X, y, test_size=0.2, random_state=42, stratify=None):
    """Membagi dataset menjadi data latih dan data uji secara manual.
    Mendukung pembagian stratifikasi untuk menjaga proporsi kelas target.
    """
    np.random.seed(random_state)
    n = len(y)
    n_test = int(n * test_size)
    
    if stratify is not None:
        train_indices = []
        test_indices = []
        if isinstance(stratify, (pd.Series, pd.DataFrame)):
            stratify = stratify.values

        for class_val in np.unique(stratify):
            class_indices = np.where(stratify == class_val)[0]
            np.random.shuffle(class_indices)
            n_class = len(class_indices)
            n_test_class = int(n_class * test_size)
            test_indices.extend(class_indices[:n_test_class])
            train_indices.extend(class_indices[n_test_class:])
        
        if isinstance(X, pd.DataFrame):
            X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
        else:
            X_train, X_test = X[train_indices], X[test_indices]
        
        if isinstance(y, pd.Series):
            y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]
        else:
            y_train, y_test = y[train_indices], y[test_indices]
        
        return X_train, X_test, y_train, y_test
    else:
        indices = np.random.permutation(n)
        
        if isinstance(X, pd.DataFrame):
            X_train, X_test = X.iloc[indices[n_test:]], X.iloc[indices[:n_test]]
        else:
            X_train, X_test = X[indices[n_test:]], X[indices[:n_test]]
        
        if isinstance(y, pd.Series):
            y_train, y_test = y.iloc[indices[n_test:]], y.iloc[indices[:n_test]]
        else:
            y_train, y_test = y[indices[n_test:]], y[indices[:n_test]]
        
        return X_train, X_test, y_train, y_test

class LabelEncoderManual:
    """Mengubah label kategorikal menjadi angka secara manual.
    Mirip dengan sklearn.preprocessing.LabelEncoder.
    """
    def __init__(self):
        self.classes_ = None 
        self.mapping_ = {}   
    
    def fit(self, y):
        """Mempelajari kelas-kelas unik dari data."""
        self.classes_ = np.unique(y)
        self.mapping_ = {val: idx for idx, val in enumerate(self.classes_)}
        return self
    
    def transform(self, y):
        """Mengubah data kategorikal menjadi angka berdasarkan pemetaan yang dipelajari."""
        return np.array([self.mapping_[val] for val in y])
    
    def fit_transform(self, y):
        """Melakukan fit dan transform secara berurutan."""
        self.fit(y)
        return self.transform(y)
    
    def inverse_transform(self, y):
        """Mengembalikan angka menjadi label kategorikal aslinya."""
        return np.array([self.classes_[idx] for idx in y])

def bin_arrival_delay(df, column='Arrival Delay in Minutes'):
    """Mengelompokkan kolom 'Arrival Delay in Minutes' ke dalam kategori.
    Kategori: 'No Delay', 'Short Delay', 'Medium Delay', 'Long Delay'.
    """
    if column not in df.columns:
        print(f"Kolom '{column}' tidak ditemukan untuk binning.")
        return df

    df[column] = df[column].astype(str).str.replace(',', '.', regex=False)
    
    df[column] = df[column].str.replace(r'\.00\.00', '.00', regex=True)
    
    df[column] = pd.to_numeric(df[column], errors='coerce')
    
    nan_count_before = df[column].isna().sum()
    df = df.dropna(subset=[column]) 
    if nan_count_before > 0:
        print(f"  Dihapus {nan_count_before} baris dengan nilai NaN di kolom '{column}'")

    bins = [-np.inf, 0, 15, 60, np.inf] 
    labels = ['No Delay', 'Short Delay', 'Medium Delay', 'Long Delay']
    df[column] = pd.cut(df[column], bins=bins, labels=labels, right=True)
    print(f"Kolom '{column}' berhasil di-binning menjadi kategori: {labels}")
    return df

def hitung_entropy(y):
    """Menghitung Entropy dari sebuah set data.
    Entropy mengukur tingkat ketidakmurnian atau ketidakteraturan data.
    Rumus: H(S) = - sum(p_i * log2(p_i))
    """
    if len(y) == 0:
        return 0
    
    counts = Counter(y)
    proportions = [count / len(y) for count in counts.values()]
    
    entropy = -sum(p * np.log2(p) for p in proportions if p > 0) # Hindari log(0)
    return entropy

def hitung_gain_ratio(y_parent, y_left, y_right):
    """Menghitung Gain Ratio untuk algoritma C4.5.
    Gain Ratio adalah perbaikan dari Information Gain yang mengatasi bias
    terhadap atribut dengan banyak nilai unik.
    Rumus: GainRatio(A) = Gain(A) / SplitInfo(A)
    """
    n_total = len(y_parent)
    if n_total == 0: return 0

    entropy_parent = hitung_entropy(y_parent)
    
    n_left, n_right = len(y_left), len(y_right)
    
    weighted_entropy = (n_left / n_total) * hitung_entropy(y_left) + \
                       (n_right / n_total) * hitung_entropy(y_right)
    
    info_gain = entropy_parent - weighted_entropy
    
    p_left = n_left / n_total
    p_right = n_right / n_total
    
    split_info = 0
    if p_left > 0: split_info -= p_left * np.log2(p_left)
    if p_right > 0: split_info -= p_right * np.log2(p_right)
    
    if split_info == 0:
        return 0
    
    return info_gain / split_info


class C45DecisionTree:
    """Implementasi Decision Tree C4.5 secara manual.
    Membangun pohon keputusan berdasarkan Gain Ratio.
    """
    def __init__(self, max_depth=5, min_samples_split=2, n_threshold_samples=10):
        self.max_depth = max_depth             
        self.min_samples_split = min_samples_split 
        self.n_threshold_samples = n_threshold_samples 
        self.tree = None                      
        self.feature_types = None              

    def fit(self, X, y):
        """Melatih model Decision Tree C4.5.
        X: Fitur (DataFrame atau array numpy)
        y: Target (Series atau array numpy)
        """
        
        self.feature_types = [X.iloc[:, i].dtype for i in range(X.shape[1])]

        X_arr = np.array(X)
        y_arr = np.array(y)
        self.tree = self._bangun_pohon(X_arr, y_arr, depth=0)
        return self

    def _bangun_pohon(self, X, y, depth):
        """Fungsi rekursif untuk membangun pohon keputusan.
        Mengembalikan node pohon (berupa dictionary) atau label daun.
        """
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y)) 

        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            return self._buat_daun(y)

        best_feature, best_threshold, best_gain_ratio = None, None, -1
        
        for feature_idx in range(n_features):
            current_feature_values = X[:, feature_idx]
            
            if np.issubdtype(self.feature_types[feature_idx], np.number):
                unique_values = np.unique(current_feature_values)
                if len(unique_values) > self.n_threshold_samples:
                    np.random.seed(RANDOM_STATE) 
                    thresholds = np.random.choice(unique_values, self.n_threshold_samples, replace=False)
                else:
                    thresholds = unique_values
            else: 
                thresholds = np.unique(current_feature_values)

            for threshold in thresholds:
                left_idx = np.where(current_feature_values <= threshold)[0]
                right_idx = np.where(current_feature_values > threshold)[0]
                
                if len(left_idx) == 0 or len(right_idx) == 0:
                    continue
                
                gain_ratio = hitung_gain_ratio(y, y[left_idx], y[right_idx])
                
                if gain_ratio > best_gain_ratio:
                    best_gain_ratio = gain_ratio
                    best_feature = feature_idx
                    best_threshold = threshold

        if best_gain_ratio <= 0 or best_feature is None:
            return self._buat_daun(y)

        left_mask = X[:, best_feature] <= best_threshold
        right_mask = X[:, best_feature] > best_threshold
        
        return {
            'feature_idx': best_feature, 
            'threshold': best_threshold, 
            'left': self._bangun_pohon(X[left_mask], y[left_mask], depth + 1), 
            'right': self._bangun_pohon(X[right_mask], y[right_mask], depth + 1) 
        }

    def _buat_daun(self, y):
        """Menentukan label kelas mayoritas untuk node daun.
        Jika node kosong, kembalikan None atau nilai default.
        """
        if len(y) == 0: 
            return None 
        return Counter(y).most_common(1)[0][0]

    def predict(self, X):
        """Melakukan prediksi untuk satu atau banyak data baru.
        X: Data fitur yang akan diprediksi (DataFrame atau array numpy)
        """
        X_arr = np.array(X)
        return np.array([self._telusuri_pohon(x, self.tree) for x in X_arr])

    def _telusuri_pohon(self, x, node):
        """Fungsi rekursif untuk menelusuri pohon keputusan untuk satu sampel data.
        x: Satu sampel data (array numpy)
        node: Node pohon saat ini
        """
        if not isinstance(node, dict):
            return node
        
        if x[node['feature_idx']] <= node['threshold']:
            return self._telusuri_pohon(x, node['left']) 
        else:
            return self._telusuri_pohon(x, node['right']) 
        

def hitung_akurasi(y_true, y_pred):
    """Menghitung akurasi model secara manual.
    Akurasi = (Jumlah prediksi benar) / (Total prediksi)
    """
    return np.sum(y_true == y_pred) / len(y_true)

def confusion_matrix_manual(y_true, y_pred):
    """Menghitung Confusion Matrix secara manual.
    Menyediakan gambaran detail tentang kinerja model untuk setiap kelas.
    """
    classes = np.unique(np.concatenate([y_true, y_pred]))
    cm = np.zeros((len(classes), len(classes)), dtype=int)
    
    for i, true_class in enumerate(classes):
        for j, pred_class in enumerate(classes):
            cm[i, j] = np.sum((y_true == true_class) & (y_pred == pred_class))
    return cm

def hitung_precision_recall_f1(y_true, y_pred):
    """Menghitung Precision, Recall, dan F1 Score untuk setiap kelas secara manual.
    Mengembalikan dictionary dengan metrics per kelas dan rata-rata.
    """
    classes = np.unique(np.concatenate([y_true, y_pred]))
    cm = confusion_matrix_manual(y_true, y_pred)
    
    results = {}
    precision_list = []
    recall_list = []
    f1_list = []
    support_list = []
    
    for idx, class_val in enumerate(classes):
        tp = cm[idx, idx]
        fp = np.sum(cm[:, idx]) - tp
        fn = np.sum(cm[idx, :]) - tp
        tn = np.sum(cm) - tp - fp - fn
        
        support = tp + fn
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        results[class_val] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': support,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'tn': tn
        }
        
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
        support_list.append(support)
    
    total_support = sum(support_list)
    
    macro_precision = np.mean(precision_list)
    macro_recall = np.mean(recall_list)
    macro_f1 = np.mean(f1_list)
    
    weighted_precision = sum(p * s for p, s in zip(precision_list, support_list)) / total_support
    weighted_recall = sum(r * s for r, s in zip(recall_list, support_list)) / total_support
    weighted_f1 = sum(f * s for f, s in zip(f1_list, support_list)) / total_support
    
    results['macro_avg'] = {
        'precision': macro_precision,
        'recall': macro_recall,
        'f1': macro_f1,
        'support': total_support
    }
    
    results['weighted_avg'] = {
        'precision': weighted_precision,
        'recall': weighted_recall,
        'f1': weighted_f1,
        'support': total_support
    }
    
    return results

def print_classification_report(y_true, y_pred, target_names=None):
    """Menampilkan laporan klasifikasi lengkap dengan format yang rapi.
    Mirip dengan sklearn.metrics.classification_report
    """
    metrics = hitung_precision_recall_f1(y_true, y_pred)
    classes = np.unique(np.concatenate([y_true, y_pred]))
    
    if target_names is None:
        target_names = {class_val: f"Class {class_val}" for class_val in classes}
    
    print("\n" + "=" * 80)
    print("CLASSIFICATION REPORT (LAPORAN KLASIFIKASI RINCI)")
    print("=" * 80)
    print(f"{'Kelas':<20} {'Precision':<15} {'Recall':<15} {'F1-Score':<15} {'Support':<10}")
    print("-" * 80)
    
    for class_val in classes:
        class_name = target_names.get(class_val, f"Class {class_val}")
        m = metrics[class_val]
        print(f"{class_name:<20} {m['precision']:<15.4f} {m['recall']:<15.4f} {m['f1']:<15.4f} {m['support']:<10}")
    
    print("-" * 80)
    
    m_macro = metrics['macro_avg']
    print(f"{'Macro Avg':<20} {m_macro['precision']:<15.4f} {m_macro['recall']:<15.4f} {m_macro['f1']:<15.4f} {m_macro['support']:<10}")
    
    m_weighted = metrics['weighted_avg']
    print(f"{'Weighted Avg':<20} {m_weighted['precision']:<15.4f} {m_weighted['recall']:<15.4f} {m_weighted['f1']:<15.4f} {m_weighted['support']:<10}")
    print("=" * 80)

if __name__ == "__main__":
    print("=" * 60)
    print("PROGRAM KLASIFIKASI KEPUASAN PELANGGAN DENGAN C4.5 MANUAL (OPTIMASI)")
    print("=" * 60)

    print("\n" + "=" * 60)
    print("TAHAP 1: MEMBACA DATASET")
    print("=" * 60)
    
    df = None
    try:
        df = pd.read_csv(DATASET_PATH, sep=';')
        print(f"Dataset berhasil dimuat dari '{DATASET_PATH}' dengan separator ';'")
    except FileNotFoundError:
        print(f"Error: File dataset tidak ditemukan di '{DATASET_PATH}'.")
        print("Pastikan file 'train.csv' berada di dalam folder 'datasetkepuasan' ")
        print("yang sejajar dengan script ini, atau sesuaikan DATASET_PATH.")
        sys.exit(1) 
    except Exception as e:
        print(f"Gagal membaca dataset dengan separator ';'. Mencoba separator ','. Error: {e}")
        try:
            df = pd.read_csv(DATASET_PATH, sep=',')
            print(f"Dataset berhasil dimuat dari '{DATASET_PATH}' dengan separator ','")
        except Exception as e_comma:
            print(f"Gagal membaca dataset dengan separator ','. Error: {e_comma}")
            print("Tidak dapat memuat dataset. Harap periksa format file CSV Anda.")
            sys.exit(1)

    print(f"Jumlah data: {df.shape[0]} baris, {df.shape[1]} kolom")
    print("5 baris pertama dataset:")
    print(df.head())

    print("\n" + "=" * 60)
    print("TAHAP 2: INFORMASI DATASET & PREPROCESSING AWAL")
    print("=" * 60)

    print(f"\nJumlah data awal: {len(df)}")
    print(f"Jumlah fitur: {df.shape[1]}")
    print(f"\nInformasi kolom dan tipe data:")
    df.info()

    print(f"\nJumlah data yang hilang (NaN) per kolom:")
    print(df.isnull().sum())

    df_clean = df.copy()
    if 'id' in df_clean.columns:
        df_clean = df_clean.drop('id', axis=1)
        print("Kolom 'id' dihapus.")
    
    initial_rows = len(df_clean)
    df_clean = df_clean.dropna()
    rows_after_na = len(df_clean)
    if initial_rows > rows_after_na:
        print(f"Menghapus {initial_rows - rows_after_na} baris dengan nilai yang hilang.")
    print(f"Jumlah data setelah pembersihan: {len(df_clean)} baris")

    print("\n" + "=" * 60)
    print("TAHAP 2.5: BINNING KOLOM 'Arrival Delay in Minutes'")
    print("=" * 60)
    df_clean = bin_arrival_delay(df_clean, column='Arrival Delay in Minutes')

    TARGET_COLUMN = 'satisfaction' 
    if TARGET_COLUMN not in df_clean.columns:
        print(f"Error: Kolom target '{TARGET_COLUMN}' tidak ditemukan di dataset.")
        sys.exit(1)

    X = df_clean.drop(TARGET_COLUMN, axis=1)
    y = df_clean[TARGET_COLUMN]
    
    print(f"\nDistribusi kelas target ('{TARGET_COLUMN}'):")
    print(y.value_counts())

    print("\n" + "=" * 60)
    print("TAHAP 3: ENCODING DATA KATEGORIKAL")
    print("=" * 60)

    encoders = {} 
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    
    if len(categorical_cols) > 0:
        print(f"Meng-encode kolom kategorikal: {list(categorical_cols)}")
        for col in categorical_cols:
            encoder = LabelEncoderManual()
            X[col] = encoder.fit_transform(X[col])
            encoders[col] = encoder
            print(f"  - Kolom '{col}' di-encode. Kelas: {encoder.classes_.tolist()}")
    else:
        print("Tidak ada kolom kategorikal yang perlu di-encode.")

    print(f"Meng-encode kolom target '{TARGET_COLUMN}'.")
    encoder_target = LabelEncoderManual()
    y_encoded = encoder_target.fit_transform(y)
    y_encoded = pd.Series(y_encoded, index=y.index)
    print(f"  - Kelas target asli: {encoder_target.classes_.tolist()}")
    print(f"  - Kelas target ter-encode: {np.unique(y_encoded).tolist()}")

    print("\n" + "=" * 60)
    print("TAHAP 4: MEMBAGI DATA LATIH DAN UJI")
    print("=" * 60)
    
    X_train, X_test, y_train, y_test = train_test_split_manual(
        X, y_encoded, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_encoded
    )
    
    print(f"Ukuran data latih (X_train, y_train): {len(X_train)} sampel")
    print(f"Ukuran data uji (X_test, y_test): {len(X_test)} sampel")
    print(f"Proporsi data latih: {len(X_train) / len(df_clean):.2%}")
    print(f"Proporsi data uji: {len(X_test) / len(df_clean):.2%}")

    print("\n" + "=" * 60)
    print("TAHAP 5: MELATIH MODEL C4.5")
    print("=" * 60)
    
    model = C45DecisionTree(max_depth=5, min_samples_split=10, n_threshold_samples=20)
    print(f"Model C4.5 diinisialisasi dengan max_depth={model.max_depth}, min_samples_split={model.min_samples_split}, n_threshold_samples={model.n_threshold_samples}.")
    
    model.fit(X_train, y_train)
    print("Model C4.5 berhasil dilatih!")

    print("\n" + "=" * 60)
    print("TAHAP 6: PREDIKSI PADA DATA UJI")
    print("=" * 60)
    
    y_pred = model.predict(X_test)
    print("Prediksi pada data uji selesai.")

    print("\n" + "=" * 60)
    print("TAHAP 7: EVALUASI MODEL")
    print("=" * 60)
    
    y_test_arr = y_test.values if isinstance(y_test, pd.Series) else y_test
    
    accuracy = hitung_akurasi(y_test_arr, y_pred)
    cm = confusion_matrix_manual(y_test_arr, y_pred)
    
    print(f"\nAkurasi Model: {accuracy * 100:.2f}%")
    print(f"\nConfusion Matrix (Baris: Aktual, Kolom: Prediksi):\n{cm}")
    
    target_names = encoder_target.inverse_transform(np.unique(y_test_arr))
    target_names_dict = {i: name for i, name in enumerate(target_names)}
    print_classification_report(y_test_arr, y_pred, target_names=target_names_dict)

    print("\n" + "=" * 60)
    print("TAHAP 8: CONTOH PREDIKSI INDIVIDUAL")
    print("=" * 60)
    
    print(f"\n{'No':<5} {'Aktual':<15} {'Prediksi':<15} {'Status':<10}")
    print("-" * 45)
    
    for i in range(min(10, len(y_test_arr))):
        actual_label = encoder_target.inverse_transform([y_test_arr[i]])[0]
        predicted_label = encoder_target.inverse_transform([y_pred[i]])[0]
        status = "BENAR" if y_test_arr[i] == y_pred[i] else "SALAH"
        print(f"{i+1:<5} {actual_label:<15} {predicted_label:<15} {status:<10}")

    print("\n" + "=" * 60)
    print("TAHAP 9: KESIMPULAN")
    print("=" * 60)
    
    print(f"Akurasi keseluruhan model C4.5 manual adalah: {accuracy * 100:.2f}%")
    
    if accuracy >= 0.85:
        status_kinerja = "SANGAT BAIK - Model menunjukkan kinerja yang luar biasa."
    elif accuracy >= 0.75:
        status_kinerja = "BAIK - Model memiliki kinerja yang solid."
    elif accuracy >= 0.65:
        status_kinerja = "CUKUP - Kinerja model dapat diterima, namun ada ruang untuk peningkatan."
    else:
        status_kinerja = "PERLU DITINGKATKAN - Model memerlukan penyesuaian lebih lanjut atau data tambahan."
    
    print(f"Status Kinerja Model: {status_kinerja}")
    print("\nPROGRAM SELESAI! Semoga kode ini membantu pemahaman Anda tentang C4.5.")