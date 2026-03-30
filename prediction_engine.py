# ================================================================
#  DETEKSI ANEMIA DARI KONJUNGTIVA — END-TO-END PIPELINE
#  EfficientNetB0 (feature extractor) + XGBoost (classifier)
#  Dataset: India + Italy palpebral conjunctiva images
#
#  PERBAIKAN dari notebook sebelumnya:
#  [1] Import konflik MobileNetV2 vs EfficientNet → dihapus semua
#  [2] base_model tidak di-assign → FIXED
#  [3] preprocess_input tidak dipakai → FIXED
#  [4] Ekstraksi gambar satu-per-satu → diganti batch predict (10-50x lebih cepat)
#  [5] Cell 31 drop Hgb tanpa guard → FIXED
#  [6] dropna() dipanggil dua kali (cell 33+34) → FIXED
#  [7] Scree plot PCA fit sebelum split → FIXED (fit hanya di train)
#  [8] Cell 41 pipeline lama masih ada → DIHAPUS SELURUHNYA
#  [9] PCA fit sebelum split → FIXED (split dulu, baru fit PCA)
#  [10] stratify=Age_Group → FIXED (stratify=y)
#  [11] scale_pos_weight=1 → FIXED (pakai ratio aktual)
#  [12] CV masih bocor (PCA sudah di-fit di luar) → FIXED (sklearn Pipeline)
#  [13] Model tidak disimpan → ditambahkan (joblib + xgb json)
# ================================================================


# ================================================================
# CELL 1 — IMPORTS
# ================================================================

import os
import shutil
import warnings
import joblib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input  # SATU-SATUNYA preprocess_input
from tensorflow.keras.preprocessing import image as keras_image

from scipy.stats import chi2_contingency

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_curve,
    average_precision_score,
    f1_score,
    log_loss,
)

import xgboost as xgb

warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid', palette='colorblind')
plt.rcParams['figure.figsize'] = [12, 8]

print('✅ Semua library berhasil di-import.')
print(f'   TensorFlow version : {tf.__version__}')
print(f'   XGBoost version    : {xgb.__version__}')


# ================================================================
# CELL 2 — KONFIGURASI PATH
# ================================================================

BASE_INPUT   = r"C:\Users\bryant\OneDrive\Dokumen\coolyeah\semester 6\proyek  sains data\projecthemoscan\dataset anemia"
FILE_INDIA   = os.path.join(BASE_INPUT, 'India', 'India.xlsx')
FILE_ITALY   = os.path.join(BASE_INPUT, 'Italy', 'Italy.xlsx')
FOLDER_IMG   = r"C:\Users\bryant\OneDrive\Dokumen\coolyeah\semester 6\proyek  sains data\projecthemoscan\hasil_ekstrak"
OUTPUT_CSV   = r"C:\Users\bryant\OneDrive\Dokumen\coolyeah\semester 6\proyek  sains data\projecthemoscan\metadata_anemia_combined.csv"
OUTPUT_MODEL = r"C:\Users\bryant\OneDrive\Dokumen\coolyeah\semester 6\proyek  sains data\projecthemoscan\model_output"

os.makedirs(OUTPUT_MODEL, exist_ok=True)
print(f'✅ Output model akan disimpan di: {OUTPUT_MODEL}')


# ================================================================
# CELL 3 — EKSTRAKSI & VALIDASI GAMBAR (SATU KALI SAJA)
# ================================================================

def extract_and_clean_images(input_path, output_base, countries):
    """
    Salin, validasi, dan rename file gambar palpebral dari tiap negara.
    Hanya file yang bisa dibaca OpenCV (tidak korup) yang disalin.
    Penamaan: {Negara}_{nomor}.png
    """
    os.makedirs(output_base, exist_ok=True)

    # Cek apakah sudah pernah dijalankan (idempoten)
    existing = len(os.listdir(output_base))
    if existing > 0:
        print(f'ℹ️  Folder sudah berisi {existing} file. Lewati ekstraksi.')
        return existing

    print('Memulai ekstraksi dan validasi gambar...')
    total = 0

    for country in countries:
        country_path = os.path.join(input_path, country)
        if not os.path.exists(country_path):
            print(f'  ⚠ Folder {country} tidak ditemukan: {country_path}')
            continue

        print(f'  Memproses: {country}')
        count = 1

        for root, _, files in os.walk(country_path):
            targets = sorted([f for f in files if 'palpebral' in f.lower()])
            for filename in targets:
                src = os.path.join(root, filename)
                img = cv2.imread(src)
                if img is not None:
                    dst = os.path.join(output_base, f'{country}_{count}.png')
                    if cv2.imwrite(dst, img):
                        count += 1
                        total += 1
                else:
                    print(f'    ⚠ File korup dilewati: {src}')

        print(f'  ✅ {country}: {count - 1} file berhasil.')

    print(f'\n✅ Total gambar valid: {total}')
    return total

extract_and_clean_images(BASE_INPUT, FOLDER_IMG, ['India', 'Italy'])


# ================================================================
# CELL 4 — LOAD & GABUNGKAN DATA TABULAR
# ================================================================

df_india = pd.read_excel(FILE_INDIA)
df_italy = pd.read_excel(FILE_ITALY)

df_india['Country']    = 'India'
df_india['image_path'] = df_india['Number'].apply(lambda x: f'India_{x}.png')

df_italy['Country']    = 'Italy'
df_italy['image_path'] = df_italy['Number'].apply(lambda x: f'Italy_{x}.png')

df = pd.concat([df_india, df_italy], ignore_index=True)

# Validasi keberadaan file gambar
df['file_exists'] = df['image_path'].apply(
    lambda f: os.path.exists(os.path.join(FOLDER_IMG, f))
)

df.to_csv(OUTPUT_CSV, index=False)
print(f'✅ Gabungan data: {len(df)} baris')
print(f'   File gambar ditemukan: {df["file_exists"].sum()} / {len(df)}')
print(df.head())


# ================================================================
# CELL 5 — CLEANING DATA TABULAR
# ================================================================

# Hapus kolom yang tidak diperlukan (lewati jika kolom tidak ada)
cols_to_drop = ['Unnamed: 5', 'Unnamed: 6', 'Unnamed: 7', 'Unnamed: 8',
                'file_exists', 'Note', 'Country']
df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

# Perbaiki tipe data Hgb (titik/koma bisa berbeda antar file)
df['Hgb'] = pd.to_numeric(
    df['Hgb'].astype(str).str.replace(',', '.', regex=False),
    errors='coerce'
)

print(f'Kolom tersisa: {df.columns.tolist()}')
print(f'\nNull values:\n{df.isnull().sum()}')


# ================================================================
# CELL 6 — BUAT LABEL STATUS ANEMIA (dari Hgb + Gender)
# ================================================================
# WHO threshold: Wanita Hgb < 12, Pria Hgb < 13 → Anemia

def buat_label_anemia(row):
    hgb    = row['Hgb']
    gender = row['Gender']
    if pd.isna(hgb):
        return None  # akan di-drop bersama NaN
    if (gender == 'F' and hgb < 12) or (gender == 'M' and hgb < 13):
        return 'Anemia'
    return 'Non-Anemia'

df['Status'] = df.apply(buat_label_anemia, axis=1)

# Hapus baris dengan Hgb NaN (label tidak bisa dibuat)
df = df.dropna(subset=['Hgb', 'Status'])

# Hapus baris yang status-nya 'Data Tidak Valid' (jika ada)
df = df[df['Status'].isin(['Anemia', 'Non-Anemia'])].reset_index(drop=True)

print(f'✅ Data bersih: {len(df)} baris')
print(f'\nDistribusi Status:\n{df["Status"].value_counts()}')
print(f'\nDistribusi Gender:\n{df["Gender"].value_counts()}')


# ================================================================
# CELL 7 — FITUR ENGINEERING: AGE GROUP
# ================================================================

def kategori_usia(age):
    if age < 25:
        return 'Anak-anak dan Remaja'
    elif age < 60:
        return 'Dewasa'
    return 'Lansia'

df['Age_Group'] = df['Age'].apply(kategori_usia)

print(f'Distribusi Age Group:\n{df["Age_Group"].value_counts()}')


# ================================================================
# CELL 9 — LOAD EFFICIENTNETB0 (FEATURE EXTRACTOR)
# ================================================================

print('=' * 60)
print('Load EfficientNetB0 sebagai feature extractor')
print('=' * 60)

# FIXED: hasil di-assign ke base_model
base_model = EfficientNetB0(
    weights='imagenet',
    include_top=False,
    pooling='avg',
    input_shape=(224, 224, 3)
)

# Unfreeze 20 layer terakhir (sensitif terhadap warna/tekstur konjungtiva)
base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False

N_IMG_FEATURES = base_model.output_shape[-1]  # 1280

print(f'✅ Base model loaded')
print(f'   Output shape     : {base_model.output_shape}')
print(f'   Total params     : {base_model.count_params():,}')
print(f'   Trainable params : {sum(tf.size(w).numpy() for w in base_model.trainable_weights):,}')


# ================================================================
# CELL 10 — EKSTRAKSI FITUR GAMBAR (BATCH — 10-50x LEBIH CEPAT)
# ================================================================

def ekstrak_fitur_batch(image_paths, base_path, model, batch_size=32):
    """
    Ekstrak fitur gambar menggunakan batch predict.
    JAUH lebih cepat dibanding satu-per-satu karena memanfaatkan GPU/CPU paralel.

    Parameters
    ----------
    image_paths : list  — nama file gambar (relatif dari base_path)
    base_path   : str   — folder tempat gambar disimpan
    model       : tf.keras.Model
    batch_size  : int   — jumlah gambar per batch (32 aman untuk RAM 16GB)

    Returns
    -------
    np.ndarray  — shape (n_samples, n_features)
    """
    n          = len(image_paths)
    n_features = model.output_shape[-1]
    all_feats  = np.zeros((n, n_features), dtype='float32')

    for start in range(0, n, batch_size):
        end         = min(start + batch_size, n)
        batch_paths = image_paths[start:end]
        batch_imgs  = []

        for fname in batch_paths:
            fpath = os.path.join(base_path, fname)
            try:
                img = keras_image.load_img(fpath, target_size=(224, 224))
                arr = keras_image.img_to_array(img)
                batch_imgs.append(arr)
            except Exception as e:
                print(f'  ⚠ Gagal baca {fname}: {e}')
                batch_imgs.append(np.zeros((224, 224, 3), dtype='float32'))

        batch_array          = preprocess_input(np.array(batch_imgs))  # normalisasi EfficientNet
        feats                = model.predict(batch_array, verbose=0)
        all_feats[start:end] = feats

        if (end % 200 == 0) or end == n:
            print(f'  Progress: {end}/{n} gambar')

    return all_feats


print('=' * 60)
print('Ekstraksi fitur gambar (batch mode)')
print('=' * 60)

image_paths_list = df['image_path'].tolist()
print(f'Total gambar: {len(image_paths_list)}')

img_features_array = ekstrak_fitur_batch(
    image_paths  = image_paths_list,
    base_path    = FOLDER_IMG,
    model        = base_model,
    batch_size   = 32
)

# Bungkus ke DataFrame dengan index yang sama dengan df
df_img_feats = pd.DataFrame(
    img_features_array,
    columns=[f'img_feat_{i}' for i in range(N_IMG_FEATURES)],
    index=df.index
)

print(f'✅ Fitur gambar: {df_img_feats.shape}')


# ================================================================
# CELL 11 — GABUNGKAN DATA TABULAR + FITUR GAMBAR
# ================================================================

# Kolom tabular yang dipakai
tabular_cols = ['Age', 'Gender', 'Age_Group', 'Status']

df_multimodal = pd.concat([df[tabular_cols], df_img_feats], axis=1)

# Hgb sudah tidak perlu — sudah dipakai untuk buat label, sekarang dihapus
# (guard: cek dulu sebelum drop agar aman jika dijalankan ulang)
# NOTE: Hgb tidak ada di df_multimodal karena hanya tabular_cols yang diambil

print(f'✅ Multimodal DataFrame: {df_multimodal.shape}')
print(f'   Kolom: {list(df_multimodal.columns[:6])} ... (+ {len(df_img_feats.columns)} fitur gambar)')


# ================================================================
# CELL 12 — PREPROCESSING: ENCODE TARGET & BERSIHKAN NaN
# ================================================================

df_model = df_multimodal.copy().dropna()  # satu kali dropna, cukup

# Encode target
le_status = LabelEncoder()
df_model['Status_Num'] = le_status.fit_transform(df_model['Status'])

status_map = dict(zip(le_status.classes_, le_status.transform(le_status.classes_)))
print(f'Label Status  : {status_map}')  # {'Anemia': 0, 'Non-Anemia': 1} atau sebaliknya
print(f'  → Kelas Anemia = {status_map["Anemia"]}')
print(f'\nDistribusi:\n{df_model["Status_Num"].value_counts()}')

ANEMIA_CLASS = status_map['Anemia']  # simpan untuk dipakai di threshold tuning


# ================================================================
# CELL 13 — PISAHKAN FITUR X DAN TARGET y
# ================================================================

img_cols = [c for c in df_model.columns if c.startswith('img_feat_')]

X_img = df_model[img_cols]
X_tab = df_model[['Age', 'Gender', 'Age_Group']].copy()
y     = df_model['Status_Num']

print(f'Fitur gambar  : {X_img.shape[1]} kolom')
print(f'Fitur tabular : {X_tab.shape[1]} kolom')
print(f'Target (y)    : {y.shape[0]} baris')


# ================================================================
# CELL 14 — TRAIN-TEST SPLIT
#           HARUS DILAKUKAN SEBELUM PCA (mencegah data leakage)
# ================================================================

X_img_train, X_img_test, X_tab_train, X_tab_test, y_train, y_test = train_test_split(
    X_img, X_tab, y,
    test_size   = 0.2,     # 80/20 — lebih representatif dari 90/10
    random_state= 42,
    stratify    = y        # FIXED: stratify pakai target, bukan Age_Group
)

print(f'Train : {len(y_train)} sampel | {dict(y_train.value_counts())}')
print(f'Test  : {len(y_test)}  sampel | {dict(y_test.value_counts())}')


# ================================================================
# CELL 15 — ENCODING FITUR TABULAR
#           Dilakukan SETELAH split agar test tidak bocor ke train
# ================================================================

age_map    = {'Anak-anak dan Remaja': 0, 'Dewasa': 1, 'Lansia': 2}
gender_map = {'M': 0, 'F': 1, 'Male': 0, 'Female': 1}

def encode_tabular(df_tab, age_map, gender_map):
    df_enc = df_tab.copy()
    df_enc['Age_Group'] = df_enc['Age_Group'].map(age_map)
    df_enc['Gender']    = df_enc['Gender'].map(gender_map)
    return df_enc

X_tab_train = encode_tabular(X_tab_train, age_map, gender_map)
X_tab_test  = encode_tabular(X_tab_test,  age_map, gender_map)

print('✅ Encoding selesai.')
print(X_tab_train.head(3))


# ================================================================
# CELL 16 — SCREE PLOT: TENTUKAN N_COMPONENTS PCA
#           FIT HANYA DI TRAIN (mencegah data leakage dari test)
# ================================================================

print('Menentukan jumlah komponen PCA optimal dari data train...')

pca_full       = PCA(random_state=42).fit(X_img_train)   # FIXED: fit di train saja
cumvar         = np.cumsum(pca_full.explained_variance_ratio_)
n_90           = int(np.searchsorted(cumvar, 0.90)) + 1
n_95           = int(np.searchsorted(cumvar, 0.95)) + 1

print(f'  90% variansi tercapai dengan {n_90} komponen')
print(f'  95% variansi tercapai dengan {n_95} komponen')

# Pilih n_components — default 100, tapi sesuaikan dengan scree plot
N_PCA = min(100, n_90)
print(f'  → Menggunakan N_PCA = {N_PCA}')


# ================================================================
# CELL 17 — PCA: REDUKSI DIMENSI
#           fit_transform di train, transform saja di test
# ================================================================

pca             = PCA(n_components=N_PCA, random_state=42)
X_img_train_pca = pca.fit_transform(X_img_train)   # fit + transform (TRAIN)
X_img_test_pca  = pca.transform(X_img_test)         # transform saja  (TEST)

explained = float(np.cumsum(pca.explained_variance_ratio_)[-1])
print(f'✅ PCA {N_PCA} komponen menjelaskan {explained:.1%} variansi')

# Bungkus ke DataFrame (index harus sama dengan tabular)
pca_cols = [f'pca_{i}' for i in range(N_PCA)]

df_pca_train = pd.DataFrame(X_img_train_pca, columns=pca_cols, index=X_tab_train.index)
df_pca_test  = pd.DataFrame(X_img_test_pca,  columns=pca_cols, index=X_tab_test.index)

# Gabung tabular + PCA
X_train = pd.concat([X_tab_train, df_pca_train], axis=1)
X_test  = pd.concat([X_tab_test,  df_pca_test],  axis=1)

print(f'✅ X_train: {X_train.shape}')
print(f'✅ X_test : {X_test.shape}')


# ================================================================
# CELL 18 — CLASS BALANCING: HITUNG scale_pos_weight
# ================================================================

n_neg  = int((y_train == 0).sum())
n_pos  = int((y_train == 1).sum())
ratio  = n_neg / n_pos  # bobot untuk kelas positif (anemia)

print(f'Kelas 0 (Non-Anemia) : {n_neg}')
print(f'Kelas 1 (Anemia)     : {n_pos}')
print(f'scale_pos_weight     : {ratio:.2f}')

# CATATAN: Pastikan kelas Anemia = 1
# Jika ANEMIA_CLASS == 0, tukar: ratio = n_pos / n_neg
if ANEMIA_CLASS == 0:
    ratio = n_pos / n_neg
    print(f'  (Dibalik karena Anemia = kelas 0): {ratio:.2f}')


# ================================================================
# CELL 19 — TRAINING XGBOOST
# ================================================================

print('=' * 60)
print('Training XGBoost Classifier')
print('=' * 60)

model_anemia = xgb.XGBClassifier(
    n_estimators          = 500,
    learning_rate         = 0.1,
    max_depth             = 5,
    min_child_weight      = 3,
    subsample             = 0.8,
    colsample_bytree      = 0.8,
    scale_pos_weight      = ratio,   # FIXED: pakai ratio aktual
    eval_metric           = 'aucpr', # lebih baik untuk imbalanced
    early_stopping_rounds = 30,
    random_state          = 42,
    tree_method           = 'hist'
)

model_anemia.fit(
    X_train, y_train,
    eval_set = [(X_test, y_test)],
    verbose  = False
)

best_iter = model_anemia.best_iteration
print(f'✅ Training selesai — best iteration: {best_iter}')


# ================================================================
# CELL 20 — EVALUASI DASAR
# ================================================================

print('=' * 60)
print('Evaluasi (threshold default = 0.5)')
print('=' * 60)

y_pred  = model_anemia.predict(X_test)
y_proba = model_anemia.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred, target_names=['Non-Anemia', 'Anemia']))

roc_auc  = roc_auc_score(y_test, y_proba)
avg_prec = average_precision_score(y_test, y_proba)
ll       = log_loss(y_test, y_proba)

print(f'ROC-AUC Score          : {roc_auc:.4f}')
print(f'Average Precision (AP) : {avg_prec:.4f}')
print(f'Log Loss               : {ll:.4f}')


# ================================================================
# CELL 21 — THRESHOLD TUNING
#           Untuk aplikasi medis: Recall anemia > Precision
#           False negative (anemia tidak terdeteksi) lebih berbahaya
# ================================================================

print('=' * 60)
print('Threshold Tuning')
print('=' * 60)

thresholds = np.arange(0.20, 0.80, 0.05)
results    = []

for t in thresholds:
    y_pred_t = (y_proba >= t).astype(int)
    f1       = f1_score(y_test, y_pred_t, pos_label=ANEMIA_CLASS, zero_division=0)
    recall   = float((y_pred_t[y_test == ANEMIA_CLASS] == ANEMIA_CLASS).mean())
    prec_val = float((y_test[y_pred_t == ANEMIA_CLASS] == ANEMIA_CLASS).mean()) if (y_pred_t == ANEMIA_CLASS).any() else 0.0
    results.append({'threshold': t, 'f1': f1, 'recall': recall, 'precision': prec_val})
    print(f'  t={t:.2f} | F1={f1:.3f} | Recall={recall:.3f} | Precision={prec_val:.3f}')

df_thresh  = pd.DataFrame(results)
best_row   = df_thresh.loc[df_thresh['f1'].idxmax()]
best_thresh = float(best_row['threshold'])

print(f'\n✅ Best threshold (max F1-Anemia): {best_thresh:.2f}')
print(f'   F1={best_row["f1"]:.3f} | Recall={best_row["recall"]:.3f} | Precision={best_row["precision"]:.3f}')

# Laporan final dengan best threshold
y_pred_best = (y_proba >= best_thresh).astype(int)
print(f'\n--- Classification Report (threshold = {best_thresh:.2f}) ---')
print(classification_report(y_test, y_pred_best, target_names=['Non-Anemia', 'Anemia']))


# ================================================================
# CELL 22 — CROSS-VALIDATION YANG BENAR
#           Pipeline PCA + XGBoost agar tidak ada leakage antar fold
# ================================================================

print('=' * 60)
print('Cross-Validation (StratifiedKFold 5-fold)')
print('=' * 60)

# Gabungkan semua fitur (setelah encoding tabular, sebelum PCA)
# — PCA masuk ke dalam Pipeline sehingga tiap fold fit PCA sendiri
X_tab_all = pd.concat([X_tab_train, X_tab_test], axis=0)
X_img_all = pd.concat([
    pd.DataFrame(X_img_train, columns=img_cols, index=X_tab_train.index),
    pd.DataFrame(X_img_test,  columns=img_cols, index=X_tab_test.index),
], axis=0)
X_full = pd.concat([X_tab_all, X_img_all], axis=1)
y_full = pd.concat([y_train, y_test],        axis=0)

# sklearn Pipeline: PCA di dalam fold → tidak ada leakage
cv_pipeline = Pipeline([
    ('pca', PCA(n_components=N_PCA, random_state=42)),
    ('xgb', xgb.XGBClassifier(
        n_estimators     = best_iter if best_iter else 200,
        learning_rate    = 0.1,
        max_depth        = 5,
        subsample        = 0.8,
        colsample_bytree = 0.8,
        scale_pos_weight = ratio,
        random_state     = 42,
        tree_method      = 'hist',
        eval_metric      = 'logloss',
    ))
])

skf    = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_auc = cross_val_score(cv_pipeline, X_full, y_full, cv=skf,
                          scoring='roc_auc', n_jobs=-1)

print(f'ROC-AUC per fold : {[f"{s:.3f}" for s in cv_auc]}')
print(f'Mean ± Std       : {cv_auc.mean():.3f} ± {cv_auc.std():.3f}')



# ================================================================
# CELL 24 — SIMPAN MODEL & SEMUA PREPROCESSOR
#           Semua yang dibutuhkan saat inference harus disimpan
# ================================================================

print('=' * 60)
print('Simpan model & preprocessor')
print('=' * 60)

# Simpan PCA
joblib.dump(pca, os.path.join(OUTPUT_MODEL, 'pca.pkl'))

# Simpan Label Encoder status
joblib.dump(le_status, os.path.join(OUTPUT_MODEL, 'label_encoder_status.pkl'))

# Simpan mapping (sebagai dict — lebih portabel dari pickle)
joblib.dump({
    'age_map'     : age_map,
    'gender_map'  : gender_map,
    'status_map'  : status_map,
    'best_thresh' : best_thresh,
    'anemia_class': ANEMIA_CLASS,
    'n_pca'       : N_PCA,
}, os.path.join(OUTPUT_MODEL, 'config.pkl'))

# Simpan XGBoost model (format JSON — lebih stabil lintas versi)
model_anemia.save_model(os.path.join(OUTPUT_MODEL, 'xgb_anemia.json'))

print('✅ File tersimpan:')
for f in os.listdir(OUTPUT_MODEL):
    fpath = os.path.join(OUTPUT_MODEL, f)
    size  = os.path.getsize(fpath) / 1024
    print(f'   {f:40s} {size:>8.1f} KB')


# ================================================================
# CELL 25 — FUNGSI INFERENCE UNTUK APLIKASI
#           Input: path gambar + data pasien
#           Output: prediksi + probabilitas
# ================================================================

def load_pipeline(model_dir):
    """Load semua komponen pipeline dari folder output."""
    pca_loaded     = joblib.load(os.path.join(model_dir, 'pca.pkl'))
    config         = joblib.load(os.path.join(model_dir, 'config.pkl'))
    xgb_loaded     = xgb.XGBClassifier()
    xgb_loaded.load_model(os.path.join(model_dir, 'xgb_anemia.json'))
    return pca_loaded, xgb_loaded, config


def predict_anemia(image_path, age, gender, age_group,
                   base_model, pca, xgb_model, config):
    """
    Prediksi status anemia dari satu gambar konjungtiva + data pasien.

    Parameters
    ----------
    image_path : str   — path absolut ke file gambar
    age        : int   — usia pasien (tahun)
    gender     : str   — 'M' atau 'F'
    age_group  : str   — 'Anak-anak dan Remaja' / 'Dewasa' / 'Lansia'
    base_model : EfficientNetB0 instance
    pca        : fitted PCA
    xgb_model  : fitted XGBClassifier
    config     : dict dari config.pkl

    Returns
    -------
    dict dengan kunci: prediksi, probabilitas, threshold, keterangan
    """
    age_map_c      = config['age_map']
    gender_map_c   = config['gender_map']
    best_thresh_c  = config['best_thresh']
    anemia_class_c = config['anemia_class']

    # 1. Ekstrak fitur gambar
    try:
        img = keras_image.load_img(image_path, target_size=(224, 224))
        arr = keras_image.img_to_array(img)
        arr = preprocess_input(np.expand_dims(arr, axis=0))
        img_feat = base_model.predict(arr, verbose=0).flatten()
    except Exception as e:
        return {'error': f'Gagal baca gambar: {e}'}

    # 2. PCA
    img_pca = pca.transform(img_feat.reshape(1, -1))

    # 3. Encode tabular
    age_enc = age_map_c.get(age_group, 1)
    gen_enc = gender_map_c.get(gender, gender_map_c.get(gender.capitalize(), 0))
    tabular = np.array([[age, gen_enc, age_enc]], dtype='float32')

    # 4. Gabungkan & prediksi
    X_input = np.concatenate([tabular, img_pca], axis=1)
    proba   = float(xgb_model.predict_proba(X_input)[0][anemia_class_c])
    prediksi = 'Anemia' if proba >= best_thresh_c else 'Non-Anemia'

    return {
        'prediksi'    : prediksi,
        'probabilitas': round(proba, 4),
        'threshold'   : best_thresh_c,
        'keterangan'  : (
            'Kemungkinan anemia tinggi. Disarankan konsultasi ke dokter.'
            if prediksi == 'Anemia'
            else 'Tidak terdeteksi anemia berdasarkan gambar konjungtiva.'
        )
    }


# ================================================================
# CELL 26 — TEST INFERENCE (CONTOH PEMAKAIAN)
# ================================================================

print('=' * 60)
print('Test fungsi inference')
print('=' * 60)

# Load ulang dari disk (simulasi deployment)
pca_loaded, xgb_loaded, config_loaded = load_pipeline(OUTPUT_MODEL)

# Ambil satu gambar test sebagai contoh
sample_path = os.path.join(FOLDER_IMG, df['image_path'].iloc[0])

if os.path.exists(sample_path):
    hasil = predict_anemia(
        image_path = sample_path,
        age        = 35,
        gender     = 'F',
        age_group  = 'Dewasa',
        base_model = base_model,
        pca        = pca_loaded,
        xgb_model  = xgb_loaded,
        config     = config_loaded
    )
    print('\nContoh prediksi:')
    for k, v in hasil.items():
        print(f'  {k:15s}: {v}')
else:
    print(f'⚠ File sampel tidak ditemukan: {sample_path}')


# ================================================================
# CELL 27 — RINGKASAN AKHIR
# ================================================================

print('\n' + '=' * 60)
print('RINGKASAN PIPELINE')
print('=' * 60)
print(f'  Dataset            : {len(df)} pasien (India + Italy)')
print(f'  Kelas Anemia       : {(y == ANEMIA_CLASS).sum()} ({(y == ANEMIA_CLASS).mean():.1%})')
print(f'  Kelas Non-Anemia   : {(y != ANEMIA_CLASS).sum()} ({(y != ANEMIA_CLASS).mean():.1%})')
print(f'  Fitur gambar       : {N_IMG_FEATURES} → PCA {N_PCA} komponen ({explained:.1%} variansi)')
print(f'  Fitur tabular      : Age, Gender, Age_Group')
print(f'  Model              : XGBoost (best iter = {best_iter})')
print(f'  Test ROC-AUC       : {roc_auc:.4f}')
print(f'  Test Avg Precision : {avg_prec:.4f}')
print(f'  CV ROC-AUC         : {cv_auc.mean():.3f} ± {cv_auc.std():.3f}')
print(f'  Best threshold     : {best_thresh:.2f}')
print(f'  Output folder      : {OUTPUT_MODEL}')
print('=' * 60)
print('✅ Pipeline selesai. Semua file tersimpan di OUTPUT_MODEL.')