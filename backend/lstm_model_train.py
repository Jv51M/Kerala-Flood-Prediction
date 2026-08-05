"""
lstm_model_train.py  —  retrained with anti-overfitting measures

Key changes vs previous version
────────────────────────────────
1. Data augmentation  – Gaussian noise injection + temporal shift on minority class
2. Smaller LSTM       – 64 units instead of 128 (right-sized for ~1k rows)
3. L2 regularisation  – applied to LSTM kernel and every Dense layer
4. Heavier Dropout    – 0.4 after LSTM, 0.3 after first Dense
5. Larger batch size  – 64 (smoother gradient updates, less memorisation)
6. Stratified split   – preserves class ratio in train/val sets
7. Class weights      – compensates for label imbalance automatically
8. Tighter EarlyStopping – patience=7, min_delta=1e-4
9. Saves scalers + prints a clear overfitting diagnostic at end
"""

import pandas as pd
import numpy as np
import os
import joblib
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, Concatenate, Input, BatchNormalization
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

# ── Paths ─────────────────────────────────────────────────────────────────────
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path   = os.path.join(current_dir, 'data', 'lstm_training_data.csv')
model_dir   = os.path.join(current_dir, 'models')
os.makedirs(model_dir, exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────
df         = pd.read_csv(data_path)
rain_cols  = sorted([col for col in df.columns if col.startswith('rain_')])
X_temporal = df[rain_cols].values                          # (n, 15)
X_static   = df[['elevation', 'slope', 'dist_to_river']].values
y          = df['label'].values

print(f"Dataset: {len(df)} rows | class balance: "
      f"{y.sum()} flood / {(y==0).sum()} no-flood")

# ── Data augmentation ─────────────────────────────────────────────────────────
def augment(X_temp, X_stat, y_arr, n_aug=3, noise_std=0.05):
    """
    For each minority-class sample generate `n_aug` augmented copies:
      • Add Gaussian noise to the rainfall sequence (σ = noise_std × feature std)
      • Random temporal shift (±1 day, wraparound) to break temporal memorisation
    """
    flood_idx = np.where(y_arr == 1)[0]
    rain_std  = X_temp.std(axis=0) + 1e-8

    aug_temp, aug_stat, aug_y = [], [], []
    for _ in range(n_aug):
        for i in flood_idx:
            seq = X_temp[i].copy()
            # Gaussian noise
            seq += np.random.normal(0, noise_std * rain_std)
            seq  = np.clip(seq, 0, None)
            # Random temporal shift ±1
            shift = np.random.choice([-1, 0, 1])
            seq   = np.roll(seq, shift)
            aug_temp.append(seq)
            aug_stat.append(X_stat[i])
            aug_y.append(1)

    if aug_temp:
        X_temp = np.vstack([X_temp, np.array(aug_temp)])
        X_stat = np.vstack([X_stat, np.array(aug_stat)])
        y_arr  = np.concatenate([y_arr, np.array(aug_y)])

    # Shuffle
    perm   = np.random.permutation(len(y_arr))
    return X_temp[perm], X_stat[perm], y_arr[perm]

np.random.seed(42)
X_temporal, X_static, y = augment(X_temporal, X_static, y, n_aug=3)
print(f"After augmentation: {len(y)} rows | "
      f"flood: {y.sum()} / no-flood: {(y==0).sum()}")

# ── Scale ─────────────────────────────────────────────────────────────────────
scaler_temp   = StandardScaler()
X_temporal_scaled = scaler_temp.fit_transform(
    X_temporal.reshape(-1, 1)
).reshape(-1, 15, 1)

scaler_static = StandardScaler()
X_static_scaled = scaler_static.fit_transform(X_static)

# Save scalers
joblib.dump(scaler_temp,   os.path.join(model_dir, 'scaler_temporal.pkl'))
joblib.dump(scaler_static, os.path.join(model_dir, 'scaler_static.pkl'))
print("Scalers saved.")

# ── Stratified train/val split ────────────────────────────────────────────────
(X_temp_train, X_temp_val,
 X_stat_train, X_stat_val,
 y_train,      y_val) = train_test_split(
    X_temporal_scaled, X_static_scaled, y,
    test_size=0.2,
    random_state=42,
    stratify=y,          # preserves class ratio in both splits
)

# ── Class weights ─────────────────────────────────────────────────────────────
classes = np.unique(y_train)
weights = compute_class_weight('balanced', classes=classes, y=y_train)
class_weight = dict(zip(classes.astype(int), weights))
print(f"Class weights: {class_weight}")

# ── Model definition ──────────────────────────────────────────────────────────
REG = l2(1e-4)   # shared L2 regularisation strength

temp_input  = Input(shape=(15, 1), name='temporal_input')
lstm_out    = LSTM(
    64,
    return_sequences=False,
    kernel_regularizer=REG,
    recurrent_regularizer=REG,
    name='lstm_layer',
)(temp_input)
lstm_out    = BatchNormalization()(lstm_out)
lstm_out    = Dropout(0.4)(lstm_out)           # was 0.3

stat_input  = Input(shape=(3,), name='static_input')
stat_out    = Dense(32, activation='relu', kernel_regularizer=REG)(stat_input)
stat_out    = BatchNormalization()(stat_out)
stat_out    = Dropout(0.2)(stat_out)           # new

merged      = Concatenate()([lstm_out, stat_out])
dense_out   = Dense(64, activation='relu', kernel_regularizer=REG)(merged)
dense_out   = Dropout(0.3)(dense_out)          # was 0.2
dense_out   = Dense(32, activation='relu', kernel_regularizer=REG)(dense_out)
dense_out   = Dropout(0.2)(dense_out)          # new extra dense block
output      = Dense(1, activation='sigmoid')(dense_out)

model = Model(inputs=[temp_input, stat_input], outputs=output)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')],
)
model.summary()

# ── Callbacks ─────────────────────────────────────────────────────────────────
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=7,             # was 10 — stops sooner before memorisation
    min_delta=1e-4,         # new — ignores noise improvements
    restore_best_weights=True,
)
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.3,
    patience=4,
    min_lr=1e-5,
    verbose=1,
)

# ── Training ──────────────────────────────────────────────────────────────────
print("\nTraining LSTM (anti-overfitting edition) …")
history = model.fit(
    [X_temp_train, X_stat_train], y_train,
    epochs=150,
    batch_size=64,          # was 16 — larger batches = smoother gradients
    validation_data=([X_temp_val, X_stat_val], y_val),
    class_weight=class_weight,
    callbacks=[early_stop, reduce_lr],
    verbose=1,
)

# ── Evaluation ────────────────────────────────────────────────────────────────
val_loss, val_acc, val_auc = model.evaluate(
    [X_temp_val, X_stat_val], y_val, verbose=0
)
print(f"\n{'='*50}")
print(f"  Val  loss: {val_loss:.4f}  acc: {val_acc:.4f}  AUC: {val_auc:.4f}")

# Overfitting diagnostic
train_loss = min(history.history['loss'])
gap = train_loss - val_loss
if gap > 0.05:
    print(f"  ⚠️  Train/val gap = {gap:.4f} — some overfitting remains")
else:
    print(f"  ✅  Train/val gap = {gap:.4f} — model generalises well")
print(f"  Stopped at epoch {early_stop.stopped_epoch or 'N/A (ran full)'}")
print(f"{'='*50}")

# ── Save ──────────────────────────────────────────────────────────────────────
save_path = os.path.join(model_dir, 'flood_lstm_model.keras')
model.save(save_path)
print(f"\nModel saved to {save_path}")
