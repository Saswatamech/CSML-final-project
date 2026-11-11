import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix, precision_recall_curve
)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# =====================================================================
# HARDWARE DETECTION & DEVICE CONFIGURATION (GPU / CPU)
# =====================================================================
print("--- Checking Available Hardware (GPU / CPU) ---")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"[INFO] GPU detected: {len(gpus)} device(s) available. Running on GPU acceleration.")
        DEVICE_MODE = "GPU"
    except RuntimeError as e:
        print(f"[WARNING] Could not configure GPU memory growth: {e}. Falling back to CPU.")
        DEVICE_MODE = "CPU"
else:
    print("[INFO] No GPU detected. Running calculations on CPU.")
    DEVICE_MODE = "CPU"


# =====================================================================
# PHASE 1: DATA PIPELINE & FEATURE ENGINEERING
# =====================================================================
def run_data_pipeline(file_path):
    print("--- Phase 1: Loading & Preprocessing Data ---")
    df = pd.read_csv(file_path)

    # Sort chronologically by road segment and timestamp to avoid leakage
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(by=['road_segment_id', 'timestamp']).reset_index(drop=True)

    # Temporal Features
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['month'] = df['timestamp'].dt.month
    df['is_rush_hour'] = (df['hour'].isin([7, 8, 9, 16, 17, 18]) & (df['is_weekend'] == 0)).astype(int)

    # Define Target Variable: Congestion in t + 30 to 60 minutes
    future_window_start = 3  # ~30 mins ahead
    future_window_end = 6    # ~60 mins ahead

    df['future_max_occupancy'] = (
        df.groupby('road_segment_id')['occupancy_pct']
        .shift(-future_window_start)
        .rolling(window=3, min_periods=1)
        .max()
    )
    congestion_threshold = 80.0  # 80% occupancy
    df['congestion_target'] = (df['future_max_occupancy'] >= congestion_threshold).astype(int)

    df = df.dropna(subset=['congestion_target']).reset_index(drop=True)

    feature_cols = [
        'avg_speed_kmph', 'density_veh_per_km', 'avg_wait_time_s', 'occupancy_pct',
        'flow_veh_per_hr', 'queue_length_veh', 'avg_accel_ms2', 'signal_state_num',
        'incident_num', 'temp_c', 'visibility_km', 'rain_intensity_mmph',
        'channel_busy_ratio_pct', 'msg_rate_hz', 'avg_comm_delay_ms', 'rssi_dbm',
        'packet_loss_pct', 'speed_density_ratio', 'congestion_pressure',
        'wireless_congestion_intensity', 'throughput_per_queued_vehicle',
        'acceleration_directionality', 'weather_factor', 'hour', 'day_of_week',
        'is_weekend', 'month', 'is_rush_hour'
    ]

    return df, feature_cols


# =====================================================================
# PHASE 2: TIME-BASED SPLITTING (Preventing Data Leakage)
# =====================================================================
def time_based_split(df, feature_cols, target_col='congestion_target'):
    print("--- Phase 2: Time-Based Train/Validation/Test Split ---")

    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    X_train, y_train = train_df[feature_cols], train_df[target_col]
    X_val, y_val = val_df[feature_cols], val_df[target_col]
    X_test, y_test = test_df[feature_cols], test_df[target_col]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.fillna(0))
    X_val_scaled = scaler.transform(X_val.fillna(0))
    X_test_scaled = scaler.transform(X_test.fillna(0))

    return (X_train_scaled, y_train), (X_val_scaled, y_val), (X_test_scaled, y_test), scaler


# =====================================================================
# PHASE 3: MODEL TRAINING & EVALUATION FRAMEWORK
# =====================================================================
def evaluate_model(y_true, y_pred, y_prob):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1 Score': f1_score(y_true, y_pred, zero_division=0),
        'ROC-AUC': roc_auc_score(y_true, y_prob),
        'PR-AUC': average_precision_score(y_true, y_prob),
        'False Positive Rate': fp / (fp + tn) if (fp + tn) > 0 else 0,
        'False Negative Rate': fn / (fn + tp) if (fn + tp) > 0 else 0,
        'Confusion Matrix': cm
    }
    return metrics


def train_baseline_models(train_data, val_data):
    print("--- Phase 3: Training Baseline & Tree Models ---")
    X_train, y_train = train_data
    X_val, y_val = val_data

    # 1. Logistic Regression (CPU)
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    lr_prob = lr.predict_proba(X_val)[:, 1]
    lr_pred = (lr_prob >= 0.5).astype(int)

    # 2. Random Forest (CPU / Multi-core via n_jobs=-1)
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_prob = rf.predict_proba(X_val)[:, 1]
    rf_pred = (rf_prob >= 0.5).astype(int)

    # 3. XGBoost (Supports GPU via device='cuda' if available)
    xgb_device = 'cuda' if DEVICE_MODE == "GPU" else 'cpu'
    xgb = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        random_state=42,
        device=xgb_device,
        tree_method='hist'
    )
    xgb.fit(X_train, y_train)
    xgb_prob = xgb.predict_proba(X_val)[:, 1]
    xgb_pred = (xgb_prob >= 0.5).astype(int)

    models = {
        'Logistic Regression': (lr, evaluate_model(y_val, lr_pred, lr_prob)),
        'Random Forest': (rf, evaluate_model(y_val, rf_pred, rf_prob)),
        'XGBoost': (xgb, evaluate_model(y_val, xgb_pred, xgb_prob))
    }
    return models


def create_sequences(features, labels, time_steps=12):
    Xs, ys = [], []
    for i in range(len(features) - time_steps):
        Xs.append(features[i:(i + time_steps)])
        ys.append(labels.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)


def train_lstm_model(train_data, val_data, feature_count, time_steps=12):
    print(f"--- Training LSTM Sequence Model on {DEVICE_MODE} ---")
    X_train, y_train = train_data
    X_val, y_val = val_data

    X_train_seq, y_train_seq = create_sequences(X_train, y_train, time_steps)
    X_val_seq, y_val_seq = create_sequences(X_val, y_val, time_steps)

    # TensorFlow automatically dispatches operations to GPU if available
    model = Sequential([
        LSTM(64, input_shape=(time_steps, feature_count), return_sequences=False),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train_seq, y_train_seq, epochs=10, batch_size=64, validation_data=(X_val_seq, y_val_seq), verbose=0)

    lstm_prob = model.predict(X_val_seq).flatten()
    lstm_pred = (lstm_prob >= 0.5).astype(int)

    return model, evaluate_model(y_val_seq, lstm_pred, lstm_prob)


# =====================================================================
# EXPERIMENTAL DESIGN & THRESHOLD OPTIMIZATION
# =====================================================================
def evaluate_on_test_set(baseline_models, lstm_model, test_data, feature_count, time_steps=12):
    print("\n" + "="*60)
    print("PHASE 3 EXPERIMENTAL DESIGN: COMPREHENSIVE TEST SET EVALUATION")
    print("="*60)

    X_test_scaled, y_test = test_data
    results = {}

    for model_name, (model_obj, _) in baseline_models.items():
        y_prob = model_obj.predict_proba(X_test_scaled)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        metrics = evaluate_model(y_test, y_pred, y_prob)
        results[model_name] = (y_prob, metrics)

        print(f"\n--- Model: {model_name} (Default Threshold = 0.5) ---")
        print(f"  - Accuracy:           {metrics['Accuracy']:.4f}")
        print(f"  - Precision:          {metrics['Precision']:.4f}")
        print(f"  - Recall:             {metrics['Recall']:.4f}")
        print(f"  - F1 Score:           {metrics['F1 Score']:.4f}")
        print(f"  - ROC-AUC:            {metrics['ROC-AUC']:.4f}")
        print(f"  - PR-AUC:             {metrics['PR-AUC']:.4f}")
        print(f"  - False Positive Rate:{metrics['False Positive Rate']:.4f}")
        print(f"  - False Negative Rate:{metrics['False Negative Rate']:.4f}")
        print(f"  - Confusion Matrix:\n{metrics['Confusion Matrix']}")

    X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test, time_steps)
    if len(X_test_seq) > 0:
        lstm_prob = lstm_model.predict(X_test_seq).flatten()
        lstm_pred = (lstm_prob >= 0.5).astype(int)
        lstm_metrics = evaluate_model(y_test_seq, lstm_pred, lstm_prob)
        results['LSTM'] = (lstm_prob, lstm_metrics)

        print(f"\n--- Model: LSTM (Default Threshold = 0.5) ---")
        print(f"  - Accuracy:           {lstm_metrics['Accuracy']:.4f}")
        print(f"  - Precision:          {lstm_metrics['Precision']:.4f}")
        print(f"  - Recall:             {lstm_metrics['Recall']:.4f}")
        print(f"  - F1 Score:           {lstm_metrics['F1 Score']:.4f}")
        print(f"  - ROC-AUC:            {lstm_metrics['ROC-AUC']:.4f}")
        print(f"  - PR-AUC:             {lstm_metrics['PR-AUC']:.4f}")
        print(f"  - False Positive Rate:{lstm_metrics['False Positive Rate']:.4f}")
        print(f"  - False Negative Rate:{lstm_metrics['False Negative Rate']:.4f}")
        print(f"  - Confusion Matrix:\n{lstm_metrics['Confusion Matrix']}")

    return results, y_test, X_test_seq


def run_cost_optimization_analysis(results, y_test, X_test_seq_labels, c_fn_weight=5.0, c_fp_weight=1.0):
    print("\n" + "="*60)
    print("PHASE 4: COST-SENSITIVE THRESHOLD & TRADE-OFF ANALYSIS")
    print("="*60)
    print(f"Operational Cost Parameters: C_FN (Missing Congestion) = {c_fn_weight}, C_FP (False Alarm) = {c_fp_weight}")

    thresholds = np.arange(0.10, 0.95, 0.05)
    optimization_summary = []

    for model_name, (y_prob, _) in results.items():
        # Correctly align labels: LSTM uses sequence labels (shorter by time_steps), others use standard y_test
        if model_name == 'LSTM':
            target_y = np.array(X_test_seq_labels)
        else:
            target_y = np.array(y_test)

        # Ensure y_prob matches target_y shape
        if len(y_prob) != len(target_y):
            y_prob = y_prob[-len(target_y):]

        best_th = 0.5
        min_cost = float('inf')

        print(f"\nAnalyzing Threshold Sweep for [{model_name}]:")
        print(f"{'Threshold':<12} | {'Precision':<10} | {'Recall':<10} | {'F1 Score':<10} | {'FN':<5} | {'FP':<5} | {'Total Cost':<10}")
        print("-" * 75)

        for th in thresholds:
            y_pred_th = (y_prob >= th).astype(int)
            cm = confusion_matrix(target_y, y_pred_th)
            tn, fp, fn, tp = cm.ravel()

            prec = precision_score(target_y, y_pred_th, zero_division=0)
            rec = recall_score(target_y, y_pred_th, zero_division=0)
            f1 = f1_score(target_y, y_pred_th, zero_division=0)

            total_cost = (c_fn_weight * fn) + (c_fp_weight * fp)

            if total_cost < min_cost:
                min_cost = total_cost
                best_th = th

            print(f"{th:<12.2f} | {prec:<10.4f} | {rec:<10.4f} | {f1:<10.4f} | {fn:<5} | {fp:<5} | {total_cost:<10.1f}")

        print(f"\nOptimal Threshold for {model_name}: {best_th:.2f} (Min Cost: {min_cost:.1f})")
        optimization_summary.append((model_name, best_th, min_cost))

    return optimization_summary


# =====================================================================
# MAIN EXECUTION SCRIPT
# =====================================================================
if __name__ == "__main__":
    file_path = "vanet_traffic_data.csv"

    try:
        df, feature_cols = run_data_pipeline(file_path)
        train_data, val_data, test_data, scaler = time_based_split(df, feature_cols)
        baseline_models = train_baseline_models(train_data, val_data)
        lstm_model, _ = train_lstm_model(train_data, val_data, feature_count=len(feature_cols))

        test_results, y_test, y_test_seq = evaluate_on_test_set(
            baseline_models, lstm_model, test_data, feature_count=len(feature_cols)
        )

        # Step 6: Cost-Sensitive Threshold Optimization & Trade-off Curves
        _, y_test_seq_labels = create_sequences(test_data[0], test_data[1], time_steps=12)

        optimization_summary = run_cost_optimization_analysis(
            test_results, y_test, y_test_seq_labels, c_fn_weight=5.0, c_fp_weight=1.0
        )

        print("\n" + "="*60)
        print("FINAL RECOMMENDATION SUMMARY")
        print("="*60)
        best_overall = min(optimization_summary, key=lambda x: x[2])
        print(f"Best Performing Setup: Model = [{best_overall[0]}], Optimal Threshold = [{best_overall[1]:.2f}]")

    except FileNotFoundError:
        print(f"Note: '{file_path}' was not found in the current directory. Please make sure the dataset is available.")