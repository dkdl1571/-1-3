import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

print("="*50)
print("1. CSV Data Loading Started...")
print("="*50)

data_dir = "data"
classes = ['A', 'B', 'C', 'D', 'E']

X_list = []
y_list = []

for idx, cls in enumerate(classes):
    cls_path = os.path.join(data_dir, cls)
    if not os.path.exists(cls_path):
        cls_path = os.path.join(data_dir, cls.lower())
    
    if not os.path.exists(cls_path):
        print(f"[Warning] Class {cls} folder not found.")
        continue
    
    files = [f for f in os.listdir(cls_path) if f.endswith('.csv')]
    print(f"- Class {cls}: Found {len(files)} CSV files")
    
    for fname in files:
        fpath = os.path.join(cls_path, fname)
        try:
            df = pd.read_csv(fpath, header=None)
            # 숫자로 변환 불가능한 값은 NaN 처리 후 0으로 채움
            df_numeric = df.apply(pd.to_numeric, errors='coerce').fillna(0)
            arr = df_numeric.to_numpy().flatten()
            
            X_list.append(arr)
            y_list.append(idx)
        except Exception as e:
            print(f"  └ Read failed ({fname}): {e}")

if len(X_list) == 0:
    print("\n[Error] No data loaded. Please check folder structure.")
else:
    min_len = min(len(x) for x in X_list)
    X = np.array([x[:min_len] for x in X_list], dtype=np.float32)
    y = np.array(y_list)
    
    print(f"\n[Success] Total samples: {X.shape[0]}, Feature dimension: {X.shape[1]}")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("\n" + "="*50)
    print("2. Training & Evaluating RandomForest Model...")
    print("="*50)
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train_scaled, y_train)
    
    y_pred = clf.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"Final Accuracy: {acc * 100:.2f}%\n")
    print("[Classification Report]")
    print(classification_report(y_test, y_pred, target_names=classes))
    print("[Confusion Matrix]")
    print(confusion_matrix(y_test, y_pred))