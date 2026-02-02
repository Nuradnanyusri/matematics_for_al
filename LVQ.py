import numpy as np
import pandas as pd

# ================================
# 1. LOAD DATASET
# ================================
file_path = "Dataset_LVQ_50_Data.xlsx"
df = pd.read_excel(file_path)

X = df[['GPA', 'Income', 'Dependents', 'Achievements']].values
y = df['Class'].values


# ================================
# 2. NORMALISASI MIN-MAX
# ================================
def min_max_normalization(X):
    return (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))

X_norm = min_max_normalization(X)


# ================================
# 3. INISIALISASI BOBOT
# ================================
idx_c1 = np.where(y == 1)[0][0]
idx_c2 = np.where(y == 2)[0][0]
idx_c3 = np.where(y == 3)[0][0]

W = np.array([X_norm[idx_c1], X_norm[idx_c2], X_norm[idx_c3]])
W_class = np.array([1, 2, 3])
W_initial = W.copy()


# ================================
# 4. PARAMETER LVQ
# ================================
alpha = 0.3
decay = 0.9
epochs = 5


# ================================
# 5. JARAK EUCLIDEAN
# ================================
def euclidean_distance(x, w):
    return np.sqrt(np.sum((x - w) ** 2))


# ================================
# 6. TRAINING + SIMPAN PERHITUNGAN
# ================================
epoch_weights = []
epoch_calculations = []

for epoch in range(epochs):
    print(f"\n{'='*65}")
    print(f"EPOCH {epoch+1} | Learning Rate (α) = {alpha:.4f}")
    print(f"{'='*65}")

    W_epoch_start = W.copy()
    epoch_rows = []

    for i in range(len(X_norm)):
        x = X_norm[i]
        target = y[i]

        d1 = euclidean_distance(x, W_epoch_start[0])
        d2 = euclidean_distance(x, W_epoch_start[1])
        d3 = euclidean_distance(x, W_epoch_start[2])

        distances = [d1, d2, d3]
        winner = np.argmin(distances) + 1

        # ===== OUTPUT RUMUS LVQ DI TERMINAL =====
        if i < 3:
            print(f"\nData ke-{i+1}")
            print(f"d_W1 = sqrt(sum((X - W1)^2)) = {d1:.6f}")
            print(f"d_W2 = sqrt(sum((X - W2)^2)) = {d2:.6f}")
            print(f"d_W3 = sqrt(sum((X - W3)^2)) = {d3:.6f}")
            print(f"Winner = argmin(d_W1, d_W2, d_W3) = {winner}")

        # simpan ke excel
        epoch_rows.append([
            df.loc[i, 'ID'],
            x[0], x[1], x[2], x[3],
            d1, d2, d3,
            winner
        ])

        # ===== UPDATE BOBOT LVQ1 =====
        if W_class[winner-1] == target:
            W[winner-1] = W[winner-1] + alpha * (x - W[winner-1])
        else:
            W[winner-1] = W[winner-1] - alpha * (x - W[winner-1])

    epoch_calculations.append(epoch_rows)
    epoch_weights.append(W.copy())
    alpha *= decay


# ================================
# 7. PREDIKSI & AKURASI
# ================================
def lvq_predict(X, W, W_class):
    return np.array([
        W_class[np.argmin([euclidean_distance(x, w) for w in W])]
        for x in X
    ])

y_pred = lvq_predict(X_norm, W, W_class)
accuracy = np.mean(y_pred == y) * 100


# ================================
# 8. SIMPAN KE EXCEL (SEMUA EPOCH)
# ================================
writer = pd.ExcelWriter("LVQ_Training_Result_50Data_5Epoch.xlsx", engine="openpyxl")

# Parameter LVQ
df_param = pd.DataFrame(W_initial, columns=['X1','X2','X3','X4'])
df_param['Class'] = W_class
df_param.to_excel(writer, sheet_name="Parameter_LVQ", index=False)

# Bobot per epoch
for i, w_epoch in enumerate(epoch_weights):
    df_epoch = pd.DataFrame(w_epoch, columns=['X1','X2','X3','X4'])
    df_epoch['Class'] = W_class
    df_epoch.to_excel(writer, sheet_name=f"Epoch_{i+1}", index=False)

# ===== PERHITUNGAN EPOCH 1–5 =====
for ep in range(5):
    df_calc = pd.DataFrame(
        epoch_calculations[ep],
        columns=['ID','X1','X2','X3','X4','d_W1','d_W2','d_W3','Winner']
    )
    df_calc.to_excel(writer, sheet_name=f"Perhitungan_LVQ_Epoch{ep+1}", index=False)

# Prediksi
df_pred = df.copy()
df_pred['Predicted_Class'] = y_pred
df_pred.to_excel(writer, sheet_name="Prediksi", index=False)

# Akurasi
pd.DataFrame({
    "Epoch": [epochs],
    "Akurasi (%)": [accuracy]
}).to_excel(writer, sheet_name="Akurasi", index=False)

writer.close()

print("\nFile 'LVQ_Training_Result_50Data_5Epoch.xlsx' BERHASIL dibuat lengkap.")
print(f"Akurasi Training LVQ = {accuracy:.2f}%")