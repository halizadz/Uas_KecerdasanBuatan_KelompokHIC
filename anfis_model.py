import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import joblib

# 1. Load data dan hasil fuzzy
df = pd.read_csv("teknologi.csv")
fuzzy_results = pd.read_csv("fuzzy_output.csv")

# 2. Persiapkan input dan target
X = df[['Scope', 'Prospects', 'Potential', 'Economy', 'Efficiency']].values
y = fuzzy_results['Criticality'].values.reshape(-1, 1)

# 3. Normalisasi
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# 4. Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=1)

# 5. Convert to torch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# 6. Define ANFIS model dengan MF Gaussian
import torch
import torch.nn as nn

class CustomANFIS(nn.Module):
    def __init__(self, n_input, n_rules):
        super().__init__()
        self.n_input = n_input
        self.n_rules = n_rules
        
        # Parameter untuk membership function (Gaussian)
        self.mu = nn.Parameter(torch.rand(n_input, n_rules))  # mean
        self.sigma = nn.Parameter(torch.rand(n_input, n_rules))  # std dev
        
        # Layer konsekuen (linear)
        self.consequent = nn.Linear(n_input * n_rules, 1, bias=False)

        # Inisialisasi untuk early stopping
        self.best_loss = float('inf')
        self.patience = 10
        self.counter = 0
    
    def forward(self, x):
        # Hitung membership values (Gaussian MF)
        x = x.unsqueeze(-1).expand(-1, -1, self.n_rules)
        mf = torch.exp(-((x - self.mu)**2) / (2 * self.sigma**2))
        
        # Hitung firing strength (product rule)
        strength = torch.prod(mf, dim=1)
        strength = strength / (strength.sum(dim=1, keepdim=True) + 1e-12)
        
        # Hitung output
        x_rep = x.reshape(x.shape[0], -1)
        consequent_out = self.consequent(x_rep)
        return torch.sum(strength * consequent_out, dim=1).unsqueeze(-1)

    def add_early_stopping(self, patience=10):
        self.best_loss = float('inf')
        self.patience = patience
        self.counter = 0

    def check_early_stopping(self, current_loss):
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

model = CustomANFIS(n_input=5, n_rules=5)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
model.add_early_stopping(patience=20)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)

# 7. Training loop
losses = []
for epoch in range(500):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

    scheduler.step(loss)

    if model.check_early_stopping(loss.item()):
        print(f"Early stopping at epoch {epoch}")
        break

print("Training selesai.")

# 8. Evaluasi
model.eval()
with torch.no_grad():
    predictions = model(X_test_tensor)
    predictions_rescaled = scaler_y.inverse_transform(predictions.numpy().reshape(-1, 1))
    y_test_rescaled = scaler_y.inverse_transform(y_test_tensor.numpy())

# 9. Klasifikasi berdasarkan beta_cr
beta_cr = 0.477  # Threshold dari fuzzy model
predicted_status = ['Critical' if x >= beta_cr else 'Non-Critical' for x in predictions_rescaled.flatten()]
actual_status = ['Critical' if x >= beta_cr else 'Non-Critical' for x in y_test_rescaled.flatten()]

# 10. Tampilkan hasil
print("\nHasil prediksi ANFIS pada data uji:")
for i, (pred, actual) in enumerate(zip(predictions_rescaled, y_test_rescaled)):
    print(f"Tech {i+1}: Pred={pred[0]:.3f} ({predicted_status[i]}) | Target={actual[0]:.3f} ({actual_status[i]})")

# 11. Simpan model
torch.save(model.state_dict(), "anfis_model.pth")

# 12. Visualisasi loss
plt.plot(losses)
plt.title('ANFIS Training Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.savefig('anfis_loss.png')
plt.show()
joblib.dump(scaler_X, 'scaler_X.save')
joblib.dump(scaler_y, 'scaler_y.save')