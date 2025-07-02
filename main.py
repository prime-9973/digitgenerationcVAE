import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# --- モデル定義 ---
class Encoder(nn.Module):
    def __init__(self, latent_dim=3, num_classes=10):
        super().__init__()
        self.label_embedding = nn.Linear(num_classes, 16)
        self.fc_hidden = nn.Linear(28 * 28 + 16, 128)
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)

    def forward(self, image, label):
        flattened_image = image.view(image.size(0), -1)
        label_one_hot = F.one_hot(label, num_classes=10).float()
        label_embedding = F.relu(self.label_embedding(label_one_hot))
        concatenated_input = torch.cat([flattened_image, label_embedding], dim=1)
        hidden = F.relu(self.fc_hidden(concatenated_input))
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim=3, num_classes=10):
        super().__init__()
        self.label_embedding = nn.Linear(num_classes, 16)
        self.fc_hidden = nn.Linear(latent_dim + 16, 128)
        self.fc_out = nn.Linear(128, 28 * 28)

    def forward(self, latent_vector, label):
        label_one_hot = F.one_hot(label, num_classes=10).float()
        label_embedding = F.relu(self.label_embedding(label_one_hot))
        concatenated = torch.cat([latent_vector, label_embedding], dim=1)
        hidden = F.relu(self.fc_hidden(concatenated))
        output = torch.sigmoid(self.fc_out(hidden))
        return output.view(-1, 1, 28, 28)

class CVAE(nn.Module):
    def __init__(self, latent_dim=3, num_classes=10):
        super().__init__()
        self.encoder = Encoder(latent_dim, num_classes)
        self.decoder = Decoder(latent_dim, num_classes)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, y):
        mu, logvar = self.encoder(x, y)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z, y), mu, logvar

# --- デバイス設定 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- モデルのロード ---
@st.cache_resource
def load_model(path="cvae.pth"):
    model = CVAE().to(device)
    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model

# --- Streamlit UI ---
st.title("CVAE 手書き数字生成アプリ")
st.write("0〜9の数字を選んで、対応する画像を生成します。")

digit = st.number_input("生成したい数字 (0〜9)", min_value=0, max_value=9, value=5)
num_samples = st.slider("生成する画像数", 1, 16, 6)

if st.button("画像を生成"):
    model = load_model()

    latent_dim = 3
    z = torch.randn(num_samples, latent_dim).to(device)
    labels = torch.full((num_samples,), digit, dtype=torch.long).to(device)

    with torch.no_grad():
        images = model.decoder(z, labels)

    images = images.cpu().numpy()

    # --- 結果の描画 ---
    fig, axes = plt.subplots(1, num_samples, figsize=(num_samples * 2, 2))
    if num_samples == 1:
        axes = [axes]
    for i in range(num_samples):
        axes[i].imshow(images[i].squeeze(), cmap="gray")
        axes[i].axis("off")
    st.pyplot(fig)
