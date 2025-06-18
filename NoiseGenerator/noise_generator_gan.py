import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from load_data import load_data
from noise_constellation import gan_generate_constellation, plot_gan_losses


class Generator(nn.Module):
    def __init__(self, latent_dim, cond_dim=16, out_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + cond_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim)
        )

    def forward(self, z, bits):
        x = torch.cat([z, bits], dim=1)
        return self.net(x)


class Discriminator(nn.Module):
    def __init__(self, cond_dim=16, in_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim + cond_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x, bits):
        x = torch.cat([x, bits], dim=1)
        return self.net(x)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Training on: {device}")
xs, ys, bits = load_data("test5.csv")
data = np.stack([xs, ys], axis=1)
bits_int = np.array([int(b, 2) for b in bits])
# one-hot
bits_oh = np.eye(16)[bits_int]

data = torch.tensor(data, dtype=torch.float32).to(device)
bits_oh = torch.tensor(bits_oh, dtype=torch.float32).to(device)
dataloader = DataLoader(TensorDataset(data, bits_oh), batch_size=128, shuffle=True)

latent_dim = 10
cond_dim = 16
generator = Generator(latent_dim, cond_dim).to(device)
discriminator = Discriminator(cond_dim).to(device)
loss_fn = nn.BCELoss()
g_optimizer = optim.Adam(generator.parameters(), lr=0.001)
d_optimizer = optim.Adam(discriminator.parameters(), lr=0.001)

epochs = 120
g_losses = []
d_losses = []
for epoch in range(epochs):
    for real_batch, bits_batch in dataloader:
        real_batch = real_batch.to(device)
        bits_batch = bits_batch.to(device)
        batch_size = real_batch.size(0)
        real_labels = torch.ones((batch_size, 1), device=device)
        fake_labels = torch.zeros((batch_size, 1), device=device)

        # Training discriminator
        g_loss = 0
        for _ in range(2):
            outputs = discriminator(real_batch, bits_batch)
            d_loss_real = loss_fn(outputs, real_labels)

            z = torch.randn(batch_size, latent_dim, device=device)
            fake_data = generator(z, bits_batch)
            outputs = discriminator(fake_data.detach(), bits_batch)
            d_loss_fake = loss_fn(outputs, fake_labels)

            d_loss = d_loss_real + d_loss_fake
            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

        # Training generator
        z = torch.randn(batch_size, latent_dim, device=device)
        fake_data = generator(z, bits_batch)
        outputs = discriminator(fake_data, bits_batch)
        g_loss = loss_fn(outputs, real_labels)
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

    g_losses.append(g_loss.item())
    d_losses.append(d_loss.item())
    print(f"Epoch {epoch}: D_loss={d_loss.item():.4f}, G_loss={g_loss.item():.4f}")


def generate_gan_samples(generator, num_samples=1000, latent_dim=10, cond_dim=16, device='cpu'):
    generator.eval()
    bits_idx = np.random.randint(0, cond_dim, size=num_samples)
    bits_oh = np.eye(cond_dim)[bits_idx]
    bits_oh = torch.tensor(bits_oh, dtype=torch.float32).to(device)
    with torch.no_grad():
        z = torch.randn(num_samples, latent_dim, device=device)
        gen_points = generator(z, bits_oh).cpu().numpy()
    return gen_points, bits_idx


if __name__ == "__main__":
    gan_points, gan_bits = generate_gan_samples(generator, num_samples=10000, latent_dim=10, cond_dim=16, device=device)
    plot_gan_losses(g_losses, d_losses)
    gan_generate_constellation(gan_points, gan_bits)
