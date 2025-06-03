import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from load_data import load_data
from noise_constellation import gan_generate_constellation

xs, ys, bits = load_data("test5.csv")
data = np.stack([xs, ys], axis=1)

bits_int = np.array([int(b, 2) for b in bits])
# one-hot
bits_oh = np.eye(16)[bits_int]

data = torch.tensor(data, dtype=torch.float32)
bits_oh = torch.tensor(bits_oh, dtype=torch.float32)
dataloader = DataLoader(TensorDataset(data, bits_oh), batch_size=128, shuffle=True)


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

# init cGAN
latent_dim = 10
cond_dim = 16
generator = Generator(latent_dim, cond_dim)
discriminator = Discriminator(cond_dim)
criterion = nn.BCELoss()
g_optimizer = optim.Adam(generator.parameters(), lr=0.001)
d_optimizer = optim.Adam(discriminator.parameters(), lr=0.001)

# Training
epochs = 30
for epoch in range(epochs):
    for real_batch, bits_batch in dataloader:
        batch_size = real_batch.size(0)
        real_labels = torch.ones((batch_size, 1))
        fake_labels = torch.zeros((batch_size, 1))

        # Training discriminator
        outputs = discriminator(real_batch, bits_batch)
        d_loss_real = criterion(outputs, real_labels)

        z = torch.randn(batch_size, latent_dim)
        fake_data = generator(z, bits_batch)
        outputs = discriminator(fake_data.detach(), bits_batch)
        d_loss_fake = criterion(outputs, fake_labels)

        d_loss = d_loss_real + d_loss_fake
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        # Training generator
        z = torch.randn(batch_size, latent_dim)
        fake_data = generator(z, bits_batch)
        outputs = discriminator(fake_data, bits_batch)
        g_loss = criterion(outputs, real_labels)
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

    print(f"Epoch {epoch}: D_loss={d_loss.item():.4f}, G_loss={g_loss.item():.4f}")


def generate_gan_samples(generator, num_samples=1000, latent_dim=10, cond_dim=16):
    generator.eval()
    # Generate random bits
    bits_idx = np.random.randint(0, cond_dim, size=num_samples)
    bits_oh = np.eye(cond_dim)[bits_idx]
    bits_oh = torch.tensor(bits_oh, dtype=torch.float32)
    with torch.no_grad():
        z = torch.randn(num_samples, latent_dim)
        gen_points = generator(z, bits_oh).numpy()
    return gen_points, bits_idx


if __name__ == "__main__":
    gan_points, gan_bits = generate_gan_samples(generator, num_samples=10000, latent_dim=10, cond_dim=16)
    gan_generate_constellation(gan_points, gan_bits)
