import numpy as np
import matplotlib.pyplot as plt

from load_data import load_data


def noise_constellation():
    # Load data
    xs, ys, bits = load_data("test5.csv")
    # 添加不同颜色
    base_colors = plt.get_cmap('tab20')(np.linspace(0, 1, 16))
    plt.scatter(xs, ys, c=[base_colors[int(b, 2)] for b in bits], s=6)
    # 作图
    plt.title('Original Noisy 16QAM Constellation')
    plt.xlabel('In-phase (I)')
    plt.ylabel('Quadrature (Q)')
    plt.grid(True, alpha=0.6)
    plt.axis('equal')
    plt.axhline(0, color='k', linewidth=2)
    plt.axvline(0, color='k', linewidth=2)
    plt.show()


def gan_generate_constellation(gen_points, gan_bits):
    base_colors = plt.get_cmap('tab20')(np.linspace(0, 1, 16))
    plt.scatter(gen_points[:, 0], gen_points[:, 1], c=[base_colors[bit] for bit in gan_bits], s=10)
    plt.title('GAN Generated Noisy 16QAM Constellation')
    plt.xlabel('In-phase (I)')
    plt.ylabel('Quadrature (Q)')
    plt.grid(True, alpha=0.6)
    plt.axis('equal')
    plt.axhline(0, color='k', linewidth=2)
    plt.axvline(0, color='k', linewidth=2)
    plt.show()


if __name__ == "__main__":
    noise_constellation()
