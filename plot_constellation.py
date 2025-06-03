import matplotlib.pyplot as plt
import numpy as np

from generate_16qam_symbols import generate_original_symbols


# 标准星座图
def standard_constellation():
    # 画出星座点
    original_symbols, bits = generate_original_symbols(100)
    xs = [point[0] for point in original_symbols]
    ys = [point[1] for point in original_symbols]
    # 添加不同颜色
    base_colors = plt.get_cmap('tab20')(np.linspace(0, 1, 16))
    plt.scatter(xs, ys, c=[base_colors[int(b, 2)] for b in bits], s=50)
    # 作图
    plt.title('16QAM Constellation')
    plt.xlabel('In-phase (I)')
    plt.ylabel('Quadrature (Q)')
    plt.grid(True, alpha=0.6)
    plt.axis('equal')
    plt.axhline(0, color='k', linewidth=2)
    plt.axvline(0, color='k', linewidth=2)
    plt.show()


if __name__ == "__main__":
    standard_constellation()
