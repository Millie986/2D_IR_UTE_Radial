import numpy as np
import matplotlib.pyplot as plt

class VirtualPhantom():
    def __init__(self, img_size):
        self.img_size = img_size
        self.inner_size = int(0.25* img_size)
        self.mixed_size = int(0.50 * img_size)
        self.outer_size = int(0.75 * img_size)
        self.upper_square = int(0.078 * img_size)

        # gap between upper square and mixed square
        self.gap = max(1, int(0.02 * img_size))

        # Inner square (pure myelin) - centered
        self.top_inner = (self.img_size - self.inner_size) // 2
        self.bottom_inner = self.top_inner + self.inner_size

        # Mixed square (mixed signals) - centered
        self.top_mixed = (self.img_size - self.mixed_size) // 2
        self.bottom_mixed = self.top_mixed + self.mixed_size

        # Outer square (pure WM) - centered
        self.top_outer = (self.img_size - self.outer_size) // 2
        self.bottom_outer = self.top_outer + self.outer_size

        # Upper square: centered horizontally, above mixed square, with a gap
        self.x0 = self.img_size // 2 - self.upper_square // 2
        self.x1 = self.x0 + self.upper_square

        self.y1 = self.top_mixed - self.gap
        self.y0 = self.y1 - self.upper_square

        if self.y0 < 0:
            self.y0 = 0
            self.y1 = self.upper_square
        
    def create_maps(self, t1, t2, inner, mixed, outer, upper_sqaure=True):
        # T1 map
        self.T1_map = np.full([self.img_size, self.img_size], t1, dtype=np.float64)

        # T2 map
        self.T2_map = np.full([self.img_size, self.img_size], t2, dtype=np.float64)

        # Probability map
        prob_map = np.zeros([self.img_size, self.img_size], dtype=np.float64)
        prob_map[self.top_outer:self.bottom_outer, self.top_outer:self.bottom_outer] = outer
        prob_map[self.top_mixed:self.bottom_mixed, self.top_mixed:self.bottom_mixed] = mixed
        prob_map[self.top_inner:self.bottom_inner, self.top_inner:self.bottom_inner] = inner
        if upper_sqaure:
            prob_map[self.y0:self.y1, self.x0:self.x1] = 1

        self.prob_map = prob_map
        return self.T1_map, self.T2_map, self.prob_map

    def phantom_display(self, title):
        plt.figure(figsize=(5, 5))

        plt.subplot(2, 2, 1)
        plt.imshow(self.T1_map, cmap='gray', vmin=0, vmax=1)
        plt.title(f"{title} T1 Map")

        plt.subplot(2, 2, 2)
        plt.imshow(self.T2_map, cmap='gray', vmin=0, vmax=1)
        plt.title(f"{title} T2 Map")

        plt.subplot(2, 2, 3)
        plt.imshow(self.prob_map, cmap='gray', vmin=0, vmax=1)
        plt.title(f"{title} Probability Map")

        plt.tight_layout()
        plt.show()

