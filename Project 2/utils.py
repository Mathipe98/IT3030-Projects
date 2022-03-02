import numpy as np
import matplotlib.pyplot as plt


def pad(label: int, need_padding: bool) -> str:
    if not need_padding:
        return str(label)
    label = str(label)
    padding = 3 - len(label)
    for _ in range(padding):
        label = "0" + label
    return label

def visualize_pictures(x: np.ndarray, y: np.ndarray, decoded_imgs: np.ndarray = None, filename: str=None) -> None:
    n = 10
    end = max(0, x.shape[0] - 11)
    if end == 0:
        random_start = 0
    else:
        random_start = np.random.randint(0, end)
    need_padding = x.shape[-1] > 1
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # Display original
        ax = plt.subplot(2, n, i+1)
        original = x[i + random_start].astype(np.float64)
        plt.imshow(original)
        plt.title(pad(y[i + random_start], need_padding))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        if decoded_imgs is not None:
            # Display reconstruction
            ax = plt.subplot(2, n, i + 1 + n)
            copy = decoded_imgs[i + random_start]
            plt.imshow(copy)
            plt.title(pad(y[i + random_start], need_padding))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
    if filename is not None:
        plt.savefig(filename)
    plt.show()