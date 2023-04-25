import matplotlib.pyplot as plt
import numpy as np

def read_file(path: str) -> str:
    for line in open('names.txt', 'r'):
        yield line.strip()



def plot_char_matrix(ctoi, N):

    # Function to define text color based on the cell background color
    def get_text_color(bg_color):
        threshold = (0.299 * bg_color[0] + 0.587 * bg_color[1] + 0.114 * bg_color[2])
        return "white" if threshold < 0.5 else "black"

    # Get inverse mapping from index to character
    itos = {v: k for k, v in ctoi.items()}

    plt.figure(figsize=(20, 20))
    cmap = plt.get_cmap('Blues')  # Change colormap here if desired
    norm = plt.Normalize(N.min(), N.max())
    colors = cmap(norm(N))
    plt.imshow(colors, aspect='auto')

    for i in range(28):
        for j in range(28):
            chstr = itos[i] + itos[j]
            text_color = get_text_color(colors[i, j])
            plt.text(j, i, chstr, ha="center", va="bottom", color=text_color, fontsize=9, fontweight='bold')
            plt.text(j, i, N[i, j].item(), ha="center", va="top", color=text_color, fontsize=9)

    plt.title("Bigram Heatmap", fontsize=24, y=1.03)
    plt.axis('off')

    # Increase space between heatmap cells
    ax = plt.gca()
    ax.set_xticks(np.arange(-0.5, 28, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 28, 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=2)

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, fraction=0.046, pad=0.04)
    cbar.ax.set_title("Counts", pad=12)

    plt.show()
