
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

files = [
    ('dloss_1d.png', 'D Loss (1D Synthetic)'),
    ('gloss_1d.png', 'G Loss (1D Synthetic)')
]

for ax, (fname, title) in zip(axes, files):
    img = mpimg.imread(fname)
    ax.imshow(img)
    ax.set_title(title)
    ax.axis('off')

plt.tight_layout()
plt.savefig('synthetic_1d_losses.png', dpi=300)
plt.close()

