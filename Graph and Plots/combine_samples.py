
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

fig, axes = plt.subplots(2, 2, figsize=(10, 8))

files = [
    ('acgan_synthetic_1d.png', 'AC-GAN'),
    ('pdgan_synthetic_1d.png', 'PD-GAN'),
    ('adcgan_synthetic_1d.png', 'ADC-GAN'),
    ('gamma_adcgan_synthetic_1d.png', 'γ-ADC-GAN (0.5)')
]

for ax, (fname, title) in zip(axes.flat, files):
    img = mpimg.imread(fname)
    ax.imshow(img)
    ax.set_title(title)
    ax.axis('off')

plt.tight_layout()
plt.savefig('synthetic_1d_samples.png', dpi=300)
plt.close()

