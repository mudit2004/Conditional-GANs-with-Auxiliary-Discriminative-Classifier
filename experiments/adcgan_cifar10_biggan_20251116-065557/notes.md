# adcgan_cifar10_biggan_20251116-065557

## Run Details
- Dataset: CIFAR-10
- Method: ADC-GAN
- Source repo: liang-hou/adcgan
- Backend: BigGAN-PyTorch
- Logs root: /content/adcgan/BigGAN-PyTorch/logs/adcgan_cifar10_biggan_20251116-065557
- Weights root: /content/adcgan/BigGAN-PyTorch/weights/adcgan_cifar10_biggan_20251116-065557
- Colab GPU: A100
- Command used:

```bash

cd /content/adcgan/BigGAN-PyTorch

python train.py \
  --shuffle \
  --batch_size 50 \
  --parallel \
  --num_G_accumulations 1 \
  --num_D_accumulations 1 \
  --num_epochs 1000 \
  --num_D_steps 4 \
  --G_lr 2e-4 \
  --D_lr 2e-4 \
  --dataset C10 \
  --G_ortho 0.0 \
  --G_attn 0 \
  --D_attn 0 \
  --G_init N02 \
  --D_init N02 \
  --ema \
  --use_ema \
  --ema_start 1000 \
  --test_every 2000 \
  --save_every 2000 \
  --num_best_copies 1 \
  --num_save_copies 0 \
  --seed 0 \
  --loss adcgan \
  --G_lambda 1.0 \
  --D_lambda 1.0 \
  --data_root /content/drive/MyDrive/adcgan-data \
  --logs_root /content/adcgan/BigGAN-PyTorch/logs/adcgan_cifar10_biggan_20251116-065557 \
  --experiment_name adcgan_cifar10_biggan_20251116-065557

```
