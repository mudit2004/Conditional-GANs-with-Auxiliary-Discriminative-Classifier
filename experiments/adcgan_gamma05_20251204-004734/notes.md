# adcgan_gamma05_20251204-004734

## Run Details
- Dataset: CIFAR-10
- Method: ADC-GAN (gamma=0.5)
- Source repo: liang-hou/adcgan
- Backend: BigGAN-PyTorch
- Logs root: /content/adcgan/BigGAN-PyTorch/logs/adcgan_gamma05_20251204-004734
- Weights root: /content/adcgan/BigGAN-PyTorch/weights/adcgan_gamma05_20251204-004734
- Colab GPU: A100
- Command used:

```bash

cd /content/adcgan/BigGAN-PyTorch

timeout 1500s \
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
  --loss adcgan \
  --use_ema \
  --ema_start 1000 \
  --save_every 2000 \
  --test_every 2000 \
  --seed 0 \
  --data_root /content/drive/MyDrive/adcgan-data \
  --logs_root /content/adcgan/BigGAN-PyTorch/logs/adcgan_gamma05_20251204-004734 \
  --experiment_name adcgan_gamma05_20251204-004734

```
