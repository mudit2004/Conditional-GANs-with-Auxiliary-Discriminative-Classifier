# adcgan_gamma05_12k_20251205-002946

## Run Details
- Dataset: CIFAR-10
- Method: ADC-GAN (12k baseline)
- Source repo: liang-hou/adcgan
- Backend: BigGAN-PyTorch
- Logs root: /content/adcgan/BigGAN-PyTorch/logs/adcgan_gamma05_12k_20251205-002946
- Weights root: /content/adcgan/BigGAN-PyTorch/weights/adcgan_gamma05_12k_20251205-002946
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
  --loss acgan \
  --use_ema \
  --ema_start 1000 \
  --test_every 2000 \
  --save_every 2000 \
  --seed 0 \
  --data_root /content/drive/MyDrive/adcgan-data \
  --logs_root /content/adcgan/BigGAN-PyTorch/logs/acgan_12k_20251205-000221 \
  --experiment_name acgan_12k_20251205-000221

```
