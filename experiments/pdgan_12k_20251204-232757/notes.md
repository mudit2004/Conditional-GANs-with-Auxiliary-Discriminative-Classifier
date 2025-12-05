# pdgan_12k_20251204-232757

## Run Details
- Dataset: CIFAR-10
- Method: PD-GAN (12k baseline)
- Source repo: liang-hou/adcgan
- Backend: BigGAN-PyTorch
- Logs root: /content/adcgan/BigGAN-PyTorch/logs/pdgan_12k_20251204-232757
- Weights root: /content/adcgan/BigGAN-PyTorch/weights/pdgan_12k_20251204-232757
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
  --loss hinge \
  --use_ema \
  --ema_start 1000 \
  --test_every 2000 \
  --save_every 2000 \
  --seed 0 \
  --data_root /content/drive/MyDrive/adcgan-data \
  --logs_root /content/adcgan/BigGAN-PyTorch/logs/pdgan_12k_20251204-232757 \
  --experiment_name pdgan_12k_20251204-232757

```
