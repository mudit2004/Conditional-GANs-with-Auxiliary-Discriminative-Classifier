# pdgan_cifar10_biggan_20251203-163507

## Run Details
- Dataset: CIFAR-10
- Method: PD-GAN
- Source repo: liang-hou/adcgan
- Backend: BigGAN-PyTorch
- Logs root: /content/adcgan/BigGAN-PyTorch/logs/pdgan_cifar10_biggan_20251203-163507
- Weights root: /content/adcgan/BigGAN-PyTorch/weights/pdgan_cifar10_biggan_20251203-163507
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
  --G_ortho 0.0 \
  --G_attn 0 \
  --D_attn 0 \
  --G_init N02 \
  --D_init N02 \
  --use_ema \
  --ema_start 1000 \
  --test_every 2000 \
  --save_every 2000 \
  --num_best_copies 1 \
  --num_save_copies 0 \
  --seed 0 \
  --loss hinge \
  --data_root /content/drive/MyDrive/adcgan-data \
  --logs_root /content/adcgan/BigGAN-PyTorch/logs/pdgan_cifar10_biggan_20251203-163507 \
  --experiment_name pdgan_cifar10_biggan_20251203-163507

```
