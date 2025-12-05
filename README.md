# Conditional GANs with Auxiliary Discriminative Classifier — Reproduction & Extensions

This repository contains the full implementation, experiments, and analysis for my CS436/536 final project:

**“Conditional GANs with Auxiliary Discriminative Classifier: Reproduction and Early Extensions”**  
*(Crystal Sembhi, Mudit Golchha, Binghamton University)*

This project reproduces the **ADC-GAN (Hou et al., ICML 2022)** paper and compares it against **AC-GAN** and **PD-GAN** under a unified BigGAN-CIFAR10 framework.  
It also introduces a small extension **γ-ADC-GAN**, which scales the auxiliary discriminative classifier loss by a factor γ.

---

# 1. Overview

Conditional GANs aim to generate class-specific images.  
The commonly used **AC-GAN** uses an auxiliary classifier trained only on real images, which causes:

- low intra-class diversity  
- mode collapse tendencies  
- training instability  

**ADC-GAN** introduces a **discriminative classifier** trained on **both real and fake samples**, and outputs:

```
Real class k → 2k
Fake class k → 2k + 1
```

This improves **stability**, **diversity**, and **early FID/IS** performance.

In this project, I:

1. Reproduce ADC-GAN on CIFAR-10 using the BigGAN backbone  
2. Implement and compare AC-GAN and PD-GAN using identical training settings  
3. Develop γ-ADC-GAN, a lightweight extension using γ ∈ {1.0, 0.5}  
4. Evaluate all models on CIFAR-10 and a custom 1D synthetic task  

---

# 2. Repository Structure

```
adcgan-repro/
│
├── BigGAN-PyTorch/               
├── custom_losses.py              
├── custom_train_fns.py           
├── experiments/                  
│
├── synthetic_1d/                 
│   ├── train_1d.py
│   ├── acgan_synthetic_1d.png
│   ├── adcgan_synthetic_1d.png
│   ├── gamma_adcgan_synthetic_1d.png
│   ├── pdgan_synthetic_1d.png
│   ├── dloss_1d.png
│   └── gloss_1d.png
│
├── plots/                        
│   ├── adcgan_fid_cifar10.png
│   ├── adcgan_is_cifar10.png
│   ├── fid_adc_pd_ac.png
│   ├── fid_adc_pd_ac_gamma_0_12k.png
│   ├── is_adc_pd_ac.png
│   └── is_adc_pd_ac_gamma_0_12k.png
│
├── ML_FinalReport.pdf            
└── README.md
```

---

# 3. Running Experiments

## 3.1 Install Dependencies
```bash
pip install torch torchvision tqdm numpy matplotlib
```

---

## 3.2 Training ADC-GAN / AC-GAN / PD-GAN

Inside **BigGAN-PyTorch/**:

### ADC-GAN
```bash
python train.py   --loss adcgan   --dataset C10   --use_ema   --batch_size 50   --num_D_steps 4   --save_every 2000   --test_every 2000   --experiment_name adcgan_run
```

### AC-GAN
```bash
python train.py   --loss acgan   --dataset C10   --use_ema   --experiment_name acgan_run
```

### PD-GAN
```bash
python train.py   --loss hinge   --dataset C10   --projection   --use_ema   --experiment_name pdgan_run
```

---

## 3.3 γ-ADC-GAN (New Hypothesis)

Set γ inside `custom_losses.py` / `custom_train_fns.py`:

```python
G_lambda = gamma
D_lambda = gamma
```

Run:
```bash
python train.py   --loss adcgan   --experiment_name gamma05_adcgan   --use_ema
```

---

# 4. 1D Synthetic Experiment

The synthetic dataset is a **3-mode Gaussian mixture** with means at -4, 0, and +4.

Run:

```bash
python train_1d.py
```

This produces:

- acgan_synthetic_1d.png  
- pdgan_synthetic_1d.png  
- adcgan_synthetic_1d.png  
- gamma_adcgan_synthetic_1d.png  
- dloss_1d.png  
- gloss_1d.png  

These visualizations help compare diversity and training stability.

---

# 5. Results Summary

## CIFAR-10 Early FID (0–12k iterations)

| Model | FID ↓ | Behavior |
|-------|-------|----------|
| AC-GAN | Slow improvement | Unstable classifier |
| PD-GAN | Moderate | Noisy projection term |
| **ADC-GAN** | Best early FID | Stable and diverse |
| **γ-ADC-GAN (0.5)** | Slight FID improvement at several points | Smoothest overall training |

---

## 1D Synthetic Results

| Model | Mode Coverage | Stability |
|-------|--------------|-----------|
| AC-GAN | Collapses to 1–2 modes | Unstable |
| PD-GAN | Covers all modes | Noisy gradients |
| **ADC-GAN** | Best accuracy | Smooth discriminator signals |
| **γ-ADC-GAN (0.5)** | Preserves modes | Smoothest training curve |

---
