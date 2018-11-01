# Unsupervised Anomaly Detection using Generative Adversarial Network on medical X-Ray image

Article: https://medium.com/vitalify-asia/gan-for-unsupervised-anomaly-detection-on-x-ray-images-6b9f678ca57d

## Data
- MURA data set https://stanfordmlgroup.github.io/competitions/mura/
- Public, detect abnormality in X-Ray images.

## Model 
- Bidirection GAN / ALI : https://arxiv.org/abs/1605.09782 / https://arxiv.org/abs/1606.00704
- Alpha-GAN (VAE + GAN): https://arxiv.org/abs/1706.04987

## Approach
Leveraging the ability to unsupervisedly learned the structure of data to generate realisitic image, this experiments aims to use that ability to perform binary classification when only trained on one class.

## Usage
Run ```python main.py --help``` for full detail.

Example:
```python main.py --batch_size 128 --imsize 64 --dataset mura --adv_loss inverse --version sabigan_wrist --image_path ~/datasets/ --use_tensorboard true --mura_class XR_WRIST --mura_type negative
```

### How:
- Train GAN model with the ability to inference on the latent variable (VAE+GAN / BiGAN) on only 'negative class'
- Let the model learn until it can generate good looking images.
- Use the Encoder, Generator, Discriminator outputs and hidden features to calculate 'Reconstruction loss' and 'Feature matching' loss.
- Classify into 'negative' or 'positive' based on the score above.

## Generative results:
#### Bigan
![](images/bigan.gif)

#### Alpha-GAN 
![](images/alpha.gif)

## Discriminative Result
#### Bigan

![](images/bigan.png)

#### Alpha-GAN

![](images/alpha.png)


#### References:
- Thank https://github.com/heykeetae/Self-Attention-GAN for great examples.