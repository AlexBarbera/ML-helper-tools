# ML-helper-tools

This repo contains helper functions and skeletons for training ML models in an attempt to avoid repeating code search.

The idea is to have a single repo to refer to for implementation and editing instead of writting from scratch, this is not supposed to be a script repo.

## Losses
 - **WassersteinLoss**
   - Earth-mover distance is a better metric for small permutations in image data compared to `MSELoss`.


 - **PerceptualLoss**
   - Uses an encoder network to compare extracted features between target and generated, useful for SuperResolution. (paper)[https://arxiv.org/pdf/1609.04802v5]


 - **TotalVariationLoss**
   - Jitters the image and calculates `MSE`.


 - **DeepEnergyLoss**
   - Maximizes difference between labels rather than set to `1` or `0`.


 - **CycleGANLoss**

## Models
 - **CycleGAN**
   - Unaligned datasets for domain tranformation.


 - **Siamese Network**
   - Classifies differences in latent space of encoder.


 - **ResNet Generator**
   - Standard ResNet


 - **MultiLayer Perceptron** (Linear and Convolutional)

## Utils
 - LightningWrapper for training
   - Allows to easily use distributed training and GPU/TPU usage.


 - **TripletDataset**
   - Handles tuples of 3 (anchor [reference], positive [simmilar], negative [different]).


 - **ABDataset**
   - Matches 2 classes of data in pairs.


 - **MulticlassDataset**
   - Matches N classes of data in pairs.


 - **ZCA Whitening**
   - Normalizes images so that covariance $`\sum`$ is the Identity matrix leading to decorrelated features.
   - According to the (paper)[https://arxiv.org/pdf/1804.08450v1], it should be applied batch-wise.
