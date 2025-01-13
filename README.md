# ML-helper-tools

This repo contains helper functions and skeletons for training ML models in an attempt to avoid repeating code search.

The idea is to have a single repo to refer to for implementation and editing instead of writting from scratch, this is not supposed to be a script repo.

## Losses
 - WassersteinLoss
 - PerceptualLoss
 - TotalVariationLoss
 - DeepEnergyLoss
 - CycleGANLoss

## Models
 - CycleGAN
 - Siamese Network
 - ResNet Generator
 - MultiLayer Perceptron (Linear and Convolutional)

## Utils
 - LightningWrapper for training
 - TripletDataset
 - ABDataset
   - Matches 2 classes of data in pairs.
 - MulticlassDataset
   - Matches N classes of data in pairs.