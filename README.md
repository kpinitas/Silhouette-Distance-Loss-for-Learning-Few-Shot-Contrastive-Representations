# Silhouette Distance Loss for Learning Few-Shot Contrastive Representations

A PyTorch implementation of the Silhouette Distance (SD) Loss from the paper:

["Silhouette Distance Loss for Learning Few-Shot Contrastive Representations"Kosmas Pinitas, Nemanja Rasajski, Konstantinos Makantasis, Georgios N. Yannakakis (2024)](https://proceedings.mlr.press/v263/kosmas24a.html)

## About

The Silhouette Distance Loss is a supervised contrastive learning objective tailored for few-shot learning scenarios. Inspired by the silhouette clustering index, this loss encourages representations that are:

* Cohesive: samples of the same class are close to each other
* Separated: different class clusters are pushed far apart

This contrasts with standard Supervised Contrastive Loss (SCL), which mainly emphasizes inter-class separation.

## Features

* Works for images and text
* Supports N-way K-shot classification episodes
* Compatible with frozen pretrained backbones (e.g. CLIP, BERT, DINOv2)
* Competitive or superior performance in 5-way, 10-way, and 20-way few-shot tasks
* Smooth, differentiable approximation of the silhouette score

## Performance

The SD loss shows strong performance:
* On mini-ImageNet, FC100, Banking77, and Clinic150 datasets
* Especially in challenging 20-way 1-shot and cross-domain settings

## Usage Example

## Citation

```bibtex
@article{pinitas2024silhouette,
  title={Silhouette Distance Loss for Learning Few-Shot Contrastive Representations},
  author={Pinitas, Kosmas and Rasajski, Nemanja and Makantasis, Konstantinos and Yannakakis, Georgios N},
  journal={Proceedings of Machine Learning Research},
  volume={1},
  pages={18},
  year={2024}
}
