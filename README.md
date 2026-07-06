# [Deep Augmented Inter-class Hashing for Cross-Modal Retrieval](https://www.sciencedirect.com/science/article/pii/S1566253526004732)

This repository provides the official implementation of **Deep Augmented Inter-class Hashing for Cross-Modal Retrieval (DAIH)**, which has been accepted by **Information Fusion**.

DAIH is a supervised cross-modal hashing framework designed for efficient image-text retrieval. The core idea is to explicitly separate **class-discriminative information** and **cross-class shared structured attributes** through a multi-modal dual-encoder architecture. In addition, DAIH introduces a clustering label strategy to generate surrogate labels without extra attribute annotations and an adversarial mutual information loss to reduce redundant overlap between the two encoding spaces.

## Datasets

The experiments are conducted on the following public benchmark datasets:

- [MIRFLICKR-25K](https://press.liacs.nl/mirflickr/)
- [NUS-WIDE](https://lms.comp.nus.edu.sg/wp-content/uploads/2019/research/)
- [MS COCO](https://cocodataset.org/)

Please download the datasets from the official websites and organize them according to the required data structure before training and evaluation.

## Acknowledgement

Our implementation refers to the codebase of [DCHMT](https://github.com/kalenforn/DCHMT). We sincerely thank the authors for releasing their code.

## Citation

If you find this code useful for your research, please consider citing our paper:

```bibtex
@article{wu2026deep,
  title={Deep Augmented Inter-class Hashing for Cross-Modal Retrieval},
  author={Wu, Lei and Qin, Qibing and Cao, Yuxin and Zhang, Wenfeng and Huang, Lei},
  journal={Information Fusion},
  year={2026}
}
