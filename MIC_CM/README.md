# 2019 ICCV MIC: Mining Interclass Characteristics for Improved Metric Learning

[[Paper]](https://ieeexplore.ieee.org/document/9010968/)
[[Code]](https://github.com/Confusezius/metric-learning-mining-interclass-characteristics)

# Changes

1. Clip's output before fc is used for init_cluster_generation.
2. Use encode_img & encode_txt for clusters update (deep_cluster).
3. For cls, MarginLoss is: img-img, txt-txt, img-txt, txt-img.
4. For cls, MutualLoss is: img-img, txt-txt.

# Performance

| Method\Dataset |  flickr  | nuswide  | coco |
|:--------------:|:--------:|:--------:|:----:|
|       m1       | 0.835@54 | 0.726@69 |  -   |

m1:
flickr:
[mAP:0.8352][I→T:0.8464][T→I:0.8239]
coco:
[mAP:0.7257][I→T:0.7192][T→I:0.7321]

# Parameters

```
# same
args.batch_size = 128
args.n_epochs = 100
args.n_bits = 32
```

| Method\Type | backbone | optimizer |  lr  |  wd  | scheduler |            memo             |
|:-----------:|:--------:|:---------:|:----:|:----:|:---------:|:---------------------------:|
|     m3      |   clip   |   adam    | 1e-5 | 1e-4 |   none    | adv_weight=10, n_samples=30 |
