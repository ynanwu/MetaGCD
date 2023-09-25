# Learning to Continually Learn in Generalized Category Discovery
This repository is the official implementation of [MetaGCD: Learning to Continually Learn in Generalized Category Discovery](https://arxiv.org/pdf/2308.11063.pdf).

<div align="center">
<img width="80%" alt="Method Overview" src="assets/overview.jpg">
</div>

## Requirements

The code was tested on python3.6 pytorch1.4.0 and CUDA9.2.

We recommend using conda environment to setup all required dependencies:

```setup
conda env create -f environment.yml
conda activate MetaGCD
```

If you have any problem with the above command, you can also install them by `pip install -r requirements.txt`.

## Offline Train
We provide the training script for the following 3 datasets from the NCD benchmark: [CIFAR10](https://pytorch.org/vision/stable/datasets.html)
, [CIFAR100](https://pytorch.org/vision/stable/datasets.html) and [Tiny-ImageNet](https://image-net.org/download.php). To train the models in the paper, run the following commands:

```Meta Training
python methods/contrastive_training/contrastive_learning_based_MAML.py --run_mode 'MetaTrain' --dataset_name <dataset>
```

Set paths to datasets, pre-trained models and desired log directories in ```config.py```

## Online Incremental Learning

To evaluate meta-trained models, run:

```eval
python methods/contrastive_training/contrastive_learning_based_MAML.py --run_mode 'MetaTest' --dataset_name <dataset>
```

## Citation
If you find this codebase useful in your research, consider citing:
```
@inproceedings{
    wu2023metagcd,
    title={MetaGCD: Learning to Continually Learn in Generalized Category Discovery},
    author={Yanan Wu and Zhixiang Chi and Yang Wang and and Songhe Feng},
    booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    year={2023}
}
```
