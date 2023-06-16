# satellite-thermal-geo-localization

This is the official repository for [Long-range UAV Thermal Geo-localization with Satellite Imagery](https://arxiv.org/abs/2306.02994).

```
@misc{xiao2023longrange,
      title={Long-range UAV Thermal Geo-localization with Satellite Imagery}, 
      author={Jiuhong Xiao and Daniel Tortei and Eloy Roura and Giuseppe Loianno},
      year={2023},
      eprint={2306.02994},
      archivePrefix={arXiv},
      primaryClass={cs.RO}
}
```
**Developer: Jiuhong Xiao<br />
Affiliation: [NYU ARPL](https://wp.nyu.edu/arpl/)<br />
Maintainer: Jiuhong Xiao (jx1190@nyu.edu)<br />**

## Dataset
Dataset link: [Download](https://long-range-uav-thermal.s3.me-central-1.amazonaws.com/thermal_h5_datasets.zip)

The ``dataset`` folder should be created in the root folder with the following structure

```
satellite-thermal-geo-localization/datasets/satellite_0_thermalmapping_135/
├── test_database.h5
├── test_queries.h5
├── train_database.h5
├── train_queries.h5
├── extended_database.h5
├── extended_queries.h5 # Need to generate by yourself using trained TGM
├── val_database.h5
└── val_queries.h5
```

Check your md5sum after downloading the dataset:
```
3d2faea188995dd687c77178621b6fc7  extended_database.h5
7ef56acc4a746e7bafe1df91131d7737  test_database.h5
f891d9feca5f4252169a811e8c8e648d  test_queries.h5
f075a7a6db5d5d61be88d252f7d6e05b  train_database.h5
b44ee39173bf356b24690ed6933a6792  train_queries.h5
31923c28dd074ddaacf0c463681f7d2b  val_database.h5
fdcb2e12d9a29b8d20a4cbd88bfe430c  val_queries.h5
```

## Conda Environment Setup
Our repository requires a conda environment. Relevant packages are listed in ``env.yml``. Run the following command to setup the conda environment.
```
conda env create -f env.yml
```

## Training
You can find the training scripts and evaluation scripts in ``scripts`` folder. The scripts is for slurm system to submit sbatch job. If you want to run bash command, change the suffix from ``sbatch`` to ``sh`` and run with bash.

### TGM training

1. To train Thermal Generate Module (TGM), use one of the following scripts:
```
./scripts/train_bing_thermal_translation_100.sbatch # Best choice
./scripts/train_bing_thermal_translation_10.sbatch
./scripts/train_bing_thermal_translation_100_noncontrast.sbatch
./scripts/train_bing_thermal_translation_10_noncontrast.sbatch
```

After training TGM, find your model folder in ``./logs/default/satellite_0_thermalmapping135-datetime``  
The ``satellite_0_thermalmapping135-datetime`` is your **TGM_model_folder_name**.

2. To generate thermal images from extended_database.h5, run one of the following scripts:

```
./eval_satellite_translation.sh   TGM_model_folder_name # Best choice
./eval_satellite_translation_nocontrast.sh   TGM_model_folder_name
```

After running this scripts, you can find a file **train_queries.h5** in ``./test/default/model_folder_name/satellite_0_thermalmapping135-datetime``  
Rename it to **extended_queries.h5** and move it to ``./datasets/satellite_0_thermalmapping135``.

3. Rename the dataset folder to one of the following:

```
./datasets/satellite_0_thermalmapping_135/                  #./scripts/train_bing_thermal_translation_10.sbatch
./datasets/satellite_0_thermalmapping_135_100/              #./scripts/train_bing_thermal_translation_100.sbatch
./datasets/satellite_0_thermalmapping_135_nocontrast/       #./scripts/train_bing_thermal_translation_10_nocontrast.sbatch
./datasets/satellite_0_thermalmapping_135_100_nocontrast/    #./scripts/train_bing_thermal_translation_100_nocontrast.sbatch
```

The dataset name depends on the training script you use, which is listed on the right of each dataset name. This is to avoid confusion about which TGM setting is used to generate.

### SGM training

To train Satellite-thermal Geo-localization Module (SGM), use one of the scripts in ``./scripts`` per name, for example:

```
./scripts/train_bing_thermal_partial_conv4096_bn_lr_wd_DANN_after_ld_dp_contrast_extended_100.sbatch # Best choice
```

After training SGM, find your model folder in ``./logs/default/satellite_0_thermalmapping135-datetime-uuid``  
The ``satellite_0_thermalmapping135-datetime-uuid`` is your **SGM_model_folder_name**.

## Evaluation
To evaluate SGM, use one of the following scripts:
```
./scripts/eval.sbatch   SGM_model_folder_name   backbone_name
./scripts/eval_nocontrast.sbatch   SGM_model_folder_name   backbone_name
```
Note that running evaluation scripts requires two arguments: **SGM_model_folder_name** and **backbone_name**. The typical backbone we use is **resnet18conv4**.

Find the test results in in ``./test/default/model_folder_name/satellite_0_thermalmapping135-datetime``. There will be two folders. This first reports **R@1** and **R@5**. The second reports **R_512@1**, **R_512@5** and **L^512_2**.

## Acknowledgement
Our implementation refers to the following repositories and appreciate their excellent work.

https://github.com/gmberton/deep-visual-geo-localization-benchmark  
https://github.com/fungtion/DANN  
https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
