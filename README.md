# visual-thermal-localization

This is the official repository for Long-range UAV Thermal Geo-localization with Satellite Imagery.

The ``dataset`` folder should be created in the root folder with the following structure

```
datasets/satellite_0_thermalmapping_135/
├── test_database.h5
├── test_queries.h5
├── train_database.h5
├── train_queries.h5
├── extended_database.h5
├── extended_queries.h5
├── val_database.h5
└── val_queries.h5
```

You can find the training scripts and evaluation scripts in ``scripts`` folder. The scripts is for slurm system to submit sbatch job. If you want to run bash command, change the suffix from ``sbatch`` to ``sh`` and run with bash.

Note that running evaluation scripts requires two arguments: ```model_folder_name``` and ```backbone_name```. An example command is:
```
sbatch ./scripts/eval_contrast.sbatch  satellite_0_thermalmapping_135-2023-02-18_05-18-21-c0c572c4-a48a-4d5b-a152-bbc51b9161aa resnet18conv4
```
Or the bash command
```
./scripts/eval_contrast.sh  satellite_0_thermalmapping_135-2023-02-18_05-18-21-c0c572c4-a48a-4d5b-a152-bbc51b9161aa resnet18conv4
```

Running training scripts does not require arguments.
