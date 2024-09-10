# Adjoint Method-based Fourier Neural Operator Surrogate Solver for Wavefront Shaping in Tunable Metasurfaces



## Data preparation
Our metalens dataset used in the paper can be accessed on [GoogleDrive](https://drive.google.com/file/d/1pFgmGAeW2CK0LA71t5rx3jm_Qy1--ah-/view?usp=sharing).

The train-test data should be placed in data directory. :
```
(repository)
└───data
      └───dataset
            ├───train
            └───valid

```



## Training
After preparing the dataset in data/DRMI_dataset directory, use
```
python fno_train.py
```
for our model,
or
```
python train.py
```
for comparison group.

To change the settings for training, modify arguments in train python files.

## Evaluation
```
python iteration.py
```
You should set the log path in the iteration python file in order to evaluate.