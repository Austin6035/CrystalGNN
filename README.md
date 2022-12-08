### Introduction
CrystalGNN: A Self-Learning-Input Graph Neural Network for materials properties prediction

### Requirements:

python3

PyTorch 1.9.0-cu10.1

PyTorch Geometric 2.0.3

Obabel

Pymatgen

### How to install:

- git clone http://ip:port/xxx

- unzip CrystalGNN.zip

### How to prepare data:
It is necessary to place the structure files of the materials and the files of the materials properties("id_prop.csv") in the same folder, and then place them in the folder "dataset".

**note：**id_prop.csv, the 'material_id' in the first column must be corresponding to the filename of the structure file. Accordingly, the structure file in a dataset should be in the form of the same file type(xyz, cif, mol, sdf), and CrystalGNN will identify the suffixes of the structure file.

```
|- dataset
	|- custom_dataset
		|- xxx.xyz
		|- ...
		|- id_prop.csv
```

### How to train model:

```shell
python train.py dataset/custom_dataset
```
You can access the settings of parameters in the "train.py", or passed through the command line.

```shell
python train.py dataset/custom_dataset optim=Adam
```

### How to predict:
To predict new materials, you should specify the pre-trained model, and locat it in the folder "weight".

```shell
python predict.py weight/pretrained_model.pth.tar custom_dataset
```

### Notes:
The data related to "Accelerate the Discovery of Metastable IrO2 for the Oxygen Evolution Reaction by Self-Learning-Input Graph Neural Network" can be accessed in the folder "dataset/article_data", "results/article_results",and  "weight/article_models", whose contents are listed below:
a."C2DB dataset"--The C2DB dataset, its corresponding model, and its training results.
b."MP dataset"--The MP dataset, its corresponding model, and its training results.
c."IrO2 dataset"--The IrO2 dataset, its corresponding model, and its training results of the 3000 initail structures(n = 1, 2),as well as  its predicting results of the 7000 complex structures(n = 3, 4, 5, 6, 7).
d."RuO2 dataset"--The RuO2 dataset, its corresponding model, and its training results of the 3000 initail structures(n = 1, 2).
e."MnO2 dataset"--The MnO2 dataset, its corresponding model, and its training results of the 3000 initail structures(n = 1, 2). 