Document classification using Probabilistic Index
======

This is the official repository for the code and data for the "Open Set Classification of Untranscribed Handwritten Text Image Documents" paper.

Requirements
===========
To use this repository use anaconda with the following requirements file.

```
conda create --name <env> --file requirements.txt
```

Usage
=====

First, we will have to create the Information Gain and TfIdf files. To do this we can use the ready-made launchers in the launcher folder. There is one on IG-TfIdf for each experiment in the original paper.

* The PrIx used in the paper are in the PrIx folder.

We will give as an example how to launch the CSC experiment with PrIx.

## Obtaining the input files

```
cd launchers
./launch_IG_tfidf_JMBD4949_4950_LOO_groups.sh
```

You will see that within these files, the scripts for IG and TfIdf are executed. They have the parameters indicated in the original paper. 
However, you can test them by modifying the pruning probability with the variable "prob", modifying the classes to use, etc.
In addition, there is a sample file of how to obtain the "production" probabilities, which would be the unseen classes (since IG uses the classes).
The difference between the "REJECT" class and the others is that the scripts use "--all_files True", which uses the unassigned classes as the "REJECT" class.


> In order to use the transcriptions use the files endes with _trans.sh. Also, to use the "REJECT" class use the files that contains the "other" word in the name.

## Training the models

The launchers are prepared to use the previously created TfIdf files, which are already sorted by IG.
The bash scripts are prepared for Leaving One Out (LOO True) experiments.
Also, to run different models from the "*ngfs*" list.

> ngfs=(0 128 128,128)  means for MLP0, MLP1 and MLP2 respectively

The *numfeats* list will create the number of input features to the model. If we set *"seq 9 11"* it will train three models, with 512, 1024 and 2048 input features respectively.
If we set "seq 5 14" and "*ngfs=( 0 128 128,128 )*" we will reproduce all the experiments in the paper.


```
./launch_JMBD4949_4950_LOO_groups.sh
```

> In order to execute the MLP0, remember to change the *lr* to *0.1*

In the folder *works_JMBD4949_4950_loo_groups* we will have the results of all models. It will create a *results.txt* for every specific model with all of the posterior probabilities.
We can use the script "utils/get_acc.py" in order to get the errors classification rate by modifying it a little bit the folders string.


Usage
=====
GNU General Public License v3.0
See [LICENSE](LICENSE) to see the full text.

Acknowledgments
===============
This code has been developed with the help of @JuanjoFlores.