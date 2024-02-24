# DrugMGR: a deep bioactive molecule binding method to identify compounds targeting proteins
Motivation: Understanding the intermolecular interactions of ligand-target pairs is key to guiding the optimization of drug research on cancers, which can greatly mitigate overburden workloads for wet labs. Several improved computational methods have been introduced and exhibit promising performance for these identification tasks, but some pitfalls restrict their practical applications. (i)First, existing methods do not sufficiently consider how multigranular molecule representations influence interaction patterns between proteins and compounds. (ii)Second, existing methods seldom explicitly model the binding sites when an interaction occurs to enable better prediction and interpretation, which may lead to unexpected obstacles to biological researchers.
Results: To address these issues, we here present DrugMGR, a deep multigranular drug representation model capable of predicting binding affinities and regions for each ligand-target pair. We conduct consistent experiments on three benchmark datasets using existing methods and introduce a new specific dataset to better validate the prediction of binding sites. For practical application, target-specific compound identification tasks is also carried out to validate the capability of real-world compound screen. Moreover, the visualization of some practical interaction scenarios provides interpretable insights from the results of the predictions. The proposed DrugMGR achieves excellent overall performance in these datasets, exhibiting its advantages and merits against state-of-the-art methods. Thus, the downstream task of DrugMGR can be fine-tuned for identifying the potential compounds that target proteins for clinical treatment.

## Architecture
<p align="center">
<img src="https://github.com/lixiaokun2020/DrugMGR/blob/main/DrugMGR.jpg" align="middle" height="80%" width="80%" />
</p>


## The environment of DrugMGR
```
python==3.7.16
numpy==1.19.2
pandas==1.3.5
torch==1.6.0
rdkit==2022.9.5
scikit-learn==1.0.2
torch_geometric==1.7.0
scipy==1.7.3
subword-nmt==0.3.8
networkx==2.6.3
dgl==0.4
```

## Dataset description
In this paper, three benchmark datasets are used, i.e., BindingDB, Davis and KIBA. The directory structure are shown below:

```txt
data
|-- BindingDB
|   |-- merged_data.csv
|   |-- train_kd.csv
|   |-- val_kd.csv
|   |-- test_kd.csv
|
|--Davis
|   |-- folds
|   |   |-- train_fold_setting1.txt
|   |   |-- test_fold_setting1.txt
|   |
|   |-- ligands_can.txt
|   |-- proteins.txt
|   |-- Y
|
|--KIBA
|   |-- folds
|   |   |-- train_fold_setting1.txt
|   |   |-- test_fold_setting1.txt
|   |
|   |-- ligands_can.txt
|   |-- proteins.txt
|   |-- Y

```

We also introduce a new dataset PDBbind to evaluate the performance of binding region prediction, and the'ESPF' file that contains sub-structure information is provided.

## Run the DrugMGR for protein-ligand binding affinity task
At first, create the data 
```sh
python creat_data.py
```
and run it with the Davis (default) dataset:
```sh
python main.py
```

# Attention visualization
If you users want to introduce the attention score in their own tools, please see the attention_demo.py file for a simple tutorial.

# Acknowledgments
The authors sincerely hope to receive any suggestions from you!


