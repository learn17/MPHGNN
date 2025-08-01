# MPHGNN
This document is intended for reproducing the experiments. It includes environment setup, dataset download, and startup scripts.

## Requirements
+ Linux
+ Python 3.9
+ torch==2.3.1+cu118
+ torchmetrics==1.7.1
+ tqdm==4.67.1
+ dgl==2.4.0+cu118
+ ogb==1.3.6
+ shortuuid==1.0.13
+ pandas==2.2.3
+ gensim==4.3.3
+ numpy==1.26.4

## Datasets
### HGB  (DBLP, IMDB, ACM and Freebase)
Use the following command to download:
```shell
sh download_hgb_datasets.sh 
```
After downloading, place the files into the ./datasets folder.

### OGBN-MAG
Manual download is not required. Simply run the corresponding script, and the code will automatically retrieve and load the dataset via the ogb library.

### OAG-Venue and OAG-L1-Field
The construction process of these two datasets follows the data preparation procedure of the NARS project. Detailed instructions can be found on its GitHub page:
https://github.com/facebookresearch/NARS/tree/main/oag_dataset

After generating the *.pk and *.npy files, move them to ./datasets/nars_academic_oag/ and rename graph_field.pk to graph_L1.pk.

## Running

Execute the corresponding script for each dataset as shown below:
```shell
bash scripts/run_DBLP.sh
bash scripts/run_IMDB.sh
bash scripts/run_ACM.sh
bash scripts/run_Freebase.sh
bash scripts/run_OGBN-MAG.sh
bash scripts/run_OAG-Venue.sh
bash scripts/run_OAG-L1-Field.sh
```
