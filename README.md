![header](imgs/header.png)

# TankBind
TankBind could predict both the protein-ligand binding structure and their affinity.
If you have any question or suggestion, please feel free to open an issue or email me at [wei.lu@galixir.com](wei.lu@galixir.com).

## Installation
````
conda create -n tankbind_py38 python=3.8
conda activate tankbind_py38
````
You might want to change the cudatoolkit version based on the GPU you are using.:
````
conda install pytorch cudatoolkit=11.3 -c pytorch
````

````
conda install torchdrug pyg biopython nglview -c milagraph -c conda-forge -c pytorch -c pyg
pip install torchmetrics tqdm mlcrate pyarrow
````

p2rank v2.3 could be installed by following the instruction here:

https://github.com/rdk/p2rank/releases/download/2.3/p2rank_2.3.tar.gz


## Prediction
To predict the drug-protein binding structure, check out 

    examples/prediction_example_using_PDB_6hd6.ipynb


## Training and test dataset construction
To process the raw PDBbind data into training and test set, check out 

    examples/construction_PDBbind_training_and_test_dataset.ipynb

## High-throughput virtual screening
TankBind also support virtual screening. In our example here, for the WDR domain of LRRK2 protein, we can screen 10,000 drug candidates in 2 minutes (or 1M in around 3 hours). Check out

    examples/high_throughput_virtual_screening_LRRK2_WDR.ipynb

