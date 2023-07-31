# TopoAI: Topological Approaches for Improved Structural Correctness

## Team
| Name                 | Email               | Github      |
| -------------------- | ------------------- | ----------- |
| Alexander Spiridonov | aspiridonov@ethz.ch | aspiridon0v |
| Alexander Veicht     | veichta@ethz.ch     | veichta     |
| András Strausz       | strausza@ethz.ch    | strausza    |
| Richard Danis        | richdanis@ethz.ch   | richdanis   |
## Setup
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install git+https://github.com/bruel-gabrielsson/TopologyLayer.git
pip install -e .
```
## Data
| Name       | URL                                                            | #images |
| ---------- | -------------------------------------------------------------- | ------- |
| CIL        | https://www.kaggle.com/competitions/cil-road-segmentation-2022 | 144     |
| EPFL       | https://www.aicrowd.com/challenges/epfl-ml-road-segmentation   | 339     |
| RoadTracer | https://paperswithcode.com/dataset/roadtracer                  | 4976    |

The preprocessed data can be downloaded using the following command:
```bash
wget https://polybox.ethz.ch/index.php/s/KhsD19D0iLEmyTH/download
```

The data is expected to have the following structure:
```
data
    ├── images
    │   ├── 000000_cil.jpg
    │   ├── 000000_epfl.jpg
    │   ├── 000000_roadtracer.jpg
    │   ├── 000001_cil.jpg
    │   ├── ...
    ├── masks
    │   ├── 000000_cil.png
    │   ├── 000000_epfl.png
    │   ├── 000000_roadtracer.png
    │   ├── 000001_cil.png
    │   ├── ...
    └── weights
        ├── 000000_cil.png
        ├── 000000_epfl.png
        ├── 000000_roadtracer.png
        ├── 000001_cil.png
        ├── ...
```
This can be achieved by running the following commands:
```bash
unzip data.zip
rm data.zip
```
### Splitting the Data
In order to split the data into train, val and test sets, run the following command:
```bash
python src/preprocessing/split_dataset.py --dataset data
```
This will create folder for each split containing ```images```, ```masks``` and ```weights``` folders.

## Training
The training can be started by running the following command:
```bash
python main.py --datasets <list of dataset> --model <model name> --device <device>
```
For a full list of arguments, run:
```bash
python main.py --help
```

### Reproducing the Baseline
The baseline results can be reproduced by running the following commands:
```bash
python main.py --data_path data --datasets cil --epochs 300 --lr 0.001 --model unet++ --patience 40
```
```bash
python main.py --data_path data --datasets cil --epochs 300 --lr 0.001 --model spin --patience 40
```
```bash
python main.py --data_path data --datasets cil --epochs 300 --lr 3e-4 --model upernet-t --patience 40 --miou_weight 1 --focal_weight 1 --mse_weight 1
```