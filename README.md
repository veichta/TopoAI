# DiffusionRoads

## Team
| Name             | Email | Github |
| ---------------- | ----- | ------ |
| András Strausz   | -     | -      |
| Alexander Veicht | -     | -      |
## Setup
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```
## Data
| Name          | URL                                                                        | #images |
| ------------- | -------------------------------------------------------------------------- | ------- |
| CIL           | https://www.kaggle.com/competitions/cil-road-segmentation-2022             | -       |
| EPFL          | https://www.aicrowd.com/challenges/epfl-ml-road-segmentation               | -       |
| DeepGlobe     | https://www.kaggle.com/datasets/balraj98/deepglobe-road-extraction-dataset | -       |
| Massachusetts | https://www.kaggle.com/datasets/balraj98/massachusetts-roads-dataset       | -       |
| RoadTracer    | https://paperswithcode.com/dataset/roadtracer                              | -       |
| Topo-Boundary | https://tonyxuqaq.github.io/projects/topo-boundary/                        | -       |

The preprocessed data can be downloaded using the following command:
```bash
wget https://polybox.ethz.ch/index.php/s/mPtk999prM80ACx/download
```