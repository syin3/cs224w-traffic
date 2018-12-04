# Traffic Forecasting with Graph Embedding

This project inherits the main structure of DCRNN described in the following paper: \
Yaguang Li, Rose Yu, Cyrus Shahabi, Yan Liu, [Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting](https://arxiv.org/abs/1707.01926), ICLR 2018.

More specifically, we replaced graph convolution with a fully-connected layer. We also added graph embeddings from *node2vec* and *SDNE* as invariant features to the input matrix.

## 1, Graph Construction
Current implementation currently only supports sensor ids in Los Angeles (see `data/sensor_graph/sensor_info_201206.csv`). We prepare the graph data as follows:

```bash
python -m scripts.gen_adj_mx.py  --sensor_ids_filename=data/sensor_graph/graph_sensor_ids.txt --normalized_k=0.1\
--output_pkl_filename=data/sensor_graph/adj_mx.pkl
```
In file `ProduceEdgeList.ipynb`, we inspect the dimension, directedness and other features of the adjacency matrix of LA highway sensor system.

## 2, Traffic Data Preparation
The traffic data file for Los Angeles, i.e., `df_highway_2012_4mon_sample.h5`, is available [here](https://drive.google.com/open?id=1tjf5aXCgUoimvADyxKqb-YUlxP8O46pb), and should be
put into the `data/METR-LA` folder.
Besides, the locations of sensors are available at [data/sensor_graph/graph_sensor_locations.csv](https://github.com/liyaguang/DCRNN/blob/master/data/sensor_graph/graph_sensor_locations.csv).
```bash
python -m scripts.generate_training_data --output_dir=data/METR-LA
```
The generated train/val/test dataset will be saved at `data/METR-LA/{train,val,test}.npz`.

## 3, Graph Embedding Data Preparation
Follow the instructions in file `embeddings.ipynb`, we produce graph embeddings of designated dimension, which later is attached to the input feature matrix and is then fed to the fully connected neural network.

## 4, Model Training
```bash
python dcrnn_train.py --config_filename=data/model/dcrnn_config.yaml
```
Each epoch takes about 5min with a single GTX 1080 Ti (on DCRNN by author of paper). Out running time on AWS Tesla K80 takes 15 minues for each epoch (DCRNN, by us).

On AWS Tesla K80, each epoch only takes 1-2 min (at least 90% faster, on FCRNN, by us).

## 5, Model evaluation

Evaluate the trained models using `run_demo.py`. Please notice you need to adjust files, directories personally.

```bash
python run_demo.py
```

## 6, Visualization

See `plot.ipynb` for the visualization of results.

## 7, Please cite original paper as specified by authors

If you find this repository useful in your research, please cite the following paper:
```
@inproceedings{li2018dcrnn_traffic,
title={Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting},
author={Li, Yaguang and Yu, Rose and Shahabi, Cyrus and Liu, Yan},
booktitle={International Conference on Learning Representations (ICLR '18)},
year={2018}
}
```
