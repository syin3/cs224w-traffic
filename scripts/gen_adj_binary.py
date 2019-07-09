from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import pandas as pd
import pickle


def get_adjacency_matrix(distDF, sensorIDs):
    """

    :param distDF: data frame with three columns: [from, to, distance].
    :param sensorIDs: list of sensor ids.
    :return:

    difference between pre-processing with distance: only consider connectivity, not aware of distance;
    do not re-calculate W matrix, do not zero out for low values
    """
    # number of sensors
    numSensors = len(sensorIDs)
    # adjacent matrix
    A = np.zeros((numSensors, numSensors), dtype=np.float32)
    
    # Builds sensor id to index map
    sensorID2Ind = {}
    for i, sensorId in enumerate(sensorIDs):
        sensorID2Ind[sensorId] = i

    # Fills cells in the matrix with binary 1/0
    # used to fill the values with row[2], the actual distance
    for row in distDF.values:
        if row[0] not in sensorID2Ind or row[1] not in sensorID2Ind:
            continue
        A[sensorID2Ind[row[0]], sensorID2Ind[row[1]]] = 1

    return sensorID2Ind, A


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sensor_ids_filename', type=str,
                        help='File containing sensor ids separated by comma.')
    parser.add_argument('--distances_filename', type=str, default='data/sensor_graph/distances_la_2012.csv',
                        help='CSV file containing sensor distances with three columns: [from, to, distance].')
    parser.add_argument('--output_pkl_filename', type=str, help='Path of the output file.')
    args = parser.parse_args()

    with open(args.sensor_ids_filename) as f:
        sensorIDs = f.read().strip().split(',')
    distDF = pd.read_csv(args.distances_filename, dtype={'from': 'str', 'to': 'str'})

    sensorID2Ind, A = get_adjacency_matrix(distDF, sensorIDs)
    
    # Save to pickle file.
    with open(args.output_pkl_filename, 'wb') as f:
        pickle.dump([sensorIDs, sensorID2Ind, A], f, protocol=2)
