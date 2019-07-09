from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import numpy as np
import os
import pandas as pd

# the dimension of matrices is controled by upstream Jupyter Notebook
# When d = 128, it is consuming to much memory of my 16G memory mbp
# let's try some value smaller
# node2vec = np.loadtxt('n2v-LA.txt')
# n2vMatrix = np.repeat(node2vec[np.newaxis, :, :], 12, axis=0)
# print("Now input_dim should be {}".format(node2vec.shape[1]+2))

# 34249*12*207*32*4/10**9 = 10.9
# Storing a training matrix of our size would consume 10.9G memory

def generate_graph_seq2seq_io_data(
        df, x_offsets, y_offsets, add_time_in_day=True, add_day_in_week=False, scaler=None
):
    """
    Generate samples from
    :param df:
    :param x_offsets:
    :param y_offsets:
    :param add_time_in_day:
    :param add_day_in_week:
    :param scaler:
    :return:
    # x: (epoch_size, input_length, num_nodes, input_dim)
    # y: (epoch_size, output_length, num_nodes, output_dim)
    """

    num_samples, num_nodes = df.shape
    # print(df.shape)
    # (34272, 207)
    # print(df.head())
    data = np.expand_dims(df.values, axis=-1)
    data_list = [data]

    # print(data_list[0].shape)
    if add_time_in_day:
        time_ind = (df.index.values - df.index.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
        time_in_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
        data_list.append(time_in_day)
    if add_day_in_week:
        day_in_week = np.zeros(shape=(num_samples, num_nodes, 7))
        day_in_week[np.arange(num_samples), :, df.index.dayofweek] = 1
        data_list.append(day_in_week)

    # print(time_in_day.shape)
    # print(time_in_day[10,:,:])

    # the same time step was in for each row.each
    # after all, the data are recorded at the same time
    # that's why we have input dimension 2
    data = np.concatenate(data_list, axis=-1)
    # print(data.shape)
    # (34272, 207, 2)
    # epoch_len = num_samples + min(x_offsets) - max(y_offsets)
    x, y = [], []
    # t is the index of the last observation.
    min_t = abs(min(x_offsets))
    max_t = abs(num_samples - abs(max(y_offsets)))  # Exclusive
    # print(min_t, max_t)
    # 11 34260

    # then since we need to make room for 12 steps before and behind, we cannot start from 0
    # print(n2vMatrix.shape)
    # (12, 207, 128)
    # global n2vMatrix
    for t in range(min_t, max_t):
        x_t = data[t + x_offsets, ...]
        y_t = data[t + y_offsets, ...]
        # print(x_t.shape)
        # x_t = np.dstack((x_t, n2vMatrix))
        # y_t = np.dstack((y_t, n2vMatrix))
        x.append(x_t)
        y.append(y_t)
        # print(x_t.shape)
        # print(x_t.shape)
        # (12, 207, 2)
        # Every time step has its own x_t (the training data)
        # 12: time steps looking back
        # 207: node of nodes
        # 2: input dimension, speed and time
        # Every slice on axis=0, we have (1, 207, 2), one training period for the time step
        # And there are 12 of them
        # So every step in 34249 has 12 before (x) and 12 beyond (y)
        # Therefore are in total 34239*12 data logs, each log has (207*2)
        # exit()
    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)

    # x shape: (34249, 12, 207, 2), y shape: (34249, 12, 207, 2)
    # 34249 is the number of training samples we can use for training
    # 12 is input length
    # 207 is the number of nodes in the network
    # 2 is the input dimension: traffic speed reading and associated recording time
    return x, y


def generate_train_val_test(args):
    df = pd.read_hdf(args.traffic_df_filename)
    # 0 is the latest observed sample.
    x_offsets = np.sort(
        # np.concatenate(([-week_size + 1, -day_size + 1], np.arange(-11, 1, 1)))
        np.concatenate((np.arange(-11, 1, 1),))
    )
    # Predict the next one hour
    y_offsets = np.sort(np.arange(1, 13, 1))
    # x: (num_samples, input_length, num_nodes, input_dim)
    # y: (num_samples, output_length, num_nodes, output_dim)

    # currect observed value is indexed 0
    # look back 11 steps to gather 12 steps of information
    # predict 12 steps in the future
    x, y = generate_graph_seq2seq_io_data(
        df,
        x_offsets=x_offsets,
        y_offsets=y_offsets,
        add_time_in_day=True,
        add_day_in_week=False,
    )

    print("x shape: ", x.shape, ", y shape: ", y.shape)
    # exit()

    # Write the data into npz file.
    # num_test = 6831, using the last 6831 examples as testing.
    # for the rest: 7/8 is used for training, and 1/8 is used for validation.
    num_samples = x.shape[0]
    num_test = round(num_samples * 0.2)
    num_train = round(num_samples * 0.7)
    num_val = num_samples - num_test - num_train

    # train
    x_train, y_train = x[:num_train], y[:num_train]
    # val
    x_val, y_val = (
        x[num_train: num_train + num_val],
        y[num_train: num_train + num_val],
    )
    # test
    x_test, y_test = x[-num_test:], y[-num_test:]
    print(x_test.shape)
    # (6850, 12, 207, 2)

    for cat in ["train", "val", "test"]:
        _x, _y = locals()["x_" + cat], locals()["y_" + cat]
        print(cat, "x: ", _x.shape, "y:", _y.shape)
        np.savez_compressed(
            os.path.join(args.output_dir, "%s.npz" % cat),
            x=_x,
            y=_y,
            x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]),
            y_offsets=y_offsets.reshape(list(y_offsets.shape) + [1]),
        )
    # so the output is (n*12*207*2)
    # y is the 12 steps of future we want to predict


def main(args):
    print("Generating training data")
    generate_train_val_test(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir", type=str, default="data/", help="Output directory."
    )
    parser.add_argument(
        "--traffic_df_filename",
        type=str,
        default="data/METR-LA/df_highway_2012_4mon_sample.h5",
        help="Raw traffic readings.",
    )
    args = parser.parse_args()
    main(args)
