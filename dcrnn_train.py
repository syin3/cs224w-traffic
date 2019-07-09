from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import tensorflow as tf
import yaml

from model.dcrnn_supervisor import DCRNNSupervisor


def main(args):
    
    with open(args.config_filename) as f:
        supervisor_config = yaml.load(f)

        tf_config = tf.ConfigProto()
        # if args.use_cpu_only:
        #     tf_config = tf.ConfigProto(device_count={'GPU': 0})
        tf_config.gpu_options.allow_growth = True
        with tf.Session(config=tf_config) as sess:
            supervisor = DCRNNSupervisor(**supervisor_config)

            supervisor.train(sess=sess)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_filename', required=True, default=None, type=str, help='Configuration filename for restoring the model.')
    parser.add_argument('--use_cpu_only', default=False, type=bool, help='Set true to only use cpu.')
    # adjacent and distance-weighted
    parser.add_argument('--weightType', required=True, choices=['a', 'd'], help='w/ or w/o distance pre-processing')
    parser.add_argument('--att', dest='attention', action='store_true', help='Call this command to raise attention mechanism in the training.')
    parser.add_argument('--no-att', dest='attention', action='store_false', help='Call this command not to raise attention mechanism in the training.')
    parser.set_defaults(attention=False)
    
    subparsers = parser.add_subparsers()

    fullyConnectParser = subparsers.add_parser('fc', help='In fully connect mode, choose embed file')
    fullyConnectParser.add_argument('--gEmbedFile', required=True, default='LA-n2v-14-0.1-1', help='Embedding file for n2v, should add up-directory when calling')
    fullyConnectParser.add_argument('--network', nargs='?', const='fc', default='fc', help='To store the choice of fully connected')

    graphConvParser = subparsers.add_parser('graphConv', help='In graph conv mode, choose W matrix form')
    graphConvParser.add_argument('--hop', required=True, type=int, default=2, 
        help='k-hop neighbors, default is 2 for distance-processed matrix; but must be one for binary matrix')
    graphConvParser.add_argument('--network', nargs='?', const='gconv', default='gconv', help='To store the choice of gconv')

    args = parser.parse_args()
    
    with open(args.config_filename) as f:
        doc = yaml.load(f)

    # default batch sizes to 64, in training, validation and in testing
    doc['data']['batch_size'] = 64
    doc['data']['test_batch_size'] = 64
    doc['data']['val_batch_size'] = 64

    # set matrix to adjacency or distance-weighted
    if args.weightType == 'd':
        doc['data']['graph_pkl_filename'] = "data/sensor_graph/adj_mx_la.pkl"
    else:
        doc['data']['graph_pkl_filename'] = "data/sensor_graph/adj_bin_la.pkl"
    
    # record necessary info to log
    doc['model']['weightMatrix'] = args.weightType
    doc['model']['attention'] = args.attention
    doc['model']['network'] = args.network
    if 'gEmbedFile' in vars(args):
        doc['model']['graphEmbedFile'] = args.gEmbedFile
    doc['model']['max_diffusion_step'] = 0
    if 'hop' in vars(args):
        doc['model']['max_diffusion_step'] = args.hop
    
    # save the info
    with open(args.config_filename, 'w') as f:
        yaml.dump(doc, f)

    main(args)
