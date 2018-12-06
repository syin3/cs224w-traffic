file=$1
python set-up.py --graph_embed_file $file
python dcrnn_train.py --config_filename=data/model/dcrnn_config.yaml