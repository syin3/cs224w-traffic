counter=0
for var in "$@"
do
	file=$var
	# echo $counter
	python set-up.py --graph_embed_file $file
	python dcrnn_train_para.py --config_filename=data/model/dcrnn_config.yaml --gpu_id $counter
	counter=$[$counter+1]

done