cd data_generator/before_data/
pwd
python feature_data_generator.py --path ../../data/$1 --number $2 --node_name $3
cd ../../
pwd

# no focus
python create_input_files.py --dataset chart --karpathy_json_path data/$1/karparthy_dataset.json --image_folder data/$1/svg --output_folder data/$1/deal_no_focus --image_type svg --need_text --max_element_number 100 
# python train.py --data_folder data/$1/deal_no_focus --svg_element_number 100 --data_name chart_5_cap_5_min_wf --image_type svg --input_nc 3,2,4,3,1 --output_nc 5,5,5,5,5 --emb_dim 512 --attention_dim 512 --decoder_dim 512 --need_text 

# with focus
# python create_input_files.py --dataset chart --karpathy_json_path data/$1/karparthy_dataset.json --image_folder data/$1/svg --output_folder data/$1/deal --image_type svg --need_text --max_element_number 100 --with_focus
# python train.py --data_folder data/$1/deal --svg_element_number 100 --data_name chart_5_cap_5_min_wf --image_type svg --input_nc 3,2,4,3,1,1 --output_nc 5,5,5,5,5,5 --emb_dim 512 --attention_dim 512 --decoder_dim 512 --need_text 

