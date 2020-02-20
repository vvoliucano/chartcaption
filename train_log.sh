

python toy_create_input.py

python train.py --data_folder /Users/tsunmac/vis/projects/autocaption/data/karpathy_output --data_name flickr8k_5_cap_per_img_5_min_word_freq

python train.py --data_folder /home/can.liu/caption/data/try/try_output --data_name flickr8k_5_cap_per_img_0_min_word_freq

python train.py --data_folder /home/can.liu/caption/data/svg_try/try_output --data_name flickr8k_5_cap_per_img_0_min_word_freq --image_type svg


# 训练假数据
python train.py --data_folder /home/can.liu/caption/data/svg_try/try_output --data_name flickr8k_5_cap_per_img_0_min_word_freq --image_type svg



# 尝试svg位的情况
python caption.py --img asf  --model BEST_checkpoint_flickr8k_5_cap_per_img_0_min_word_freq.pth.tar --word_map /home/can.liu/caption/data/try/try_output/WORDMAP_flickr8k_5_cap_per_img_0_min_word_freq.json  --image_type svg


# 尝试像素位的情况
python caption.py --img /home/can.liu/caption/data/coco_2014/val2014/COCO_val2014_000000340047.jpg  --model BEST_checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar --word_map /home/can.liu/caption/data/karpathy_output/WORDMAP_coco_5_cap_per_img_5_min_word_freq.json  --image_type pixel

python train.py --data_folder ../data/svg_try/try_output --data_name flickr8k_5_cap_per_img_5_min_word_freq --image_type svg

python train.py --data_folder ../data/svg_try/try_output --data_name chart_5_cap_0_min_wf --image_type svg

