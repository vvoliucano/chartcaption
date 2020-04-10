

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

python caption.py --img ../data/coco_2014/val2014/COCO_val2014_000000340047.jpg  --model checkpoint/chart_5_cap_0_min_wf/Best.pth.tar --word_map ../data/svg_try/try_output/WORDMAP_chart_5_cap_0_min_wf.json  --image_type svg


# 创建新的数据创建器

python create_input_files.py --dataset chart --karpathy_json_path ../data/real_svg/template.json --image_folder ../data/real_svg/svg --output_folder ../data/real_svg/output --image_type svg --min_word_freq 0

# 试试真正的svg的训练

python train.py --data_folder ../data/real_svg/output --data_name chart_5_cap_5_min_wf --image_type svg

python train.py --data_folder ../data/real_svg/output --data_name chart_5_cap_0_min_wf --image_type svg

# 试试真正的svg的测试

python caption.py --img ../data/real_svg/svg/1.svg  --model checkpoint/chart_5_cap_5_min_wf/epoch_119.pth.tar --word_map ../data/real_svg/output/WORDMAP_chart_5_cap_5_min_wf.json  --image_type svg

# 吴聪生成的第一版的数据集

# python create_input_files.py --dataset chart --karpathy_json_path data_generator/dataset.json --image_folder ./data_generator/svg --output_folder data/svg_output --image_type svg

python create_input_files.py --dataset chart --karpathy_json_path data_generator/svg/dataset.json --image_folder ./data_generator/svg --output_folder data/svg_output_20200305 --image_type svg 

python train.py --data_folder data/svg_output_20200305 --data_name chart_5_cap_5_min_wf --image_type svg

python caption.py --img ./data_generator/svg/117.svg  --model checkpoint/chart_5_cap_5_min_wf/Best.pth.tar --word_map data/svg_output/WORDMAP_chart_5_cap_5_min_wf.json  --image_type svg

python caption.py --img ./data_generator/svg/93.svg  --model checkpoint/chart_5_cap_5_min_wf/Best.pth.tar --word_map data/svg_output/WORDMAP_chart_5_cap_5_min_wf.json  --image_type svg


# python caption.py --img ../data/real_svg/svg/1.svg  --model checkpoint/chart_5_cap_5_min_wf/epoch_119.pth.tar --word_map ../data/real_svg/output/WORDMAP_chart_5_cap_5_min_wf.json  --image_type svg


# 20200305

python create_input_files.py --dataset chart --karpathy_json_path data_generator/svg/dataset.json --image_folder ./data_generator/svg --output_folder data/svg_output_20200305 --image_type svg 

python train.py --data_folder data/svg_output_20200305 --data_name chart_5_cap_5_min_wf --image_type svg

python caption.py --img ./data_generator/svg/117.svg  --model checkpoint/chart_5_cap_5_min_wf/Best.pth.tar --word_map data/svg_output_20200305/WORDMAP_chart_5_cap_5_min_wf.json  --image_type svg

python train.py --data_folder data/svg_output_20200305 --data_name chart_5_cap_5_min_wf --image_type svg --pretrained_model Best.pth.tar --max_epoch 1000

# 下一个版本的数据；

# 具有多个

# 20200325

#生成数据集
python3 gen_sent2.py -n 50000 -p svg2

# 数据集格式转化
python create_input_files.py --dataset chart --karpathy_json_path data_generator/svg2/dataset.json --image_folder ./data_generator/svg2 --output_folder data/svg_output_20200326 --image_type svg 

# 训练模型
python train.py --data_folder data/svg_output_20200326 --data_name chart_5_cap_5_min_wf --image_type svg

# 使用模型
python caption.py --img ./data_generator/svg/2.svg  --model checkpoint/chart_5_cap_5_min_wf-2020-03-26-10-56/Best.pth.tar --word_map data/svg_output_20200326/WORDMAP_chart_5_cap_5_min_wf.json  --image_type svg

# 使用模型on dl
python caption.py --img ./data_generator/svg2/2.svg  --model checkpoint/chart_5_cap_5_min_wf-2020-03-26-13-05/Best.pth.tar --word_map data/svg_output_20200326/WORDMAP_chart_5_cap_5_min_wf.json  --image_type svg


# 20200328
python create_input_files.py --dataset chart --karpathy_json_path data_generator/svg2/dataset.json --image_folder ./data_generator/svg2 --output_folder data/svg_output_20200328 --image_type svg 

# 训练模型
python train.py --data_folder data/svg_output_20200328 --data_name chart_5_cap_5_min_wf --image_type svg --svg_channel 13

# 使用模型
python caption.py --img ./data_generator/svg/2.svg  --model checkpoint/chart_5_cap_5_min_wf-2020-03-26-10-56/Best.pth.tar --word_map data/svg_output_20200328/WORDMAP_chart_5_cap_5_min_wf.json  --image_type svg

# 使用模型on dl
python caption.py --img ./data_generator/svg2/2.svg  --model checkpoint/chart_5_cap_5_min_wf-2020-03-26-13-05/Best.pth.tar --word_map data/svg_output_20200328/WORDMAP_chart_5_cap_5_min_wf.json  --image_type svg


# 20200328

# 训练模型
python train.py --data_folder data/svg_output_20200328 --data_name chart_5_cap_5_min_wf --image_type svg --svg_channel 13

python train.py --data_folder data/svg_output_20200328 --data_name chart_5_cap_5_min_wf --image_type svg --input_nc 3,2,4,3,1 --output_nc 5,5,5,5,5



# 使用模型on dl
python caption.py --img ./data_generator/svg2/2.svg  --model checkpoint/chart_5_cap_5_min_wf-2020-03-26-13-05/Best.pth.tar --word_map data/svg_output_20200328/WORDMAP_chart_5_cap_5_min_wf.json  --image_type svg


# 20200331 


python caption.py --img ./data_generator/svg2/2.svg  --model checkpoint/chart_5_cap_5_min_wf-2020-03-29-00-26/Best.pth.tar --word_map data/svg_output_20200328/WORDMAP_chart_5_cap_5_min_wf.json  --image_type svg

# 采用更少的参数： embed dim 32, attention_dim 32, decoder_dim 32

python train.py --data_folder data/svg_output_20200328 --data_name chart_5_cap_5_min_wf --image_type svg --input_nc 3,2,4,3,1 --output_nc 5,5,5,5,5 --emb_dim 32 --attention_dim 32 --decoder_dim 32



python caption.py --img ./data_generator/svg2/2.svg  --model checkpoint/chart_5_cap_5_min_wf-2020-03-31-16-31/Best.pth.tar --word_map data/svg_output_20200328/WORDMAP_chart_5_cap_5_min_wf.json  --image_type svg


# 20200407


python train.py --data_folder data/svg_output_20200328 --data_name chart_5_cap_5_min_wf --image_type svg --input_nc 3,2,4,3,1 --output_nc 5,5,5,5,5 --emb_dim 512 --attention_dim 512 --decoder_dim 512


python caption.py --img ./data_generator/svg2/2.svg  --model checkpoint/chart_5_cap_5_min_wf-2020-04-07-15-32/Best.pth.tar --word_map data/svg_output_20200328/WORDMAP_chart_5_cap_5_min_wf.json  --image_type svg



# 20200408

python create_input_files.py --dataset chart --karpathy_json_path data_generator/svg2/dataset.json --image_folder ./data_generator/svg2 --output_folder data/svg_output_20200408 --image_type svg --need_text 

python train.py --data_folder data/svg_output_20200408 --data_name chart_5_cap_5_min_wf --image_type svg --input_nc 3,2,4,3,1 --output_nc 5,5,5,5,5 --emb_dim 512 --attention_dim 512 --decoder_dim 512

python train.py --data_folder data/svg_output_20200408 --data_name chart_5_cap_5_min_wf --image_type svg --input_nc 3,2,4,3,1 --output_nc 5,5,5,5,5 --emb_dim 512 --attention_dim 512 --decoder_dim 512 --need_text


# 20200409

python create_input_files.py --dataset chart --karpathy_json_path data_generator/svg2/dataset.json --image_folder ./data_generator/svg2 --output_folder data/svg_output_20200409 --image_type svg --need_text 

python train.py --data_folder data/svg_output_20200409 --data_name chart_5_cap_5_min_wf --image_type svg --input_nc 3,2,4,3,1 --output_nc 5,5,5,5,5 --emb_dim 512 --attention_dim 512 --decoder_dim 512 --need_text

# 20200410

python3 gen_sent2.py -n 10 -p svg20200410
# 测试生成数据的过程

python3 gen_sentence.py -n 50000 -p svg20200410
# 测试能否生成具有文字的数据集// 事实证明，可以

# 下一步是解析数据集，即从数据集中解析其中的相应的数据


python create_input_files.py --dataset chart --karpathy_json_path data_generator/svg20200410/dataset.json --image_folder ./data_generator/svg20200410 --output_folder data/svg_output_20200410 --image_type svg --need_text 


python train.py --data_folder data/svg_output_20200410 --data_name chart_5_cap_5_min_wf --image_type svg --input_nc 3,2,4,3,1 --output_nc 5,5,5,5,5 --emb_dim 512 --attention_dim 512 --decoder_dim 512 --need_text


python caption.py --img ./data_generator/svg20200410/222.svg  --model checkpoint/chart_5_cap_5_min_wf-2020-04-10-16-11/Best.pth.tar --word_map data/svg_output_20200410/WORDMAP_chart_5_cap_5_min_wf.json  --image_type svg --need_text


# /home/can.liu/caption/chartcaption/checkpoint/chart_5_cap_5_min_wf-2020-04-10-16-11

# 再下一步是测试其中是否可以正常运行

# 我觉得OK

