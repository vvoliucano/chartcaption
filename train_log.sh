

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

python3 gen_sentence.py -n 10 -p svg20200410
# 测试能否生成具有文字的数据集// 事实证明，可以

# 下一步是解析数据集，即从数据集中解析其中的相应的数据


python create_input_files.py --dataset chart --karpathy_json_path data_generator/svg20200410/dataset.json --image_folder ./data_generator/svg20200410 --output_folder data/svg_output_20200410 --image_type svg --need_text 

python train.py --data_folder data/svg_output_20200410 --data_name chart_5_cap_5_min_wf --image_type svg --input_nc 3,2,4,3,1 --output_nc 5,5,5,5,5 --emb_dim 512 --attention_dim 512 --decoder_dim 512 --need_text

python caption.py --img ./data_generator/svg20200410/222.svg  --model checkpoint/chart_5_cap_5_min_wf-2020-04-10-16-11/Best.pth.tar --word_map data/svg_output_20200410/WORDMAP_chart_5_cap_5_min_wf.json  --image_type svg --need_text

python caption.py --img ./data_generator/svg20200410/2.svg  --model checkpoint/chart_5_cap_5_min_wf-2020-04-10-23-26/Best.pth.tar --word_map data/svg_output_20200410/WORDMAP_chart_5_cap_5_min_wf.json  --image_type svg --need_text



# /home/can.liu/caption/chartcaption/checkpoint/chart_5_cap_5_min_wf-2020-04-10-16-11

# 再下一步是测试其中是否可以正常运行

# 我觉得OK


20200416

# 切换到通用数据集所在的文件目录
cd data_generator/before_data

# 利用本地的nodejs 测试生成相应的可视化图表
node gen_svg.js --input_file try_set.json --directory svg_try 

# 以上测试说明，生成图表的js 在本地运行正常，其中使用了本地运行的js 运行d3的库，解决了d3网络版本和本地版本运行的差异
# 涉及到文件读取等方式需要重新考量

# 生成部分的配置文件，测试命令行参数的运行状况
python new_data.py --number 10 --path svg_20200416

python new_data.py --number 100000 --path svg_20200416


node gen_svg.js --input_file svg_20200416/origin_data.json --directory svg_20200416


# 然而这种方法在生成的数据量大的情况下，node 没有办法执行，因此需要将配置文件更换一种方式存储


# 我们生成了通用的数据集，在这个数据集中，构建了图表相关的大多数的特征种类以及描述的方式。我们同时也提供了


20200417

# 将生成的数据统一放在data 目录之下，

python new_data.py --number 100000 --path ../../data/svg_origin_20200417

node gen_svg.js --input_file ../../data/svg_origin_20200417/origin_data.json --directory ../../data/svg_origin_20200417

node gen_svg.js --input_file ../../data/svg_origin_20200417/origin_data.json --directory ../../data/svg_origin_20200417

./gen_svg.js --input_file ../../data/svg_origin_20200417/origin_data.json --directory ../../data/svg_origin_20200417

# 这种方式会让node 文件内存溢出，现将相关文件分离出来

python new_data.py --number 10 --path ../../data/svg_origin_20200417

# 现在让每个配置文件运行一次，发现速度太慢
python new_data.py --number 100000 --path ../../data/svg_origin_20200417


./gen_svg.js --input_file ../../data/svg_origin_20200417/origin_data.json --directory ../../data/svg_origin_20200417
# 现在让每个配置文件每一百次运行一次


python new_data.py --number 1000 --path try_dir --period 100

# on the dl
python new_data.py --number 100000 --path ../../data/20200417_dataset_bar --period 500

# local
python new_data.py --number 1000 --path ../../data/20200417_dataset_bar --period 100


cd ../../

python create_input_files.py --dataset chart --karpathy_json_path data/20200417_dataset_bar/karparthy_dataset.json --image_folder data/20200417_dataset_bar/svg --output_folder data/20200417_dataset_bar/deal --image_type svg --need_text --max_element_number 100


python train.py --data_folder data/20200417_dataset_bar/deal --svg_element_number 100 --data_name chart_5_cap_5_min_wf --image_type svg --input_nc 3,2,4,3,1 --output_nc 5,5,5,5,5 --emb_dim 512 --attention_dim 512 --decoder_dim 512 --need_text


python caption.py --img data/20200417_dataset_bar/svg/000000.svg  --model checkpoint/chart_5_cap_5_min_wf-2020-04-17-23-11/Best.pth.tar --word_map data/20200417_dataset_bar/deal/WORDMAP_chart_5_cap_5_min_wf.json  --image_type svg --need_text --max_element_number 100

python caption.py --img data/20200417_dataset_bar/svg/000001.svg  --model checkpoint/chart_5_cap_5_min_wf-2020-04-17-23-11/Best.pth.tar --word_map data/20200417_dataset_bar/deal/WORDMAP_chart_5_cap_5_min_wf.json  --image_type svg --need_text --max_element_number 100

python caption.py --img data/20200417_dataset_bar/svg/000002.svg  --model checkpoint/chart_5_cap_5_min_wf-2020-04-17-23-11/epoch_12.pth.tar --word_map data/20200417_dataset_bar/deal/WORDMAP_chart_5_cap_5_min_wf.json  --image_type svg --need_text --max_element_number 100



20200418 

# 本地测试版本
# local 生成 配置文件，生成相应的svg 的文件
python new_data.py --number 200 --path ../../data/20200418_dataset_bar --period 100

# 解析相应的数据，从配置文件和相应的svg 文件出发
python create_input_files.py --dataset chart --karpathy_json_path data/20200418_dataset_bar/karparthy_dataset.json --image_folder data/20200418_dataset_bar/svg --output_folder data/20200418_dataset_bar/deal --image_type svg --need_text --max_element_number 100


# 实验室服务器版本 解决大小写问题
python new_data.py --number 100000 --path ../../data/20200418_dataset_bar --period 100

# 解析相应的数据，从配置文件和相应的svg 文件出发
python create_input_files.py --dataset chart --karpathy_json_path data/20200418_dataset_bar/karparthy_dataset.json --image_folder data/20200418_dataset_bar/svg --output_folder data/20200418_dataset_bar/deal --image_type svg --need_text --max_element_number 100


python train.py --data_folder data/20200418_dataset_bar/deal --svg_element_number 100 --data_name chart_5_cap_5_min_wf --image_type svg --input_nc 3,2,4,3,1 --output_nc 5,5,5,5,5 --emb_dim 512 --attention_dim 512 --decoder_dim 512 --need_text

python caption.py --img data/20200418_dataset_bar/svg/000001.svg  --model checkpoint/chart_5_cap_5_min_wf-2020-04-18-17-03/Best.pth.tar --word_map data/20200418_dataset_bar/deal/WORDMAP_chart_5_cap_5_min_wf.json  --image_type svg --need_text --max_element_number 100

20200419

python caption.py --img data/20200418_dataset_bar/svg/000010.svg  --model checkpoint/chart_5_cap_5_min_wf-2020-04-18-17-03/epoch_39.pth.tar --word_map data/20200418_dataset_bar/deal_run/WORDMAP_chart_5_cap_5_min_wf.json  --image_type svg --need_text --max_element_number 100

python caption.py --img data/20200418_dataset_bar/svg/000030.svg  --model checkpoint/chart_5_cap_5_min_wf-2020-04-18-17-03/epoch_39.pth.tar --word_map data/20200418_dataset_bar/deal_run/WORDMAP_chart_5_cap_5_min_wf.json  --image_type svg --need_text --max_element_number 100

python test_module.py --img data/20200418_dataset_bar/svg/000030.svg  --model checkpoint/chart_5_cap_5_min_wf-2020-04-18-17-03/epoch_39.pth.tar --word_map data/20200418_dataset_bar/deal_run/WORDMAP_chart_5_cap_5_min_wf.json  --image_type svg --need_text --max_element_number 100

python test_module.py --img data/20200418_dataset_bar/svg/000020.svg



20200422


# local
python new_data.py --number 100 --path ../../data/20200422_dataset_bar --period 10

python create_input_files.py --dataset chart --karpathy_json_path data/20200418_dataset_bar/karparthy_dataset.json --image_folder data/20200418_dataset_bar/svg --output_folder data/20200422_dataset_bar/deal --image_type svg --need_text --max_element_number 100

python new_data.py --number 1000 --path ../../data/20200422_dataset_bar --period 100

python create_input_files.py --dataset chart --karpathy_json_path data/20200422_dataset_bar/karparthy_dataset.json --image_folder data/20200422_dataset_bar/svg --output_folder data/20200422_dataset_bar/deal --image_type svg --need_text --max_element_number 100

python train.py --data_folder data/20200422_dataset_bar/deal --svg_element_number 100 --data_name chart_5_cap_5_min_wf --image_type svg --input_nc 3,2,4,3,1 --output_nc 5,5,5,5,5 --emb_dim 512 --attention_dim 512 --decoder_dim 512 --need_text

# python caption.py --img data/20200418_dataset_bar/svg/000001.svg  --model checkpoint/chart_5_cap_5_min_wf-2020-04-18-17-03/Best.pth.tar --word_map data/20200418_dataset_bar/deal/WORDMAP_chart_5_cap_5_min_wf.json  --image_type svg --need_text --max_element_number 100

python test_module.py --img data/20200418_dataset_bar/svg/000060.svg  --model checkpoint/chart_5_cap_5_min_wf-2020-04-18-17-03/epoch_39.pth.tar --word_map data/20200418_dataset_bar/deal_run/WORDMAP_chart_5_cap_5_min_wf.json  --image_type svg --need_text --max_element_number 100

python test_module.py --img data/20200418_dataset_bar/svg/000030.svg  --model checkpoint/chart_5_cap_5_min_wf-2020-04-18-17-03/epoch_39.pth.tar --word_map data/20200418_dataset_bar/deal_run/WORDMAP_chart_5_cap_5_min_wf.json  --image_type svg --need_text --max_element_number 100

python test_module.py --img data/20200418_dataset_bar/svg/000030.svg  --model checkpoint/chart_5_cap_5_min_wf-2020-04-22-19-59/epoch_0.pth.tar --word_map checkpoint/chart_5_cap_5_min_wf-2020-04-22-19-59/WORDMAP_chart_5_cap_5_min_wf.json  --image_type svg --need_text --max_element_number 100 --replace_token

checkpoint/chart_5_cap_5_min_wf-2020-04-22-19-59


20200423

python test_module.py --img data/20200418_dataset_bar/svg/000030.svg  --model checkpoint/chart_5_cap_5_min_wf-2020-04-22-19-59/epoch_16.pth.tar --word_map checkpoint/chart_5_cap_5_min_wf-2020-04-22-19-59/WORDMAP_chart_5_cap_5_min_wf.json  --image_type svg --need_text --max_element_number 100 --replace_token

python new_data.py --number 100 --path ../../data/20200423_dataset_bar --period 10

