

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

#local: we have move the training checkpoint model to local

python test_module.py --img data/20200418_dataset_bar/svg/000030.svg  --model checkpoint/chart_5_cap_5_min_wf-2020-04-22-19-59/epoch_16.pth.tar --word_map checkpoint/chart_5_cap_5_min_wf-2020-04-22-19-59/WORDMAP_chart_5_cap_5_min_wf.json  --image_type svg --need_text --max_element_number 100 --replace_token

python new_data.py --number 100 --path ../../data/20200423_dataset_bar --period 10

python new_data.py --number 100 --path try_dir --period 10


# remote
python new_data.py --number 100000 --path ../../data/20200423_dataset_bar --period 100

python create_input_files.py --dataset chart --karpathy_json_path data/20200423_dataset_bar/karparthy_dataset.json --image_folder data/20200423_dataset_bar/svg --output_folder data/20200423_dataset_bar/deal --image_type svg --need_text --max_element_number 100

python train.py --data_folder data/20200423_dataset_bar/deal --svg_element_number 100 --data_name chart_5_cap_5_min_wf --image_type svg --input_nc 3,2,4,3,1 --output_nc 5,5,5,5,5 --emb_dim 512 --attention_dim 512 --decoder_dim 512 --need_text

node gen_svg.js --input ../../data/20200422_dataset_bar/json/airport.json --output_dir ./

python test_module.py --img 000004.svg  --model checkpoint/chart_5_cap_5_min_wf-2020-04-22-19-59/epoch_0.pth.tar --word_map checkpoint/chart_5_cap_5_min_wf-2020-04-22-19-59/WORDMAP_chart_5_cap_5_min_wf.json  --image_type svg --need_text --max_element_number 100 --replace_token

python new_data.py --number 100000 --path ../../data/20200423_dataset_bar_new --period 100

python create_input_files.py --dataset chart --karpathy_json_path data/20200423_dataset_bar_new/karparthy_dataset.json --image_folder data/20200423_dataset_bar_new/svg --output_folder data/20200423_dataset_bar_new/deal --image_type svg --need_text --max_element_number 100

python train.py --data_folder data/20200423_dataset_bar_new/deal --svg_element_number 100 --data_name chart_5_cap_5_min_wf --image_type svg --input_nc 3,2,4,3,1 --output_nc 5,5,5,5,5 --emb_dim 512 --attention_dim 512 --decoder_dim 512 --need_text

20200424

# remote
python test_module.py --img data/try_dataset/000004.svg  --model checkpoint/chart_5_cap_5_min_wf-2020-04-23-23-40/epoch_12.pth.tar --word_map data/20200423_dataset_bar_new/deal/WORDMAP_chart_5_cap_5_min_wf.json  --image_type svg --need_text --max_element_number 100 --replace_token

# 创建数据集 remote
python new_data.py --number 1000 --path ../../data/20200424_dataset_bar --period 100

python create_input_files.py --dataset chart --karpathy_json_path data/20200424_dataset_bar/karparthy_dataset.json --image_folder data/20200424_dataset_bar/svg --output_folder data/20200424_dataset_bar/deal --image_type svg --need_text --max_element_number 100

python train.py --data_folder data/20200424_dataset_bar/deal --svg_element_number 100 --data_name chart_5_cap_5_min_wf --image_type svg --input_nc 3,2,4,3,1 --output_nc 5,5,5,5,5 --emb_dim 512 --attention_dim 512 --decoder_dim 512 --need_text --pretrained_model checkpoint/chart_5_cap_5_min_wf-2020-04-23-23-40/epoch_12.pth.tar

# 累加训练 chart_5_cap_5_min_wf-2020-04-24-10-19
python test_module.py --img data/20200424_dataset_bar/svg/000999.svg  --model checkpoint/chart_5_cap_5_min_wf-2020-04-24-10-19/epoch_25_bleu_0.759521329608022.pth.tar --word_map data/20200424_dataset_bar/deal/WORDMAP_chart_5_cap_5_min_wf.json  --image_type svg --need_text --max_element_number 100 --replace_token

python test_module.py --img data/20200424_dataset_bar/svg/000999.svg  --model checkpoint/chart_5_cap_5_min_wf-2020-04-24-10-19/epoch_25_bleu_0.759521329608022.pth.tar --word_map data/20200424_dataset_bar/deal/WORDMAP_chart_5_cap_5_min_wf.json  --image_type svg --need_text --max_element_number 100 --replace_token

# 小规模训练 chart_5_cap_5_min_wf-2020-04-24-11-44
python train.py --data_folder data/20200424_dataset_bar/deal --svg_element_number 100 --data_name chart_5_cap_5_min_wf --image_type svg --input_nc 3,2,4,3,1 --output_nc 5,5,5,5,5 --emb_dim 512 --attention_dim 512 --decoder_dim 512 --need_text 

# 出现一些问题 一些词汇没有正确被替代，导致生成结果并不正确
python test_module.py --img data/20200424_dataset_bar/svg/000999.svg  --model checkpoint/chart_5_cap_5_min_wf-2020-04-24-11-44/epoch_99_bleu_0.6830259834046531.pth.tar --word_map data/20200424_dataset_bar/deal/WORDMAP_chart_5_cap_5_min_wf.json  --image_type svg --need_text --max_element_number 100 --replace_token

# local 现在修改了相应的创建数据集的代码 运行良好
python create_input_files.py --dataset chart --karpathy_json_path data/20200423_dataset_bar/karparthy_dataset.json --image_folder data/20200423_dataset_bar/svg --output_folder data/20200423_dataset_bar/deal --image_type svg --need_text --max_element_number 100

# remote 将创建数据集的代码 在远程运行 过程中修改了每个图片的caption 数量不一致的问题。
python create_input_files.py --dataset chart --karpathy_json_path data/20200424_dataset_bar/karparthy_dataset.json --image_folder data/20200424_dataset_bar/svg --output_folder data/20200424_dataset_bar/deal --image_type svg --need_text --max_element_number 100

# 重新在新的数据集下进行训练 所在目录 chart_5_cap_5_min_wf-2020-04-24-17-11
python train.py --data_folder data/20200424_dataset_bar/deal --svg_element_number 100 --data_name chart_5_cap_5_min_wf --image_type svg --input_nc 3,2,4,3,1 --output_nc 5,5,5,5,5 --emb_dim 512 --attention_dim 512 --decoder_dim 512 --need_text 

# 查看训练后的结果 显然在小规模数据集下，其功能并不完好
python test_module.py --img data/20200424_dataset_bar/svg/000999.svg  --model checkpoint/chart_5_cap_5_min_wf-2020-04-24-17-11/Best.pth.tar --word_map data/20200424_dataset_bar/deal/WORDMAP_chart_5_cap_5_min_wf.json  --image_type svg --need_text --max_element_number 100 --replace_token

python test_module.py --img data/20200424_dataset_bar/svg/000998.svg  --model checkpoint/chart_5_cap_5_min_wf-2020-04-24-17-11/Best.pth.tar --word_map data/20200424_dataset_bar/deal/WORDMAP_chart_5_cap_5_min_wf.json  --image_type svg --need_text --max_element_number 100 --replace_token

# 那我们就在更为大型的数据集上进行测试

python create_input_files.py --dataset chart --karpathy_json_path data/20200423_dataset_bar_new/karparthy_dataset.json --image_folder data/20200423_dataset_bar_new/svg --output_folder data/20200424_dataset_bar_new/deal --image_type svg --need_text --max_element_number 100

# chart_5_cap_5_min_wf-2020-04-24-23-52
python train.py --data_folder data/20200424_dataset_bar_new/deal --svg_element_number 100 --data_name chart_5_cap_5_min_wf --image_type svg --input_nc 3,2,4,3,1 --output_nc 5,5,5,5,5 --emb_dim 512 --attention_dim 512 --decoder_dim 512 --need_text 

python test_module.py --img data/20200424_dataset_bar/svg/000999.svg  --model checkpoint/chart_5_cap_5_min_wf-2020-04-24-23-52/Best.pth.tar --word_map data/20200424_dataset_bar_new/deal/WORDMAP_chart_5_cap_5_min_wf.json  --image_type svg --need_text --max_element_number 100 --replace_token
python test_module.py --img data/20200424_dataset_bar/svg/000998.svg  --model checkpoint/chart_5_cap_5_min_wf-2020-04-24-23-52/Best.pth.tar --word_map data/20200424_dataset_bar_new/deal/WORDMAP_chart_5_cap_5_min_wf.json  --image_type svg --need_text --max_element_number 100 --replace_token

# 2020428

python new_data.py --number 10000 --path ../../data/20200428_dataset_bar_10000 --period 100

python new_data.py --number 20000 --path ../../data/20200428_dataset_bar_20000 --period 100

python new_data.py --number 40000 --path ../../data/20200428_dataset_bar_40000 --period 100

python nes_data.py --number 80000 --path ../../data/20200428_dataset_bar_80000 --period 100

python new_data.py --number 160000 --path ../../data/20200428_dataset_bar_160000 --period 100

python create_input_files.py --dataset chart --karpathy_json_path data/20200428_dataset_bar_10000/karparthy_dataset.json --image_folder data/20200428_dataset_bar_10000/svg --output_folder data/20200428_dataset_bar_10000/deal --image_type svg --need_text --max_element_number 100

python create_input_files.py --dataset chart --karpathy_json_path data/20200428_dataset_bar_20000/karparthy_dataset.json --image_folder data/20200428_dataset_bar_20000/svg --output_folder data/20200428_dataset_bar_20000/deal --image_type svg --need_text --max_element_number 100

python create_input_files.py --dataset chart --karpathy_json_path data/20200428_dataset_bar_40000/karparthy_dataset.json --image_folder data/20200428_dataset_bar_40000/svg --output_folder data/20200428_dataset_bar_40000/deal --image_type svg --need_text --max_element_number 100

python create_input_files.py --dataset chart --karpathy_json_path data/20200428_dataset_bar_80000/karparthy_dataset.json --image_folder data/20200428_dataset_bar_80000/svg --output_folder data/20200428_dataset_bar_80000/deal --image_type svg --need_text --max_element_number 100

python create_input_files.py --dataset chart --karpathy_json_path data/20200428_dataset_bar_160000/karparthy_dataset.json --image_folder data/20200428_dataset_bar_160000/svg --output_folder data/20200428_dataset_bar_160000/deal --image_type svg --need_text --max_element_number 100

python train.py --data_folder data/20200428_dataset_bar_10000/deal --svg_element_number 100 --data_name chart_5_cap_5_min_wf --image_type svg --input_nc 3,2,4,3,1 --output_nc 5,5,5,5,5 --emb_dim 512 --attention_dim 512 --decoder_dim 512 --need_text 
# chart_5_cap_5_min_wf-2020-04-29-10-12

python train.py --data_folder data/20200428_dataset_bar_20000/deal --svg_element_number 100 --data_name chart_5_cap_5_min_wf --image_type svg --input_nc 3,2,4,3,1 --output_nc 5,5,5,5,5 --emb_dim 512 --attention_dim 512 --decoder_dim 512 --need_text 
# chart_5_cap_5_min_wf-2020-04-29-10-15

python train.py --data_folder data/20200428_dataset_bar_40000/deal --svg_element_number 100 --data_name chart_5_cap_5_min_wf --image_type svg --input_nc 3,2,4,3,1 --output_nc 5,5,5,5,5 --emb_dim 512 --attention_dim 512 --decoder_dim 512 --need_text 
# chart_5_cap_5_min_wf-2020-04-29-10-45

python train.py --data_folder data/20200428_dataset_bar_80000/deal --svg_element_number 100 --data_name chart_5_cap_5_min_wf --image_type svg --input_nc 3,2,4,3,1 --output_nc 5,5,5,5,5 --emb_dim 512 --attention_dim 512 --decoder_dim 512 --need_text 
# chart_5_cap_5_min_wf-2020-04-29-10-52

python train.py --data_folder data/20200428_dataset_bar_160000/deal --svg_element_number 100 --data_name chart_5_cap_5_min_wf --image_type svg --input_nc 3,2,4,3,1 --output_nc 5,5,5,5,5 --emb_dim 512 --attention_dim 512 --decoder_dim 512 --need_text 

# 20200819

python create_input_files.py --dataset chart --karpathy_json_path data/20200423_dataset_bar/karparthy_dataset.json --image_folder data/20200423_dataset_bar/svg --output_folder data/20200820_dataset_try/ --image_type svg --need_text --max_element_number 100

# 20200821 local
python new_data.py --number 20 --path ../../data/20200820_dataset_bar_20 --period 20

python create_input_files.py --dataset chart --karpathy_json_path data/20200820_dataset_bar_20/karparthy_dataset.json --image_folder data/20200820_dataset_bar_20/svg --output_folder data/20200820_dataset_bar_20/deal --image_type svg --need_text --max_element_number 100 --with_focus

python train.py --data_folder data/20200820_dataset_bar_20/deal --svg_element_number 100 --data_name chart_5_cap_5_min_wf --image_type svg --input_nc 3,2,4,3,1,1 --output_nc 5,5,5,5,5,5 --emb_dim 512 --attention_dim 512 --decoder_dim 512 --need_text 

# try
node gen_svg.js --input try_setting.json --output_dir try_dir

node gen_svg.js --input try_data.json --output_dir try_dir

python feature_data_generator.py --path ../../data/20200826

python data_generator/before_data/feature_data_generator.py --path data/20200826

python create_input_files.py --dataset chart --karpathy_json_path data/20200826/karparthy_dataset.json --image_folder data/20200826/svg --output_folder data/20200826/deal --image_type svg --need_text --max_element_number 100 --with_focus

python train.py --data_folder data/20200820_dataset_bar_20/deal --svg_element_number 100 --data_name chart_5_cap_5_min_wf --image_type svg --input_nc 3,2,4,3,1,1 --output_nc 5,5,5,5,5,5 --emb_dim 512 --attention_dim 512 --decoder_dim 512 --need_text 

./run.sh 20200827_20 20

python feature_data_generator.py --path ../../data/20200827_24 --number 24

python test_module.py --img data/20200424_dataset_bar/svg/000998.svg  --model checkpoint/chart_5_cap_5_min_wf-2020-04-23-23-40/epoch_12.pth.tar --word_map data/20200424_dataset_bar_new/deal/WORDMAP_chart_5_cap_5_min_wf.json  --image_type svg --need_text --max_element_number 100 --replace_token


# 20200829

# local
python test_module.py --img data/20200828/svg/1.svg  --model checkpoint/chart_5_cap_5_min_wf-2020-08-28-14-49/Best.pth.tar --word_map data/20200828/deal/WORDMAP_chart_5_cap_5_min_wf.json  --image_type svg --need_text --max_element_number 100 --replace_token --need_focus --focus 0,1,2

# remote
python test_module.py --img data/20200829_20000/svg/0.svg  --model checkpoint/chart_5_cap_5_min_wf-2020-08-29-11-02/Best.pth.tar --word_map data/20200829_20000/deal/WORDMAP_chart_5_cap_5_min_wf.json  --image_type svg --need_text --max_element_number 100 --replace_token --need_focus --focus 6,7,8,9,10,11


# 20200830

# local 


# remote
# dataset: 20200830_30000 processed_data_position: deal 			model_name: chart_5_cap_5_min_wf-2020-08-30-10-14
# dataset: 20200830_30000 processed_data_position: deal_no_focus 	model_name:	chart_5_cap_5_min_wf-2020-08-30-18-06

# without focus  
python create_input_files.py --dataset chart --karpathy_json_path data/20200830_30000/karparthy_dataset.json --image_folder data/20200830_30000/svg --output_folder data/20200830_30000/deal_no_focus --image_type svg --need_text --max_element_number 100 

python train.py --data_folder data/20200830_30000/deal_no_focus --svg_element_number 100 --data_name chart_5_cap_5_min_wf --image_type svg --input_nc 3,2,4,3,1 --output_nc 5,5,5,5,5 --emb_dim 512 --attention_dim 512 --decoder_dim 512 --need_text 










