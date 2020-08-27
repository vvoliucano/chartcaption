# This file is to generate datasets with features.

from generate_data.generate_rule_data import generate_data_by_setting

from sentence_template import generate_sentence_by_feature
import json
import os
import shutil
import argparse
import random
from tqdm import tqdm
import sys

if sys.platform == "linux":
	node_name = "./"
else:
	node_name = "node "

	

def generate_single_trend_setting():
	setting = {}
	setting["data_type"] = "ocq"
	setting['vis_type'] = random.choice(["load_group_bar_chart", "load_group_bar_chart_horizontal", "load_stack_bar_chart", "load_stack_bar_chart_horizontal"])
	cat_num = random.randint(2, 5)
	ord_num = random.randint(3, 7)

	setting["category_name"] = ["item" + str(i) for i in range(cat_num)]
	setting["ordinal_name"] = ["ord" + str(i) for i in range(ord_num)]

	feature = get_trend(random.choice(setting["category_name"]), setting["ordinal_name"])
	setting["feature"] = [feature]

	return setting

def get_trend(cat, ordinal_array):
	value1 = random.randint(20, 100)
	value2 = random.randint(20, 100)
	feature = {}
	feature["feature_type"] = "trend"
	feature["name"] = cat
	feature["step"] = [{"position": ordinal_array[0], "value": value1}, {"position": ordinal_array[-1], "value": value2}]
	return feature

def generate_setting(number = 1000):
	setting_array = []
	for i in range(number):
		if i % 100 == 0:
			print(f"current is the {i}-th 100-number")
		current_setting = generate_single_trend_setting()
		current_setting["filename"] = str(i) + ".svg"
		setting_array.append(current_setting)

	return setting_array


def gen_tvt(train, val, test):
    def ret_func():
        rv = random.random()
        if rv < train:
            return "train"
        if rv < train + val:
            return "val"
        return "test"
    return ret_func

def convert_to_karparthy(original_data):
    res = {"dataset": "weak", "images": []}
    cur_sentence_id = 0
    gen_split = gen_tvt(0.8, 0.1, 0.1)
    for pindex, item in enumerate(original_data):
        # print(item)
        sentences = item["feature"]
        here_sentence_num = len(sentences)
        image = {
            "sentids": [],
            "imgid": pindex,
            "sentences": [],
            "split": gen_split(),
            "filename": item["filename"]
        }
        for sen_index, sentence_item in enumerate(sentences):
            sentid = cur_sentence_id + sen_index
            sen = sentence_item["sentence"].replace(",", " ,").replace(".", " .")
            # print(sen)
            sentence = {
                "tokens": [item.lower() for item in sen.split(" ")],
                "raw": sen,
                "imgid": pindex,
                "sentid": sentid,
                "focus_id": sentence_item["focus"]
            }
            image["sentids"].append(sentid)
            image["sentences"].append(sentence)
        res["images"].append(image)
        cur_sentence_id += here_sentence_num
    return res

def make_sure_dir(dir_name):
	if not os.path.isdir(dir_name):
		os.mkdir(dir_name)
	return

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='generate dataset')
	parser.add_argument('--number', '-n', type=int, default = 100, help='number')
	parser.add_argument('--path', '-p', type=str, default = "try_dir", help='The path')
	parser.add_argument('--period', '-i', type = int, default = 100, help = 'number of the iterater')
	
	args = parser.parse_args()
	# 解析相应的数据

	print(f"The size of the dataset is {args.number}")

	if args.number < args.period:
		args.period = args.number

	args.number = int(args.number / args.period) * args.period


	
	general_path = args.path

	json_path = os.path.join(general_path, "json")
	svg_path = os.path.join(general_path, "svg")
	karparthy_file = os.path.join(general_path, "karparthy_dataset.json")

	make_sure_dir(general_path)
	make_sure_dir(json_path)
	make_sure_dir(svg_path)
	# os.mkdir(json_path)
	# os.mkdir(svg_path)

	setting_array = generate_setting(number = args.number)

	data_array = []
	for setting in setting_array:
		data = generate_data_by_setting(setting)
		setting = generate_sentence_by_feature(setting)
		data["feature"] = setting["feature"]
		data["filename"] = setting["filename"]
		data_array.append(data)

	unit_size = args.period

	for i in range(int(len(data_array) / unit_size)):
		current_json_path = os.path.join(json_path, f"{i * unit_size}-{(i + 1) * unit_size - 1}.json")
		print(current_json_path, "generate svg")
		with open(current_json_path, "w") as f:
			json.dump(data_array[i * unit_size: (i + 1) * unit_size], f, indent = 2)
		os.system(f"{node_name}gen_svg.js --input {current_json_path} --output_dir {svg_path}")  


	karparthy_dataset = convert_to_karparthy(data_array)
	print("save to karparthy file")
	with open(karparthy_file, "w") as f:
		json.dump(karparthy_dataset, f, indent = 2)

	# with open("try_data.json", "w") as f:
	# 	json.dump(data_array, f, indent = 2)



