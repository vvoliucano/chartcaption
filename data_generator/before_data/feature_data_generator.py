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
import numpy

if sys.platform == "linux":
	node_name = "./"
else:
	node_name = "node "


def add_aspect_ratio(setting):
	aspect_ratio = 1 + abs(numpy.random.normal(0, 0.5))
	if numpy.random.random() > 0.5:
		aspect_ratio = 1 / aspect_ratio

	aspect_ratio = aspect_ratio * 1.4
	setting["aspect_ratio"] = aspect_ratio

	padding_value = numpy.random.uniform(0.1, 0.4)
	setting["paddingValue"] = padding_value
	return setting

def generate_single_trend_setting():
	setting = basic_trend_setting()
	
	feature = random.choice([get_sim_trend, get_comp_trend])(random.choice(setting["category_name"]), setting["ordinal_name"])
	setting["feature"] = [feature]

	return setting

def generate_couple_trend_setting():
	setting = basic_trend_setting()
	cat_chosen_num = 2
	cat_choice = random.sample(setting["category_name"], cat_chosen_num)
	setting["feature"] = [random.choice([get_sim_trend, get_comp_trend])(cat_choice[i], setting["ordinal_name"]) for i in range(cat_chosen_num)]
	return setting

def basic_trend_setting():
	setting = {}
	setting["data_type"] = "ocq"
	setting['vis_type'] = random.choice(["load_group_bar_chart", "load_group_bar_chart_horizontal", "load_stack_bar_chart", "load_stack_bar_chart_horizontal"])
	cat_num = random.randint(2, 5)
	ord_num = random.randint(3, 7)
	setting["category_name"] = ["item" + str(i) for i in range(cat_num)]
	setting["ordinal_name"] = ["ord" + str(i) for i in range(ord_num)]

	return setting



def get_sim_trend(cat, ordinal_array):
	value1 = random.randint(20, 100)
	value2 = random.randint(20, 100)
	feature = {}
	feature["feature_type"] = "trend"
	feature["name"] = cat
	feature["step"] = [{"position": ordinal_array[0], "value": value1}, {"position": ordinal_array[-1], "value": value2}]
	return feature

def get_comp_trend(cat, ordinal_array):
	ordinal_array_num = len(ordinal_array)

	if ordinal_array_num < 3:
		return get_sim_trend(cat, ordinal_array)
	# value1 = random.randint(20, 100)
	value_array = numpy.sort(numpy.random.randint(20, 100, 3, dtype = "int"))
	choice = ["mid_small", "mid_large"]

	current_choice = random.choice(choice)

	# 中间的值是较小的值
	if current_choice == "mid_small":
		v1_index = random.choice([1,2])
		v3_index = 3 - v1_index
		value_2 = value_array[0]
		value_1 = value_array[v1_index]
		value_3 = value_array[v3_index]
		value_final = [value_1, value_2, value_3]

	# 中间的值是较大的值
	else:
		v1_index = random.choice([0,1])
		v3_index = 1 - v1_index
		value_2 = value_array[2]
		value_1 = value_array[v1_index]
		value_3 = value_array[v3_index]
		value_final = [value_1, value_2, value_3]

	value_final = [int(v) for v in value_final]

	feature = {}
	feature["feature_type"] = "trend"
	feature["feature"] = "trend"
	feature["name"] = cat

	step_index = [0, numpy.random.randint(1, ordinal_array_num - 1), ordinal_array_num - 1]

	feature["step"] = [{"position": ordinal_array[step_index[i]], "value": value_final[i]} for i in range(3)]

	# feature["step"] = [{"position": ordinal_array[0], "value": value[0]}, {"position": ordinal_array[-1], "value": value[2]}]
	return feature

def generate_setting(number = 1000):
	setting_array = []
	for i in range(number):
		if i % 100 == 0:
			print(f"current is the {i}-th 100-number")
		current_setting = random.choice([generate_single_trend_setting, generate_couple_trend_setting])()

		current_setting["filename"] = str(i) + ".svg"
		setting_array.append(current_setting)

	return setting_array

def get_extreme_feature(setting, data):
	# print("setting", setting)
	if (data["type"] == "ocq"):
		data_array = data["data_array"]
		cat_num = len(data["c0"])
		ord_num = len(data["o0"])

		max_or_min = [max, min]
		extreme_function = random.choice(max_or_min)
		chosen_cat_index = random.choice([i for i in range(cat_num)])

		extreme_value = extreme_function([item["q0"] for item in data_array if item["c0"] == chosen_cat_index])
		extreme_index = [item["id"] for item in data_array if item['q0'] == extreme_value and item["c0"] == chosen_cat_index]
		extreme_item = data_array[extreme_index[0]]

		feature = {}
		feature["feature_type"] = "maximum" if extreme_function == max else "minimum"
		feature["name"] = data["c0"][chosen_cat_index]
		feature["position"] = data["o0"][extreme_item['o0']] # position of the value
		feature['value'] = extreme_item['q0']
		feature['focus'] = [extreme_item["id"]]
		setting["feature"].append(feature)
		# print("setting after", setting)
	
	return

def get_compare_trend(setting, data):
	if (data["type"] != "ocq"):
		return

	trend_features = [feature for feature in setting["feature"] if feature["feature_type"] == "trend"]

	if len(trend_features) < 2: # 如果没有两个 trend 那就没有意义
		return

	chosen_features = random.sample(trend_features, 2) # 如果有两个 trend 我们考虑一下。

	# 首先获得两个数组

	cat_array = data["c0"]
	ord_array = data['o0']

	ord_num = len(data['o0'])

	data_array = data["data_array"]

	compare_array = []
	compare_name = []
	compare_cat_idx = []

	for feature in chosen_features:
		cat_name = feature["name"]
		cat_index = cat_array.index(cat_name)
		current_array = [item["q0"] for item in data_array if item['c0'] == cat_index]
		compare_array.append(current_array)
		compare_name.append(cat_name)
		compare_cat_idx.append(cat_index)

	# print("compare_array", compare_array)
	diff_array = [compare_array[0][i] > compare_array[1][i] for i in range(len(compare_array[0]))]
	# print("diff_array", diff_array)

	if sum(diff_array) == len(diff_array):
		name1 = compare_name[0]
		name2 = compare_name[1]
		focus = [compare_cat_idx[0] * ord_num + i for i in range(ord_num)] + [compare_cat_idx[1] * ord_num + i for i in range(ord_num)]
		# print("compare focus (higher): ", focus)
		relation = "higher than"
		position = f"from {ord_array[0]} to {ord_array[-1]}"
		feature = get_compare_feature(name1, name2, position, relation, focus)
		setting["feature"].append(feature)

	elif sum(diff_array) == 0:
		name1 = compare_name[1]
		name2 = compare_name[0]
		focus = [compare_cat_idx[0] * ord_num + i for i in range(ord_num)] + [compare_cat_idx[1] * ord_num + i for i in range(ord_num)]
		# print("compare focus (lower): ", focus)
		relation = "lower than"
		position = f"from {ord_array[0]} to {ord_array[-1]}"
		feature = get_compare_feature(name1, name2, position, relation, focus)
		setting["feature"].append(feature)

	else:
		for i in range(len(diff_array) - 1):
			if diff_array[i] != diff_array[i + 1]:
				if diff_array[i]:
					name1 = compare_name[1]
					name2 = compare_name[0]
				else:
					name1 = compare_name[0]
					name2 = compare_name[1]

				focus = [compare_cat_idx[0] * ord_num + i + 1, compare_cat_idx[1] * ord_num + i + 1]
				position = ord_array[i + 1]
				feature = get_surpass_feature(name1, name2, position, focus)
				# print("surpass feature: ", feature)
				setting["feature"].append(feature)


		
	# 计算两个数组的差
	# 如果发现一个数组恒大于另一个数组，那就是比较类型
	# 如果发现一个数组前半段大于、后半段小于，那就是surpass 类型
	# 
def get_surpass_feature(name1, name2, position, focus):
	feature = {}
	feature["feature_type"] = "surpass"
	feature["name1"] = name1
	feature["name2"] = name2
	feature["focus"] = focus
	feature["position"] = position
	return feature

def get_compare_feature(name1, name2, position, relation, focus):
	feature = {}
	feature["feature_type"] = "compare"
	feature["name1"] = name1
	feature["name2"] = name2
	feature["focus"] = focus
	feature["position"] = position
	feature["relation"] = relation
	return feature

def extract_feature_from_data(setting, data):
	get_extreme_feature(setting, data)
	get_compare_trend(setting, data)
	return setting

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
		setting = extract_feature_from_data(setting, data)
		setting = generate_sentence_by_feature(setting)
		data["feature"] = setting["feature"]
		data["filename"] = setting["filename"]
		add_aspect_ratio(data)
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



