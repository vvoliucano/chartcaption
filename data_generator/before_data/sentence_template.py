import json
import platform
import time
import random
import os

from generate_data.generate_rule_data import generate_rule_data



current_system = "linux"

if platform.system() == "Darwin":
	current_system = "macos"

if current_system == "linux":
	import language_check


# a = f'sdf{sdf}'

# a

# def get_ordinal_array():



# def get_feature_set():



def get_checked_sentence(text = "China's GDP increase from 2019 to 2020."):

	tool = language_check.LanguageTool('en-US')
	# text = u'A sentence with a error in the Hitchhiker’s Guide tot he Galaxy'
	matches = tool.check(text)
	# print(matches)
	new_text = language_check.correct(text, matches)
	return new_text


f = open(os.path.join(os.path.dirname(__file__), 'template.json'))

sentence_template = json.load(f)


def get_single_sentence_piece(step_0, step_1, slight_value = 0.03):
	begin = step_0["position"]
	begin_value = step_0["value"]
	end = step_1["position"]
	end_value = step_1["value"]
	if begin_value > end_value * (1 + slight_value):
		cap_type = "trend_dec"

	elif begin_value < end_value * (1 - slight_value):
		cap_type = "trend_inc"

	else:
		cap_type = "trend_stable"

	return get_trend_sentence("{name}", begin, end, begin_value, end_value, cap_type)

def combine_sentence_piece(name, sentence_piece_array):
	sentence = sentence_piece_array[0].format(name = name)

	# 第一个部分就是原型，而后的部分就是 it，and 呗
	for piece in sentence_piece_array[1:]: 
		sentence = sentence + ", and " + piece.format(name = "it")

	return sentence

def get_extreme_sentence_by_setting(feature_setting):
	name = feature_setting["name"]
	position = feature_setting["position"]
	cap_type = feature_setting["feature_type"]
	template_choice = sentence_template[cap_type]
	current_template = random.choice(template_choice)
	sentence = current_template.format(name = name, position = position)
	return sentence


def get_trend_sentence(name, begin, end, begin_value, end_value, cap_type = "trend_inc"):
	# sen1 = sentence_template["trend_inc"][0].format(name = "{name}", begin = begin, end = end, end_value = end_value)
	# sen2 = sen1.format(name = "this value")
	# print(sen1)
	# print(sen2)
	template_choice = sentence_template[cap_type]
	current_template = random.choice(template_choice)
	sentence = current_template.format(name = name, begin = begin, end = end, end_value = end_value)
	return sentence




def get_trend_sentence_by_setting(feature_setting, slight_value = 0.03):
	name = feature_setting["name"]
	steps = feature_setting["step"]
	piece_array = []

	for i in range(len(steps) - 1):
		current_steps = steps[i: i + 2]
		current_sentence_piece = get_single_sentence_piece(current_steps[0], current_steps[1], slight_value)
		piece_array.append(current_sentence_piece)

	return combine_sentence_piece(name, piece_array )

	# return get_trend_sentence(name, begin, end, begin_value, end_value, cap_type)





def get_compare_sentence(name1, name2, value1, value2, position):

	if value1 == value2:
		relation = "equals to"
	if value1 < value2:
		relation = "smaller than"
	else:
		relation = "higher than"

	if abs(value1 - value2) / value1 < 0.1:
		relation = "slightly " + relation
	template_choice = sentence_template["compare"]
	current_template = random.choice(template_choice)
	sentence = current_template.format(name1 = name1, name2 = name2, relation = relation, position = position)
	return sentence

def get_compare_sentence_by_setting(feature_setting):
	name1 = feature_setting["name1"]
	name2 = feature_setting["name2"]
	position = feature_setting['position']
	relation = feature_setting['relation']

	template_choice = sentence_template["compare"]
	current_template = random.choice(template_choice)
	sentence = current_template.format(name1 = name1, name2 = name2, relation = relation, position = position)
	return sentence

def get_surpass_sentence_by_setting(feature_setting):
	name1 = feature_setting['name1']
	name2 = feature_setting["name2"]
	position = feature_setting['position']

	template_choice = sentence_template["surpass"]
	current_template = random.choice(template_choice)
	sentence = current_template.format(name1 = name1, name2 = name2, position = position)
	return sentence

def get_entity(name = "", owner = "", color = "", shape = "" ):
	# name is the object
	# owner is the owner

	if name != "" and owner != "":
		return random.choice([f"{name} of {owner}", f'{owner}\'s'])

	if color != "" and shape != "":
		return f"the {color} {shape}"

	if owner != "":
		return owner

	if color != "":
		return f"the {color} one"

	if name != '':
		return name

	else:
		return "the value"


def generate_sentence_by_feature(setting):
	# print('setting_generate_sentence', setting)
	features = setting["feature"]
	for feature in features:
		feature_type = feature["feature_type"]
		if feature_type == "trend":
			feature["sentence"] = get_trend_sentence_by_setting(feature)
		elif feature_type == "maximum" or feature_type == "minimum":
			feature['sentence'] = get_extreme_sentence_by_setting(feature)
		elif feature_type == "surpass":
			feature['sentence'] = get_surpass_sentence_by_setting(feature)
		elif feature_type == "compare":
			feature['sentence'] = get_compare_sentence_by_setting(feature)
		else:
			print('currently we can not handle this feature type', feature_type)
			feature["sentence"] = ""

	return setting





if __name__ == '__main__':

	begin = 2001
	end = 2020
	begin_value = 20
	end_value = 200
	cap_type = "trend_inc"
	name = get_entity(name = "GDP", owner = "China")
	sentence = get_trend_sentence(name, begin, end, begin_value, end_value, cap_type = cap_type)
	print(cap_type, sentence)
	cap_type = "trend_dec"
	sentence = get_trend_sentence(name, begin, end, begin_value, end_value, cap_type = cap_type)
	print(cap_type, sentence)
	sentence = get_compare_sentence(name, "India's GDP", 12, 13, "2001")
	print("compare", sentence)

	data = generate_rule_data('ocq', 'load_group_bar_chart','ocq_common')
	print(data)
	if current_system == "linux":
		new_sentence = get_checked_sentence(sentence)
		print("New-sentence", new_sentence)