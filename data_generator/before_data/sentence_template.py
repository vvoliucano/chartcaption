import json
import language_check

import time
import random

# a = f'sdf{sdf}'

# a


def get_checked_sentence(text = "China's GDP increase from 2019 to 2020."):

	tool = language_check.LanguageTool('en-US')
	# text = u'A sentence with a error in the Hitchhiker’s Guide tot he Galaxy'
	matches = tool.check(text)
	# print(matches)
	new_text = language_check.correct(text, matches)
	return new_text


f = open('template.json')
sentence_template = json.load(f)


def get_trend_sentence(name, begin, end, begin_value, end_value, cap_type = "trend_inc"):


	# sen1 = sentence_template["trend_inc"][0].format(name = "{name}", begin = begin, end = end, end_value = end_value)
	# sen2 = sen1.format(name = "this value")
	# print(sen1)
	# print(sen2)

	template_choice = sentence_template[cap_type]
	current_template = random.choice(template_choice)

	sentence = current_template.format(name = name, begin = begin, end = end, end_value = end_value)

	return sentence



if __name__ == '__main__':

	begin = 2001
	end = 2020
	begin_value = 20
	end_value = 200
	cap_type = "trend_inc"
	sentence = get_trend_sentence("China's GDP", begin, end, begin_value, end_value, cap_type = cap_type)

	print("Sentence: ", sentence)

	new_sentence = get_checked_sentence(sentence)
	print("new_sentence", new_sentence)