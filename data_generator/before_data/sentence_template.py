import json
import language_check

import time

# a = f'sdf{sdf}'

# a


def get_checked_sentence(text = "China's GDP increase from 2019 to 2020."):

	t1 = time.time()
	tool = language_check.LanguageTool('en-US')
	t2 = time.time()
	# text = u'A sentence with a error in the Hitchhiker’s Guide tot he Galaxy'
	matches = tool.check(text)
	# print(matches)
	new_text = language_check.correct(text, matches)
	t3 = time.time()

	print("inter", t2 - t1)
	print('inter', t3 - t2)
	return new_text


f = open('template.json')
sentence_template = json.load(f)


def get_trend_sentence(name, begin, end, begin_value, end_value):
	sen1 = sentence_template["trend_inc"][0].format(name = "{name}", begin = begin, end = end, end_value = end_value)
	sen2 = sen1.format(name = "this value")
	# print(sen1)
	# print(sen2)

	template_choice = sentence_template["trend_inc"]

	return sen2



if __name__ == '__main__':

	begin = 2001
	end = 2020
	begin_value = 20
	end_value = 200
	sentence = get_trend_sentence("China's GDP", begin, end, begin_value, end_value)
	print("original_sentence", sentence)

	new_sentence = get_checked_sentence(sentence)
	print("new_sentence", new_sentence)