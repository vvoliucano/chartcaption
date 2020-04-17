# This file is to generate datasets with features.

from sentence_generator.sentence_generator import generate_sentence_by
from generate_data.data_generator import get_data
import json
import os
import shutil
import argparse
import random
from tqdm import tqdm


def get_data_sentence():
    data = get_data("rule")
    sentences = []
    for sentence in data['pre_gen_focus']:
        focus_id = sentence['focus_id']
        compare_id = sentence['compare_id']
        major_name = data['major_name']
        second_name = data['second_name']
        answers = generate_sentence_by(data, focus_id, compare_id, major_name, second_name)
        for answer in answers:
            if answer['type'] == sentence['type']:
                sentences.append(answer)
                break;
    data['sentences'] = sentences
    return data

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
        sentences = item["sentences"]
        here_sentence_num = len(sentences)
        image = {
            "sentids": [],
            "imgid": pindex,
            "sentences": [],
            "split": gen_split(),
            "filename": item["filename"]
        }
        for sen_index, sen in enumerate(sentences):
            sentid = cur_sentence_id + sen_index
            sen = sen["sentence"].replace(",", " ,").replace(".", " .")
            # print(sen)
            sentence = {
                "tokens": sen.split(" "),
                "raw": sen,
                "imgid": pindex,
                "sentid": sentid,
            }
            image["sentids"].append(sentid)
            image["sentences"].append(sentence)
        res["images"].append(image)
        cur_sentence_id += here_sentence_num
    return res

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Show, Attend, and Tell - Tutorial - Generate Caption for SVG')
    parser.add_argument('--number', '-n', type=int, default = 10, help='number')
    parser.add_argument('--path', '-p', type=str, default = "svg", help='The path')
    # parser.add_argument()
    args = parser.parse_args()

    setting_dir = args.path
    number = args.number
    if os.path.isdir(setting_dir):
        print(f"The file {setting_dir} already exists, so we can delete them.")
        shutil.rmtree(setting_dir)

    os.mkdir(setting_dir)

    # dataset_name = os.path.join(setting_dir, "origin_data.json")
    karparthy_file = os.path.join(setting_dir, "karparthy_dataset.json")
    json_dir = os.path.join(setting_dir, "json")
    svg_dir = os.path.join(setting_dir, "svg")

    os.mkdir(json_dir) 
    os.mkdir(svg_dir)

    # dataset_name = "try_set.json"
    output_data_set = []
    for i in tqdm(range(number)):
        current_data = get_data_sentence()
        svg_filename = str(i).zfill(6) + ".svg"
        current_data["filename"] = svg_filename
        svg_filename = os.path.join(svg_dir, svg_filename)
        output_data_set.append(current_data)

        current_filename = os.path.join(json_dir, str(i).zfill(6) + ".json")
        
        with open(current_filename, "w") as f:
            json.dump(current_data, f, indent = 2)
        os.system(f"./gen_svg.js --input {current_filename} --output {svg_filename}", )


    karparthy_dataset = convert_to_karparthy(output_data_set)

    # print(karparthy_dataset)

    with open(karparthy_file, "w") as f:
        json.dump(karparthy_dataset, f, indent = 2)
    # with open(dataset_name, "w") as f:
    #     json.dump(output_data_set, f, indent=2)
