import json
import random
import gen_svg
import os
import shutil
import argparse

parser = argparse.ArgumentParser(description='Show, Attend, and Tell - Tutorial - Generate Caption for SVG')
parser.add_argument('--number', '-n', type=int, default = 1000, help='number')
parser.add_argument('--path', '-p', type=str, default = "svg", help='path')

args = parser.parse_args()

sen_count = args.number

print("输出数目：", sen_count)

svg_out_dir = args.path


# color_set = ["red", "orange", "yellow", "green", "blue", "cyan", "purple"]
color_set = ["#FF0000", "#FFA500", "#FFFF00", "#008000", "#0000FF", "#00FFFF", "#800080"]
max_bar_count = 10
max_bar_height = 400
canvas_width = 500
canvas_height = max_bar_height + 10
bar_border = 0.1 # left + right
bar_padding = 0.1 # sum of all internal

# trends = ["up", "down", "keep", "up_down", "down_up", "up_keep", ]
trends = ["up", "down", "keep"]
max_sentence_parts = [["The maximum occur"], ["the maximum"]]
min_sentence_parts = [["The minimum occur"], ["the minimum"]]

up1_factor = 5 # 上升的比例
up2_factor = 20 # 明显上升的比例
keep_factor = up1_factor # 保持不变的比例

def make_sure_dir(dirname):
    if os.path.exists(dirname):
        shutil.rmtree(dirname)
    os.mkdir(dirname)

def get_max_min(bar_values):
    max_value = -2
    min_value = 2
    max_idx = []
    min_idx = []
    for idx, value in enumerate(bar_values):
        if value > max_value:
            max_value = value
            max_idx = [idx]
        elif value == max_value:
            max_idx.append(idx)
        if value < min_value:
            min_value = value
            min_idx = [idx]
        elif value == min_value:
            min_idx.append(idx)
    return max_value, min_value, max_idx, min_idx

def gen_rank(n, x_axis_list):
    return x_axis_list[n]
    # return f"<#{n+1}>"

def gen_mm_sentence(idxs, max0_or_min1, x_axis_list):
    selection = random.randint(0, 1)
    if max0_or_min1 == 0:
        sen_part_1 = max_sentence_parts[selection][random.randint(0, len(max_sentence_parts[selection])-1)]
    else:
        sen_part_1 = min_sentence_parts[selection][random.randint(0, len(min_sentence_parts[selection])-1)]
    idxslen = len(idxs)
    sen_part_2 = ""
    if idxslen == 1:
        sen_part_2 = gen_rank(idxs[0], x_axis_list)
    elif idxslen == 2:
        sen_part_2 = f"{gen_rank(idxs[0], x_axis_list)} and {gen_rank(idxs[1], x_axis_list)}"
    else:
        for i in range(0, idxslen-1):
            sen_part_2 += f"{gen_rank(idxs[i], x_axis_list)}, "
        sen_part_2 += f"and {gen_rank(idxs[idxslen-1], x_axis_list)}"
    sentence = ""
    if idxslen == 1:
        if selection == 0:
            sentence = sen_part_1 + "s in " + sen_part_2
        else:
            sentence = sen_part_2 + " is " + sen_part_1
    else:
        if selection == 0:
            sentence = sen_part_1 + "s in " + sen_part_2
        else:
            sentence = sen_part_2 + " are " + sen_part_1
    return sentence

def gen_max_min_sentence(bar_values, x_axis_list):
    max_value, min_value, max_idx, min_idx = get_max_min(bar_values)
    sen1 = gen_mm_sentence(max_idx, 0, x_axis_list = x_axis_list)
    sen2 = gen_mm_sentence(min_idx, 1, x_axis_list = x_axis_list)
    return sen1, sen2

def gen_keep_values(start, count):
    res = [start]
    pos_neg = [1, -1]
    for i in range(count):
        next_value = res[-1] * (1 + random.choice(pos_neg) * random.randint(0, keep_factor) / 100)
        res.append(next_value)
    return res

def gen_up_values(start, count):
    res = [start]
    for i in range(count):
        next_value = res[-1] * (1 + random.randint(up1_factor, up2_factor) / 100)
        res.append(next_value)
    return res

def gen_down_values(start, count):
    res = [start]
    for i in range(count):
        next_value = res[-1] * (1 - random.randint(up1_factor, up2_factor) / 100)
        res.append(next_value)
    return res

## add new process function process_trend_{trend_name}
def process_trend_up(count, x_axis_list):
    start_value = max(0.5, random.random())
    bar_ratio = gen_up_values(start_value, count-1)
    sen_max, sen_min = gen_max_min_sentence(bar_ratio, x_axis_list)
    return bar_ratio, ["the [color] bar goes up", sen_max, sen_min]

def process_trend_down(count, x_axis_list):
    start_value = max(0.5, random.random())
    bar_ratio = gen_down_values(start_value, count-1)
    sen_max, sen_min = gen_max_min_sentence(bar_ratio, x_axis_list)
    return bar_ratio, ["the [color] bar goes down", sen_max, sen_min]

def process_trend_keep(count, x_axis_list):
    start_value = max(0.5, random.random())
    bar_ratio = gen_keep_values(start_value, count-1)
    sen_max, sen_min = gen_max_min_sentence(bar_ratio, x_axis_list)
    return bar_ratio, ["the [color] bar goes steady", sen_max, sen_min]

## register here
process_func = {
    "up": process_trend_up,
    "down": process_trend_down,
    "keep": process_trend_keep,
}

gen_ratio_func = {
    "up": gen_up_values,
    "down": gen_down_values,
    "keep": gen_keep_values,
}

def assign_color(sentence, color):
    sentence = sentence.split(" ")
    sentence[1] = color
    sentence = " ".join(sentence)
    return sentence

def process_trend(count, trend, color, x_axis_list):
    bar_ratio = []
    bar_sentences = []
    if trend in process_func:
        # print(trend)
        trend_func = process_func[trend]
        bar_ratio, bar_sentences = trend_func(count, x_axis_list)
        bar_sentences[0] = assign_color(bar_sentences[0], color)
    else:
        print("not support now", trend)
    return bar_ratio, bar_sentences

def get_trend_name(trend):
    if trend != "keep":
        return trend
    return "steady"

def process_two_trends(count, trend1, trend2, color, x_axis_list):
    # print(trend1, trend2)
    bar_ratio = []
    bar_sentences = []
    if trend1 == trend2:
        bar_ratio, bar_sentences = process_func[trend1](count, x_axis_list)
        bar_sentences[0] = assign_color(bar_sentences[0], color)
    else:
        mid_bar = random.randint(1, count-2)
        sen_trend = f"the {color} bar goes {get_trend_name(trend1)} from {gen_rank(0, x_axis_list = x_axis_list)} to {gen_rank(mid_bar, x_axis_list = x_axis_list)} and goes {get_trend_name(trend2)} from {gen_rank(mid_bar, x_axis_list = x_axis_list)} to {gen_rank(count-1, x_axis_list = x_axis_list)}"
        start_value = max(0.5, random.random())
        bar_ratio1 = gen_ratio_func[trend1](start_value, mid_bar)
        bar_ratio2 = gen_ratio_func[trend2](bar_ratio1[-1], count-mid_bar-1)[1:]
        bar_ratio = bar_ratio1 + bar_ratio2
        sen_max, sen_min = gen_max_min_sentence(bar_ratio, x_axis_list)
        bar_sentences = [sen_trend, sen_max, sen_min]
    return bar_ratio, bar_sentences

def gen_bars(max_bar_count, max_bar_height, trend1, trend2, x_axis_list = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]):
    half_max_bar_count = max_bar_count // 2
    bar_count = half_max_bar_count + random.randint(0, max_bar_count - half_max_bar_count)
    bar_height = max(max_bar_height/2, max_bar_height * random.random())
    color = random.choice(color_set)
    bar_colors = [color for i in range(bar_count)]
    # bar_ratio, bar_sentences = process_trend(bar_count, trend1, bar_colors[0])
    bar_ratio, bar_sentences = process_two_trends(bar_count, trend1, trend2, bar_colors[0], x_axis_list = x_axis_list)
    bar_values = list(map(lambda x: x * bar_height, bar_ratio))
    return bar_values, bar_colors, bar_sentences

def gen_n_pairs(n, out_dir):
    pairs = []
    for i in range(n):
        trend1 = random.choice(trends)
        trend2 = random.choice(trends)
        begin = random.randint(1800, 2200)
        interval = random.randint(1, 20)
        x_axis_list = [str(item) for item in range(begin, begin + 20 * interval, interval)]
        # print(x_axis_list)
        # x_axis_list = range
        bar_values, colors, sentences = gen_bars(max_bar_count, max_bar_height, trend1, trend2, x_axis_list = x_axis_list)
        if len(sentences) == 0:
            continue
        svg_name = f"{i}.svg"
        svg_file_name = os.path.join(out_dir, svg_name)
        gen_svg.draw_barchart_with_text(bar_values, colors, canvas_width, canvas_height, bar_border, bar_padding, svg_file_name, x_axis_list)
        pairs.append([svg_name, [bar_values, sentences]])
    return pairs

def gen_tvt(train, val, test):
    def ret_func():
        rv = random.random()
        if rv < train:
            return "train"
        if rv < train + val:
            return "val"
        return "test"
    return ret_func

def trans_template(pairs):
    res = {"dataset": "weak", "images": []}
    cur_sentence_id = 0
    gen_split = gen_tvt(0.8, 0.1, 0.1)
    for pindex, pair in enumerate(pairs):
        sentences = pair[1][1]
        here_sentence_num = len(sentences)
        image = {
            "sentids": [],
            "imgid": pindex,
            "sentences": [],
            "split": gen_split(),
            "filename": pair[0]
        }
        for sen_index, sen in enumerate(sentences):
            sentid = cur_sentence_id + sen_index
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
    make_sure_dir(svg_out_dir)
    pairs = gen_n_pairs(sen_count, svg_out_dir)
    # print(pairs)
    res = trans_template(pairs)
    with open(os.path.join(svg_out_dir, "dataset.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
