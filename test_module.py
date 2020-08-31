import torch
import torch.nn.functional as F
import numpy as np
import json
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import skimage.transform
import argparse
from skimage.io import imread
from skimage.transform import resize as imresize
from PIL import Image
from utils import svg_read, add_image_focus
import bs4
import os
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

encoder = "undefine"
decoder = "undefine"
word_map = []
rev_word_map = []
max_element_number = 0


def get_pixel_image_from_file(image_path):
    # Read image and process
    img = imread(image_path)
    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]
        img = np.concatenate([img, img, img], axis=2)
    img = imresize(img, (256, 256))
    img = img.transpose(2, 0, 1)
    img = img / 255.
    img = torch.FloatTensor(img).to(device)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([normalize])
    image = transform(img)  # (3, 256, 256)
    return image

def deal_with_soup(soup, image, image_text, encoder, decoder, word_map, rev_word_map, beam_size = 5):
   
    k = beam_size
    vocab_size = len(word_map)
    # rev_replace = {v: k for k, v in replace_dict.items()}
    # encoded_image_text = [word_map.get(word, 0) for word in img_text]
    # 注意，此时的image text 还是原始的文字
    # 此时需要将文字转化成为相应的对象
    print("image_size", image.shape)

    time_start=time.time()

    # print("element_number")
    image = image.unsqueeze(0)  # (1, 13, 40)
    image_text = image_text.unsqueeze(0) # 添加一个 (1, 40)
    encoder_out = encoder(image)  # (1, enc_image_size, encoder_dim)
    encoded_image_text = decoder.embedding(image_text) # (1, 40, 512)
    encoder_out = torch.cat((encoder_out, encoded_image_text), 2)
    enc_image_size = encoder_out.size(1)
    # print("encoder_out.shape", encoder_out.shape)
    encoder_dim = encoder_out.size(-1) # 


    # Flatten encoding
    encoder_out = encoder_out.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
    num_pixels = encoder_out.size(1)

    # We'll treat the problem as having a batch size of k
    encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)  # (k, num_pixels, encoder_dim)

    # Tensor to store top k previous words at each step; now they're just <start>
    k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(device)  # (k, 1)

    # Tensor to store top k sequences; now they're just <start>
    seqs = k_prev_words  # (k, 1)

    # Tensor to store top k sequences' scores; now they're just 0
    top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

    #
    seqs_alpha = torch.ones(k, 1, enc_image_size).to(device)  # (k, 1, enc_image_size, enc_image_size)

    # Lists to store completed sequences, their alphas and scores
    complete_seqs = list()
    complete_seqs_alpha = list()
    complete_seqs_scores = list()

    # Start decoding
    step = 1
    h, c = decoder.init_hidden_state(encoder_out)

    # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
    while True:
        embeddings = decoder.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)
        awe, alpha = decoder.attention(encoder_out, h)  # (s, encoder_dim), (s, num_pixels)
        alpha = alpha.view(-1, enc_image_size)

        # print("alpha.shape", alpha.shape)

        gate = decoder.sigmoid(decoder.f_beta(h))  # gating scalar, (s, encoder_dim)
        awe = gate * awe
        h, c = decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))  # (s, decoder_dim)
        scores = decoder.fc(h)  # (s, vocab_size)
        scores = F.log_softmax(scores, dim=1)

        # Add
        scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

        # For the first step, all k points will have the same scores (since same k previous words, h, c)
        if step == 1:
            top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
        else:
            # Unroll and find top scores, and their unrolled indices
            top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

        # Convert unrolled indices to actual indices of scores
        prev_word_inds = top_k_words / vocab_size  # (s)
        next_word_inds = top_k_words % vocab_size  # (s)

        # Add new words to sequences, alphas
        seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)
      
        seqs_alpha = torch.cat([seqs_alpha[prev_word_inds], alpha[prev_word_inds].unsqueeze(1)],
                               dim=1)  # (s, step+1, enc_image_size, enc_image_size)

        # Which sequences are incomplete (didn't reach <end>)?
        incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                           next_word != word_map['<end>']]
        complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

        # Set aside complete sequences
        if len(complete_inds) > 0:
            complete_seqs.extend(seqs[complete_inds].tolist())
            complete_seqs_alpha.extend(seqs_alpha[complete_inds].tolist())
            complete_seqs_scores.extend(top_k_scores[complete_inds])
        k -= len(complete_inds)  # reduce beam length accordingly

        # Proceed with incomplete sequences
        if k == 0:
            break
        seqs = seqs[incomplete_inds]
        seqs_alpha = seqs_alpha[incomplete_inds]
        h = h[prev_word_inds[incomplete_inds]]
        c = c[prev_word_inds[incomplete_inds]]
        encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
        top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
        k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

        # Break if things have been going on too long
        if step > 50:
            break
        step += 1

    # print("complete_seqs_scores from svg", complete_seqs_scores)
    sorted_seqs_scores = sorted(complete_seqs_scores, reverse = True)
    # print("sorted_seqs_scores", sorted_seqs_scores)
    seqs = [complete_seqs[complete_seqs_scores.index(seq_score)] for seq_score in sorted_seqs_scores]
    alphas_of_seqs = [complete_seqs_alpha[complete_seqs_scores.index(seq_score)] for seq_score in sorted_seqs_scores]

    time_end=time.time()
    print('time cost',time_end-time_start,'s')

    return seqs, alphas_of_seqs, soup, sorted_seqs_scores


def visualize_att_svg(soup, element_number, image_path, seq, alphas, rev_word_map, output_file = "tmp.jpg", smooth=True, replace_dict = {}):
    """
    Visualizes caption with weights at every word.

    Adapted from paper authors' repo: https://github.com/kelvinxu/arctic-captions/blob/master/alpha_visualization.ipynb

    :param image_path: path to image that has been captioned
    :param seq: caption
    :param alphas: weights
    :param rev_word_map: reverse word mapping, i.e. ix2word
    :param smooth: smooth weights?
    """
    # image = Image.open(image_path)
    # image = image.resize([14 * 24, 14 * 24], Image.LANCZOS)
    # print("seq", seq)
    words = [rev_word_map[ind] for ind in seq]
    rev_replace_dict = {v: k for k, v in replace_dict.items()}
    # print("words before", words)
    # print(rev_replace_dict)
    for i, word in enumerate(words):
        if word in rev_replace_dict:
            words[i] = rev_replace_dict[word]
    # print("words after", words)
    soup_list = []
    for t in range(len(words)):
        if t > 50:
            break
        # print("hhhhh")

        current_alpha = alphas[t, :]
        current_alpha = current_alpha.view(current_alpha.shape[0], 1)
        current_alpha_numpy = current_alpha.numpy().reshape(current_alpha.shape[0])
        # print(np.max(current_alpha_numpy), np.min(current_alpha_numpy))
        # print("element_number", element_number)
        # print("current_alpha_numpy ", current_alpha_numpy)
        current_alpha_max = np.max(current_alpha_numpy[0:element_number])
        
        for i in range(element_number):
            current_element = soup.findAll(attrs = {"caption_id":  str(i)})[0]
            current_element["opacity"] = current_alpha_numpy[i] / current_alpha_max
        soup_string = str(soup.select("svg")[0])
        # print(soup_string)
        # print(f"第{t}个soup", soup_string)
        new_soup = bs4.BeautifulSoup(soup_string, "html5lib")
        soup_list.append(new_soup)


    # print(soup_list)
    # print(soup)

    adjust_width = 150

    empty_string = '<!--?xml version="1.0" encoding="utf-8" ?--><html><head></head><body></body></html>'
    soup_total = bs4.BeautifulSoup(empty_string, "html5lib")
    body = soup_total.select("body")[0]
    for i, new_soup in enumerate(soup_list):
        # current_div = body.new_tag("div")
        # print(new_soup.select['svg'])
        # print("new_soup", new_soup)
        width = float(new_soup.select("svg")[0]["width"])
        height = float(new_soup.select('svg')[0]["height"])
        # print(width, height)
        current_svg = new_soup.select("svg")[0]
        if not current_svg.has_attr("viewBox"):
            current_svg["viewBox"] = f'0 0 {width} {height}'
        current_svg['width'] = adjust_width
        current_svg["height"] = adjust_width / width * height
        rect_str = f'<rect width="{width}" height="{height}" stroke-width="2px" fill-opacity="0"></rect>'
        text_str = f'<text x="250" y="150" font-family="Verdana" text-anchor="middle" font-size="55">{words[i]}</text>'
        new_soup.select("svg")[0].append(bs4.BeautifulSoup(text_str, 'html.parser'))
        body.append(new_soup)
    # print(soup_list)
        
    
    # plt.savefig(output_file)

    with open(f"{output_file}.html", "w") as file:
        file.write(str(soup_total))

def read_model(model_path):
    checkpoint = torch.load(model_path, map_location=str(device))
    decoder = checkpoint['decoder']
    decoder = decoder.to(device)
    decoder.eval()
    encoder = checkpoint['encoder']
    encoder = encoder.to(device)
    encoder.eval()

    return encoder, decoder

def read_word_map(wora_map_path):
    with open(wora_map_path, 'r') as j:
        word_map = json.load(j)
    rev_word_map = {v: k for k, v in word_map.items()}  # ix2word
    return word_map, rev_word_map

def init_model(model_path, word_map_path, max_ele_num = 100):
    
    encoder, decoder = read_model(model_path = model_path)
    word_map, rev_word_map = read_word_map(word_map_path)

    return encoder, decoder, word_map, rev_word_map


def pre_process_svg(img, soup, image_text, wordmap, replace_token = False):
    print("image_text", image_text)
    replace_dict = {}
    if replace_token:
        for word in image_text:
            if word != "" and word != "<pad>":
                current_token = "<#" + str(len(replace_dict)) + ">"
                replace_dict[word] = current_token
            # else:

        encoded_image_text = [word_map.get(replace_dict.get(word, ""), 0) for word in image_text]
    else:
        encoded_image_text = [wordmap.get(word, 0) for word in image_text]

    print("encoded_image_text", encoded_image_text)
    element_number = sum([item != "<pad>" for item in image_text])
    img = torch.FloatTensor(img).to(device)
    encoded_image_text = torch.LongTensor(encoded_image_text).to(device)
    return img, soup, element_number, encoded_image_text, replace_dict

def process_svg_string(svg_string):
    print("svg_string, ", svg_string)
    image, soup, element_number, image_text, replace_dict = parse_svg_string(svg_string, need_text = True, wordmap = word_map, max_element_number = max_element_number)
    seqs, alphas, soup, scores = deal_with_soup(soup, image, image_text, encoder, decoder, word_map, rev_word_map)
    return seqs, alphas, scores, soup, element_number, replace_dict

def process_image(image_path):
    f = open(image_path)
    svg_string = f.read()
    seqs, alphas, scores, soup, element_number = process_svg_string(svg_string)
    return seqs, alphas, scores, soup, element_number

# This is the old code.

# def parse_svg_string(svg_string, need_text, wordmap, max_element_number, replace_token = False):
#     # img = np.random.random_sample((20, 10))
#     # print("need_text or not ", need_text)
#     img, soup, image_text = svg_read(svg_string = svg_string, need_soup = True, need_text = True, svg_number = max_element_number, use_svg_string = True)
#     return pre_process_svg(img, soup, image_text, wordmap)



def parse_svg_string(svg_string, need_text, wordmap, max_element_number, replace_token = False, need_focus = False, focus = []):
    # img = np.random.random_sample((20, 10))
    # print("need_text or not ", need_text)

    img, soup, image_text = svg_read(svg_string = svg_string, need_soup = True, need_text = True, svg_number = max_element_number, use_svg_string = True)
    # elements = soup.findAll(attrs = {"caption_sha", "5"})
    # elements = soup.findAll(attrs = {"caption_id":  "2"})

    # print(img.shape)
    if need_focus:
        print("Currently, the result needs focus!")
        img = add_image_focus(img, focus)

    return pre_process_svg(img, soup, image_text, wordmap, replace_token = replace_token)

def get_svg_string_from_file(image_path):
    f = open(filename)
    # print("open file", filename)
    svg_string = f.read()
    return svg_string

def run_model_file(image_path, encoder, decoder, word_map, rev_word_map, max_element_number = 100, replace_token = False, need_focus = False, focus = []):
    svg_string = get_svg_string_from_file(image_path)

    return run_model_with_svg_string(svg_string, encoder, decoder, word_map, rev_word_map, max_element_number = max_element_number, replace_token = args.replace_token, need_focus = args.need_focus, focus = args.focus)

    # image, soup, element_number, image_text, replace_dict = parse_svg_string(svg_string, need_text = True, wordmap = word_map, max_element_number = max_element_number, replace_token = replace_token, need_focus = need_focus, focus = focus)

    # print(replace_dict)
    # seqs, alphas, soup, scores = deal_with_soup(soup, image, image_text, encoder, decoder, word_map, rev_word_map)
    # return seqs, alphas, scores, soup, replace_dict, element_number

def run_model_with_svg_string(svg_string, encoder, decoder, word_map, rev_word_map, max_element_number = 100, replace_token = False, need_focus = False, focus = [])

    image, soup, element_number, image_text, replace_dict = parse_svg_string(svg_string, need_text = True, wordmap = word_map, max_element_number = max_element_number, replace_token = replace_token, need_focus = need_focus, focus = focus)

    print(replace_dict)
    seqs, alphas, soup, scores = deal_with_soup(soup, image, image_text, encoder, decoder, word_map, rev_word_map)
    return seqs, alphas, scores, soup, replace_dict, element_number

def get_word_seq_score(seqs, rev_word_map, replace_dict, scores):

    sentences = []

    assert len(seqs) == len(scores), "the length of seqs is not equal to that of scores"

    print("length", len(seqs), len(scores))

    for seq_index, seq in enumerate(seqs):

        words = [rev_word_map[ind] for ind in seq]
        rev_replace_dict = {v: k for k, v in replace_dict.items()}
        for i, word in enumerate(words):
            if word in rev_replace_dict:
                words[i] = rev_replace_dict[word]

        print("index", seq_index)
        print("sentence", words)
        print("scores", scores[seq_index])

        current_sentence = {}
        current_sentence["sentence"] = words
        current_sentence["score"] = float(scores[seq_index])
        sentences.append(current_sentence)

    print("sentences: ", sentences)

    return sentences


def parse_svg():
    svg_string = '<svg xmlns="http://www.w3.org/2000/svg" id="mySvg" width="544.9756153216193" height="500"><g transform="translate(80,80)" class="main_canvas"><g class="axis" transform="translate(0,240)" fill="none" font-size="10" font-family="sans-serif" text-anchor="middle"><g class="tick" opacity="1" transform="translate(67.5,0)"><text fill="currentColor" y="9" dy="0.71em" style="font-family: Oxygen; fill: #253039;">ord0</text></g><g class="tick" opacity="1" transform="translate(163.5,0)"><text fill="currentColor" y="9" dy="0.71em" style="font-family: Oxygen; fill: #253039;">ord1</text></g><g class="tick" opacity="1" transform="translate(259.5,0)"><text fill="currentColor" y="9" dy="0.71em" style="font-family: Oxygen; fill: #253039;">ord2</text></g></g><g class="axis" fill="none" font-size="10" font-family="sans-serif" text-anchor="end"><g class="tick" opacity="1" transform="translate(0,240.5)"><line stroke="currentColor" x2="326.9853691929716"/><text fill="currentColor" x="-3" dy="0.32em" style="font-family: Oxygen; fill: #253039;">0</text></g><g class="tick" opacity="1" transform="translate(0,197.5)"><line stroke="currentColor" x2="326.9853691929716" style="stroke-opacity: 0.3;"/><text fill="currentColor" x="-3" dy="0.32em" style="font-family: Oxygen; fill: #253039;">20</text></g><g class="tick" opacity="1" transform="translate(0,155.5)"><line stroke="currentColor" x2="326.9853691929716" style="stroke-opacity: 0.3;"/><text fill="currentColor" x="-3" dy="0.32em" style="font-family: Oxygen; fill: #253039;">40</text></g><g class="tick" opacity="1" transform="translate(0,112.5)"><line stroke="currentColor" x2="326.9853691929716" style="stroke-opacity: 0.3;"/><text fill="currentColor" x="-3" dy="0.32em" style="font-family: Oxygen; fill: #253039;">60</text></g><g class="tick" opacity="1" transform="translate(0,69.5)"><line stroke="currentColor" x2="326.9853691929716" style="stroke-opacity: 0.3;"/><text fill="currentColor" x="-3" dy="0.32em" style="font-family: Oxygen; fill: #253039;">80</text></g><g class="tick" opacity="1" transform="translate(0,26.5)"><line stroke="currentColor" x2="326.9853691929716" style="stroke-opacity: 0.3;"/><text fill="currentColor" x="-3" dy="0.32em" style="font-family: Oxygen; fill: #253039;">100</text></g></g><text transform="translate(-35, 200) rotate(-90)" text-anchor="start" font-size="20px"/><g class="bars"><rect fill="#ffff99" id="0" class="element_0 elements ordinary" x="41" y="41" height="199" width="17.1" rx="0" ry="0"/><rect fill="#ffff99" id="1" class="element_1 elements ordinary" x="137" y="161" height="79" width="17.1" rx="0" ry="0"/><rect fill="#ffff99" id="2" class="element_2 elements ordinary" x="233" y="78" height="162" width="17.1" rx="0" ry="0"/><rect fill="#7fc97f" id="3" class="element_3 elements ordinary" x="60" y="18" height="222" width="17.1" rx="0" ry="0"/><rect fill="#7fc97f" id="4" class="element_4 elements ordinary" x="156" y="0" height="240" width="17.1" rx="0" ry="0"/><rect fill="#7fc97f" id="5" class="element_5 elements ordinary" x="252" y="56" height="184" width="17.1" rx="0" ry="0"/><rect fill="#beaed4" id="6" class="element_6 elements ordinary" x="79" y="65" height="175" width="17.1" rx="0" ry="0"/><rect fill="#beaed4" id="7" class="element_7 elements ordinary" x="175" y="177" height="63" width="17.1" rx="0" ry="0"/><rect fill="#beaed4" id="8" class="element_8 elements ordinary" x="271" y="4" height="236" width="17.1" rx="0" ry="0"/></g><g transform="translate(326.9853691929716,0)" class="legend-wrap"><g transform="translate(0,0)"><rect width="12.959999999999999" height="12.959999999999999" fill="#ffff99" id="color-0" color-data="#ffff99" custom-id="0" data-toggle="popover" data-container="body" data-placement="right" onclick="addColorPicker(this)"/><text x="15.12" y="10.799999999999999" text-anchor="start" font-size="10.799999999999999">item0</text></g><g transform="translate(0,14.399999999999999)"><rect width="12.959999999999999" height="12.959999999999999" fill="#7fc97f" id="color-1" color-data="#7fc97f" custom-id="1" data-toggle="popover" data-container="body" data-placement="right" onclick="addColorPicker(this)"/><text x="15.12" y="10.799999999999999" text-anchor="start" font-size="10.799999999999999">item1</text></g><g transform="translate(0,28.799999999999997)"><rect width="12.959999999999999" height="12.959999999999999" fill="#beaed4" id="color-2" color-data="#beaed4" custom-id="2" data-toggle="popover" data-container="body" data-placement="right" onclick="addColorPicker(this)"/><text x="15.12" y="10.799999999999999" text-anchor="start" font-size="10.799999999999999">item2</text></g></g><text class="title" text-anchor="middle" font-size="28.799999999999997" x="163.4926845964858" y="-38.4" style="font-family: Oxygen; font-weight: bold; fill: #253039;">THE GDP</text></g></svg>'




if __name__ == '__main__':

    # global max_element_number
    parser = argparse.ArgumentParser(description='Show, Attend, and Tell - Tutorial - Generate Caption')

    parser.add_argument('--img', '-i', default = "", help='path to image')
    parser.add_argument('--model', '-m', default = "checkpoint/chart_5_cap_5_min_wf-2020-04-18-17-03/epoch_39.pth.tar",  help='path to model')
    parser.add_argument('--word_map', '-wm', default = "data/20200418_dataset_bar/deal_run/WORDMAP_chart_5_cap_5_min_wf.json", help='path to word map JSON')
    parser.add_argument('--beam_size', '-b', default=5, type=int, help='beam size for beam search')
    parser.add_argument('--dont_smooth', dest='smooth', action='store_false', help='do not smooth alpha overlay')
    parser.add_argument('--image_type', type=str, default = 'svg', help='image type as input')
    parser.add_argument('--need_text', action='store_true', help="decide whether need text")
    parser.add_argument('--max_element_number', '-e', default=100, type=int, help='maximum element number')
    parser.add_argument('--port', '-p', default=9999, type=int, help='maximum element number')
    parser.add_argument('--replace_token', action = "store_true", help="replace token")
    parser.add_argument('--result_file', default = "tmp.json", help = "temperal file to store the results")
    parser.add_argument('--need_focus', action = "store_true", help = "Using focus as input")
    parser.add_argument('--focus', default = '0,1,2', help = "The array of focused id of the chart")

    args = parser.parse_args()
    args.need_text = True

    args.focus = [int(i) for i in args.focus.split(",")]

    output_file = "data/" + args.model.split('/')[-2] + "/result_of_" +  args.img.split("/")[-1] + "/"
    print(output_file)
    os.system(f"mkdir -p {output_file}")

    max_element_number = args.max_element_number

    model_path = args.model
    word_map_path = args.word_map
    image_path = args.img 

    encoder, decoder, word_map, rev_word_map = init_model(model_path, word_map_path, max_ele_num = max_element_number)

    seqs, alphas, scores, soup, replace_dict, element_number = run_model_file(image_path, encoder, decoder, word_map, rev_word_map, max_element_number = max_element_number, replace_token = args.replace_token, need_focus = args.need_focus, focus = args.focus)

    sentences = get_word_seq_score(seqs, rev_word_map, replace_dict, scores)

    with open(args.result_file, "w") as f:
        json.dump(sentences, f, indent = 2)

    # 可视化到相应的文件目录
    # seqs, alphas, scores, soup, element_number, rev_word_map, replace_dict = run_model_file(model_path, word_map_path, image_path, max_element_number = max_element_number, replace_token = args.replace_token)
    alphas = [torch.FloatTensor(alpha) for alpha in alphas]


    for i, seq in enumerate(seqs):
        visualize_att_svg(soup, element_number, args.img, seq, alphas[i], rev_word_map, output_file + str(i), args.smooth, replace_dict = replace_dict)


