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
from utils import svg_read
import bs4
import os

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

def parse_svg_string(svg_string, need_text, wordmap, max_element_number):
    # img = np.random.random_sample((20, 10))
    # print("need_text or not ", need_text)
    img, soup, image_text = svg_read(svg_string = svg_string, need_soup = True, need_text = True, svg_number = max_element_number, use_svg_string = True)
    # elements = soup.findAll(attrs = {"caption_sha", "5"})
    # elements = soup.findAll(attrs = {"caption_id":  "2"})
    print("image_text", image_text)
    encoded_image_text = [wordmap.get(word, 0) for word in image_text]
    print("image_text", encoded_image_text)
    # image_text_try = torch.LongTensor(image_text)
    # print("image text length ", len(image_text))
    # elements = soup.findAll(attrs = {"caption_sha":  "5"})
    # print("elements", elements)
    element_number = sum([item != "<pad>" for item in image_text])
    # print("element_number", element_number)
    # element_number = len()
    # print("element_number", element_number)
    # print(soup.findAll(attrs = {"caption_id":  "2"}))
    # print(soup)
    # with open("1.html", "w") as file:
    #     file.write(str(soup))
    img = torch.FloatTensor(img).to(device)
    encoded_image_text = torch.LongTensor(encoded_image_text).to(device)
    return img, soup, element_number, encoded_image_text


def get_svg_image_from_file(image_path, need_text, wordmap, max_element_number):
    # img = np.random.random_sample((20, 10))
    # print("need_text or not ", need_text)

    img, soup, image_text = svg_read(image_path, need_soup = True, need_text = True, svg_number = max_element_number)
    # elements = soup.findAll(attrs = {"caption_sha", "5"})
    # elements = soup.findAll(attrs = {"caption_id":  "2"})
    print("image_text", image_text)
    encoded_image_text = [wordmap.get(word, 0) for word in image_text]
    print("image_text", encoded_image_text)
    # image_text_try = torch.LongTensor(image_text)
    # print("image text length ", len(image_text))
    # elements = soup.findAll(attrs = {"caption_sha":  "5"})
    # print("elements", elements)
    element_number = sum([item != "<pad>" for item in image_text])
    # print("element_number", element_number)
    # element_number = len()
    # print("element_number", element_number)
    # print(soup.findAll(attrs = {"caption_id":  "2"}))
    # print(soup)
    # with open("1.html", "w") as file:
    #     file.write(str(soup))
    img = torch.FloatTensor(img).to(device)
    encoded_image_text = torch.LongTensor(encoded_image_text).to(device)
    return img, soup, element_number, encoded_image_text

def deal_with_soup(soup, image, image_text, encoder, decoder, word_map, rev_word_map, beam_size = 3):
   
    k = beam_size
    vocab_size = len(word_map)
    
    # encoded_image_text = [word_map.get(word, 0) for word in img_text]
    # 注意，此时的image text 还是原始的文字
    # 此时需要将文字转化成为相应的对象
    print("image_size", image.shape)
    # print("element_number")
    image = image.unsqueeze(0)  # (1, 13, 40)
    image_text = image_text.unsqueeze(0) # 添加一个 (1, 40)
    encoder_out = encoder(image)  # (1, enc_image_size, encoder_dim)
    encoded_image_text = decoder.embedding(image_text) # (1, 40, 512)
    encoder_out = torch.cat((encoder_out, encoded_image_text), 2)
    enc_image_size = encoder_out.size(1)
    print("encoder_out.shape", encoder_out.shape)
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

    print("complete_seqs_scores from svg", complete_seqs_scores)
    sorted_seqs_scores = sorted(complete_seqs_scores, reverse = True)
    # print("sorted_seqs_scores", sorted_seqs_scores)
    seqs = [complete_seqs[complete_seqs_scores.index(seq_score)] for seq_score in sorted_seqs_scores]
    alphas_of_seqs = [complete_seqs_alpha[complete_seqs_scores.index(seq_score)] for seq_score in sorted_seqs_scores]


    return seqs, alphas_of_seqs, soup, sorted_seqs_scores



def visualize_att(image_path, seq, alphas, rev_word_map, smooth=True):
    """
    Visualizes caption with weights at every word.

    Adapted from paper authors' repo: https://github.com/kelvinxu/arctic-captions/blob/master/alpha_visualization.ipynb

    :param image_path: path to image that has been captioned
    :param seq: caption
    :param alphas: weights
    :param rev_word_map: reverse word mapping, i.e. ix2word
    :param smooth: smooth weights?
    """
    image = Image.open(image_path)
    image = image.resize([14 * 24, 14 * 24], Image.LANCZOS)

    words = [rev_word_map[ind] for ind in seq]

    for t in range(len(words)):
        if t > 50:
            break
        print("hhhhh")
        plt.subplot(np.ceil(len(words) / 5.), 5, t + 1)

        plt.text(0, 1, '%s' % (words[t]), color='black', backgroundcolor='white', fontsize=12)
        # plt.imshow(image)
        current_alpha = alphas[t, :]
        if smooth:
            alpha = skimage.transform.pyramid_expand(current_alpha.numpy(), upscale=24, sigma=8)
        else:
            alpha = skimage.transform.resize(current_alpha.numpy(), [14 * 24, 14 * 24])
        if t == 0:
            plt.imshow(alpha, alpha=0)
        else:
            plt.imshow(alpha, alpha=0.8)
        plt.set_cmap(cm.Greys_r)
        plt.axis('off')
    plt.savefig("tmp.jpg")

def visualize_att_svg(soup, element_number, image_path, seq, alphas, rev_word_map, output_file = "tmp.jpg", smooth=True):
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
    print("seq", seq)
    words = [rev_word_map[ind] for ind in seq]
    print("words", words)
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
    global encoder
    global decoder 
    global word_map
    global rev_word_map
    global max_element_number

    encoder, decoder = read_model(model_path = model_path)
    word_map, rev_word_map = read_word_map(word_map_path)
    max_element_number = max_ele_num

def process_image(image_path):
    f = open(image_path)
        # print("open file", filename)
    svg_string = f.read()
#  

    # image, soup, element_number, image_text = parse_svg_string(svg_string, need_text = True, wordmap = word_map, max_element_number = max_element_number)
    # seqs, alphas, soup, scores = deal_with_soup(soup, image, image_text, encoder, decoder, word_map, rev_word_map)
    seqs, alphas, scores, soup, element_number = process_svg_string(svg_string)
    return seqs, alphas, scores, soup, element_number

def process_svg_string(svg_string):
    print("svg_string, ", svg_string)
    image, soup, element_number, image_text = parse_svg_string(svg_string, need_text = True, wordmap = word_map, max_element_number = max_element_number)
    seqs, alphas, soup, scores = deal_with_soup(soup, image, image_text, encoder, decoder, word_map, rev_word_map)
    return seqs, alphas, scores, soup, element_number

def run_model(model_path, word_map_path, image_path, max_element_number = 100):
    encoder, decoder = read_model(model_path = model_path)
    # Load word map (word2ix)
    word_map, rev_word_map = read_word_map(word_map_path)
    image, soup, element_number, image_text = get_svg_image_from_file(image_path, need_text = True, wordmap = word_map, max_element_number = max_element_number)
    seqs, alphas, soup, scores = deal_with_soup(soup, image, image_text, encoder, decoder, word_map, rev_word_map)
    return seqs, alphas,scores, soup, element_number, rev_word_map

def run_model_separate(model_path, word_map_path, image_path, max_element_number = 100):
    init_model(model_path, word_map_path, max_ele_num = max_element_number)
    svg_string = '<svg id="mySvg" width="800" height="350" xmlns="http://www.w3.org/2000/svg"><g transform="translate(80,35)" class="main_canvas"><g class="axis" transform="translate(0,280)" fill="none" font-size="10" font-family="sans-serif" text-anchor="middle"><g class="tick" opacity="1" transform="translate(56.5,0)"><text fill="currentColor" y="9" dy="0.71em" style="font-family: Oxygen; fill: #253039;">2010</text></g><g class="tick" opacity="1" transform="translate(144.5,0)"><text fill="currentColor" y="9" dy="0.71em" style="font-family: Oxygen; fill: #253039;">2011</text></g><g class="tick" opacity="1" transform="translate(232.5,0)"><text fill="currentColor" y="9" dy="0.71em" style="font-family: Oxygen; fill: #253039;">2012</text></g><g class="tick" opacity="1" transform="translate(320.5,0)"><text fill="currentColor" y="9" dy="0.71em" style="font-family: Oxygen; fill: #253039;">2013</text></g><g class="tick" opacity="1" transform="translate(408.5,0)"><text fill="currentColor" y="9" dy="0.71em" style="font-family: Oxygen; fill: #253039;">2014</text></g><g class="tick" opacity="1" transform="translate(496.5,0)"><text fill="currentColor" y="9" dy="0.71em" style="font-family: Oxygen; fill: #253039;">2015</text></g><g class="tick" opacity="1" transform="translate(584.5,0)"><text fill="currentColor" y="9" dy="0.71em" style="font-family: Oxygen; fill: #253039;">2016</text></g></g><g class="axis" fill="none" font-size="10" font-family="sans-serif" text-anchor="end"><g class="tick" opacity="1" transform="translate(0,280.5)"><line stroke="currentColor" x2="640"></line><text fill="currentColor" x="-3" dy="0.32em" style="font-family: Oxygen; fill: #253039;">0.0</text></g><g class="tick" opacity="1" transform="translate(0,204.5)"><line stroke="currentColor" x2="640" style="stroke-opacity: 0.3;"></line><text fill="currentColor" x="-3" dy="0.32em" style="font-family: Oxygen; fill: #253039;">0.5</text></g><g class="tick" opacity="1" transform="translate(0,129.5)"><line stroke="currentColor" x2="640" style="stroke-opacity: 0.3;"></line><text fill="currentColor" x="-3" dy="0.32em" style="font-family: Oxygen; fill: #253039;">1.0</text></g><g class="tick" opacity="1" transform="translate(0,53.5)"><line stroke="currentColor" x2="640" style="stroke-opacity: 0.3;"></line><text fill="currentColor" x="-3" dy="0.32em" style="font-family: Oxygen; fill: #253039;">1.5</text></g></g><text transform="translate(-35, 175) rotate(-90)" text-anchor="start" font-size="20px"></text><g><g fill="#fb8072"><rect class="element_0" id="0" o0="2010" c0="F" q0="0.5916307422726574" x="21" y="210" height="70" width="70"></rect><rect class="element_1" id="1" o0="2011" c0="F" q0="0.499292187816756" x="109" y="221" height="59" width="70"></rect><rect class="element_2" id="2" o0="2012" c0="F" q0="0.5833698942964443" x="197" y="211" height="69" width="70"></rect><rect class="element_3" id="3" o0="2013" c0="F" q0="0.580821550862816" x="285" y="211" height="69" width="70"></rect><rect class="element_4" id="4" o0="2014" c0="F" q0="0.5778160391056617" x="373" y="212" height="68" width="70"></rect><rect class="element_5" id="5" o0="2015" c0="F" q0="0.5374138350916083" x="461" y="216" height="64" width="70"></rect><rect class="element_6" id="6" o0="2016" c0="F" q0="0.5190015788311658" x="549" y="219" height="61" width="70"></rect></g><g fill="#d9d9d9"><rect class="element_7" id="7" o0="2010" c0="B" q0="1" x="21" y="92" height="118" width="70"></rect><rect class="element_8" id="8" o0="2011" c0="B" q0="1.0735472169341573" x="109" y="94" height="127" width="70"></rect><rect class="element_9" id="9" o0="2012" c0="B" q0="1.1706944287796168" x="197" y="73" height="138" width="70"></rect><rect class="element_10" id="10" o0="2013" c0="B" q0="1.2575450048813972" x="285" y="63" height="148" width="70"></rect><rect class="element_11" id="11" o0="2014" c0="B" q0="1.3372426823618313" x="373" y="53" height="159" width="70"></rect><rect class="element_12" id="12" o0="2015" c0="B" q0="1.7954980378161547" x="461" y="4" height="212" width="70"></rect><rect class="element_13" id="13" o0="2016" c0="B" q0="1.848192767898499" x="549" y="0" height="219" width="70"></rect></g></g><g transform="translate(640,0)" class="legend-wrap"><g transform="translate(0,0)"><rect width="15.120000000000001" height="15.120000000000001" fill="#fb8072" id="color-0" color-data="#fb8072" custom-id="0" data-toggle="popover" data-container="body" data-placement="right" onclick="addColorPicker(this)"></rect><text x="17.64" y="12.600000000000001" text-anchor="start" font-size="12.600000000000001">F</text></g><g transform="translate(0,16.8)"><rect width="15.120000000000001" height="15.120000000000001" fill="#d9d9d9" id="color-1" color-data="#d9d9d9" custom-id="1" data-toggle="popover" data-container="body" data-placement="right" onclick="addColorPicker(this)"></rect><text x="17.64" y="12.600000000000001" text-anchor="start" font-size="12.600000000000001">B</text></g></g><text class="title" text-anchor="middle" font-size="33.6" x="320" y="-44.8" style="font-family: Oxygen; font-weight: bold; fill: #253039;">THE VALUE</text></g></svg>'
    seqs, alphas, scores, soup, element_number = process_svg_string(svg_string)
    return seqs, alphas, scores, soup, element_number



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
    
    args = parser.parse_args()
    args.need_text = True

    output_file = "data/" + args.model.split('/')[-2] + "/result_of_" +  args.img.split("/")[-1] + "/"
    print(output_file)
    os.system(f"mkdir -p {output_file}")

    

    max_element_number = args.max_element_number

    model_path = args.model
    word_map_path = args.word_map
    image_path = args.img 

    seqs, alphas, scores, soup, element_number =  run_model_separate(model_path, word_map_path, image_path, max_element_number)
    
    print(seqs)


    # 可视化到相应的文件目录
    # seqs, alphas, scores, soup, element_number, rev_word_map = run_model(model_path, word_map_path, image_path, max_element_number = max_element_number)
    # alphas = [torch.FloatTensor(alpha) for alpha in alphas]
    # for i, seq in enumerate(seqs):
    #     visualize_att_svg(soup, element_number, args.img, seq, alphas[i], rev_word_map, output_file + str(i), args.smooth)


