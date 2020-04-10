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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def get_svg_image_from_file(image_path, need_text):
    # img = np.random.random_sample((20, 10))
    print("need_text or not ", need_text)
    img, soup, image_text = svg_read(image_path, need_soup = True, need_text = need_text)
    # elements = soup.findAll(attrs = {"caption_sha", "5"})
    # elements = soup.findAll(attrs = {"caption_id":  "2"})
    elements = soup.findAll(attrs = {"caption_sha":  "5"})
    # print("elements", elements)
    element_number = len(elements)
    # element_number = len()
    # print("element_number", element_number)
    # print(soup.findAll(attrs = {"caption_id":  "2"}))
    # print(soup)
    # with open("1.html", "w") as file:
    #     file.write(str(soup))
    img = torch.FloatTensor(img).to(device)
    image_text = torch.FloatTensor(image_text).to(device)
    return img, soup, element_number, image_text

def caption_image_beam_search(encoder, decoder, image_path, word_map, beam_size=3, image_type="pixel", need_text = False):
    """
    Reads an image and captions it with beam search.

    :param encoder: encoder model
    :param decoder: decoder model
    :param image_path: path to image
    :param word_map: word map
    :param beam_size: number of sequences to consider at each decode-step
    :return: caption, weights for visualization
    """

    k = beam_size
    vocab_size = len(word_map)

    
    if need_text:
        image, soup, element_number, image_text = get_svg_image_from_file(image_path, need_text = need_text)
        image = image.unsqueeze(0)  # (1, 3, 256, 256)
        image_text = image_text.unsqueeze(0) # 添加一个
        encoder_out = encoder(image)  # (1, enc_image_size, encoder_dim)
        enc_image_size = encoder_out.size(1)
        print("encoder_out.shape", encoder_out.shape)
        encoder_dim = encoder_out.size(-1) # 

    else:  
        image, soup, element_number = get_svg_image_from_file(image_path, need_text = need_text)
        image = image.unsqueeze(0)  # (1, 3, 256, 256)
        encoder_out = encoder(image)  # (1, enc_image_size, enc_image_size, encoder_dim)
        enc_image_size = encoder_out.size(1)
        print("encoder_out.shape", encoder_out.shape)
        encoder_dim = encoder_out.size(-1) # 

    # Encode
    image = image.unsqueeze(0)  # (1, 3, 256, 256)
    encoder_out = encoder(image)  # (1, enc_image_size, enc_image_size, encoder_dim)
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

    # Tensor to store top k sequences' alphas; now they're just 1s
    if image_type == "pixel":
        seqs_alpha = torch.ones(k, 1, enc_image_size, enc_image_size).to(device)  # (k, 1, enc_image_size, enc_image_size)
    else:
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

        if image_type == "pixel":
            alpha = alpha.view(-1, enc_image_size, enc_image_size)  # (s, enc_image_size, enc_image_size)
        else:
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

    print("complete_seqs_scores", complete_seqs_scores)
    sorted_seqs_scores = sorted(complete_seqs_scores, reverse = True)
    # print("sorted_seqs_scores", sorted_seqs_scores)
    seqs = [complete_seqs[complete_seqs_scores.index(seq_score)] for seq_score in sorted_seqs_scores]
    alphas_of_seqs = [complete_seqs_alpha[complete_seqs_scores.index(seq_score)] for seq_score in sorted_seqs_scores]

    i = complete_seqs_scores.index(max(complete_seqs_scores))
    print("complete_seqs", complete_seqs)
    print("complete_seqs_scores", complete_seqs_scores)
    print("sorted seqs", seqs)
    seq = complete_seqs[i]
    alphas = complete_seqs_alpha[i]
    if image_type == "pixel":   
        return seq, alphas
    else:
        return seqs, alphas_of_seqs, soup, element_number

    


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
        current_alpha_max = np.max(current_alpha_numpy[0:element_number])
        
        for i in range(element_number):
            current_element = soup.findAll(attrs = {"caption_id":  str(i)})[0]
            current_element["opacity"] = current_alpha_numpy[i] / current_alpha_max
        soup_string = str(soup.select("svg")[0])
        # print(soup_string)
        # print(f"第{t}个soup", soup_string)
        new_soup = bs4.BeautifulSoup(soup_string, "html5lib")
        soup_list.append(new_soup)




        # with open(f"data/{t}_{words[t]}.html", "w") as file:
        #     file.write(str(soup))

        # for i in range(element_number):
            

        # print(current_alpha.shape)
      


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




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Show, Attend, and Tell - Tutorial - Generate Caption')

    parser.add_argument('--img', '-i', help='path to image')
    parser.add_argument('--model', '-m', help='path to model')
    parser.add_argument('--word_map', '-wm', help='path to word map JSON')
    parser.add_argument('--beam_size', '-b', default=5, type=int, help='beam size for beam search')
    parser.add_argument('--dont_smooth', dest='smooth', action='store_false', help='do not smooth alpha overlay')
    parser.add_argument('--image_type', type=str, default = 'pixel', help='image type as input')
    parser.add_argument('--need_text', action='store_true', help="decide whether need text")


# python caption.py --img /home/can.liu/caption/data/coco_2014/val2014/COCO_val2014_000000204853.jpg --model /home/can.liu/caption/chartcaption/BEST_checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar --word_map /home/can.liu/caption/data/karpathy_output/WORDMAP_coco_5_cap_per_img_5_min_word_freq.json

    args = parser.parse_args()

    # output_file = "data/" + args.img.split("/")[-1][0:-4] + ".jpg"
    # print(output_file)

    # Load model
    checkpoint = torch.load(args.model, map_location=str(device))
    decoder = checkpoint['decoder']
    decoder = decoder.to(device)
    decoder.eval()
    encoder = checkpoint['encoder']
    encoder = encoder.to(device)
    encoder.eval()

    # Load word map (word2ix)
    with open(args.word_map, 'r') as j:
        word_map = json.load(j)
    rev_word_map = {v: k for k, v in word_map.items()}  # ix2word

    # Encode, decode with attention and beam search
    if args.image_type == "pixel":
        seq, alphas = caption_image_beam_search(encoder, decoder, args.img, word_map, args.beam_size, args.image_type, need_text = args.need_text)
        alphas = torch.FloatTensor(alphas)
        visualize_att(args.img, seq, alphas, rev_word_map, args.smooth)

    else:
        seqs, alphas, soup, element_number = caption_image_beam_search(encoder, decoder, args.img, word_map, args.beam_size, args.image_type, need_text = args.need_text)
        alphas = [torch.FloatTensor(alpha) for alpha in alphas]
        # alphas = torch.FloatTensor(alphas)
        for i, seq in enumerate(seqs):
            visualize_att_svg(soup, element_number, args.img, seq, alphas[i], rev_word_map, output_file + str(i), args.smooth)
