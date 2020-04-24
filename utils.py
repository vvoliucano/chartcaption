import os
import numpy as np
import h5py
import json
import torch
from skimage.io import imread
from skimage.transform import resize as imresize
from tqdm import tqdm
from collections import Counter
from random import seed, choice, sample
from extract_svg import parse_svg_string
# fr import svg_read

svg_channel = 13 # 
# svg_number = 40


def make_sure_dir(dirname):
    if os.path.exists(dirname):
        os.system(f"rm -r {dirname}")
        # shutil.rmtree(dirname)
    os.mkdir(dirname)



def svg_read(filename = "", need_soup = False, need_text = False, svg_number = 40, svg_string = "", use_svg_string = False ):
    # a = []
    # img = np.random.random_sample((20, 10))

    if use_svg_string:
        svg_string = svg_string
    else:
        f = open(filename)
        # print("open file", filename)
        svg_string = f.read()
#  

    # print(svg_string)
    if need_text:
        a_numpy, id_array, soup, text = parse_svg_string(svg_string, min_element_num=svg_number, simple = True, need_text = need_text)
    else:
        a_numpy, id_array, soup = parse_svg_string(svg_string, min_element_num=svg_number, simple = True, need_text = need_text)
    # print(a_numpy[0])

    # 需要考虑用 sentence 中的句子替换一下
    # print(text)
    img = np.transpose(a_numpy)
    img = img - 0.5
    # 查看图像的大小
    # print("img size", img.shape)
    # for i in img:
    #     print(i[0])
    # for i in img[0]:
    #     print(i)

    if need_soup:
        if need_text:
            return img, soup, text
        else:
            return img, soup
    if need_text:
        return img, text
    return img

def create_input_files_replace_token(dataset, karpathy_json_path, image_folder, captions_per_image, min_word_freq, output_folder,
                       max_len=100, image_type = "pixel", need_text = False, max_element_number = 40):
    """
    Creates input files for training, validation, and test data.

    :param dataset: name of dataset, one of 'coco', 'flickr8k', 'flickr30k'
    :param karpathy_json_path: path of Karpathy JSON file with splits and captions
    :param image_folder: folder with downloaded images
    :param captions_per_image: number of captions to sample per image
    :param min_word_freq: words occuring less frequently than this threshold are binned as <unk>s
    :param output_folder: folder to save files
    :param max_len: don't sample captions longer than this length
    """
    print("min_freq", min_word_freq)

    make_sure_dir(output_folder)

    assert dataset in {'coco', 'flickr8k', 'flickr30k', "chart"}

    # Read Karpathy JSON
    with open(karpathy_json_path, 'r') as j:
        data = json.load(j)

    # Read image paths and captions for each image
    train_image_paths = []
    train_image_captions = []
    val_image_paths = []
    val_image_captions = []
    test_image_paths = []
    test_image_captions = []
    word_freq = Counter()


    # 此处需要进行修改，不能直接使用tokens，而对每个图像应该维护一个replace 列表。
    # 1、首先从图中构建一个replace 列表
    # 2、查找重复出现的可以替换的部分
    # 3、

    for img in data['images']:
        captions = []
        # 直接从此处获取 tokens
        for c in img['sentences']:
            # Update word frequency
            word_freq.update(c['tokens'])
            if len(c['tokens']) <= max_len:
                captions.append(c['raw'])

        if len(captions) == 0:
            continue

        # print(word_freq)

        path = os.path.join(image_folder, img['filepath'], img['filename']) if dataset == 'coco' else os.path.join(
            image_folder, img['filename'])

        if img['split'] in {'train', 'restval'}:
            train_image_paths.append(path)
            train_image_captions.append(captions)
        elif img['split'] in {'val'}:
            val_image_paths.append(path)
            val_image_captions.append(captions)
        elif img['split'] in {'test'}:
            test_image_paths.append(path)
            test_image_captions.append(captions)

    # Sanity check
    assert len(train_image_paths) == len(train_image_captions)
    assert len(val_image_paths) == len(val_image_captions)
    assert len(test_image_paths) == len(test_image_captions)

    # Create word map
    words = [w for w in word_freq.keys() if word_freq[w] > min_word_freq]
    word_map = {k: v + 1 for v, k in enumerate(words)}
    # 不初始化

    # word_map = {}
    word_map['<unk>'] = len(word_map) + 1
    word_map['<start>'] = len(word_map) + 1
    word_map['<end>'] = len(word_map) + 1
    # word_map['<#1>'] = len(word_map) + 1
    word_map['<pad>'] = 0

    # 这样初始化就什么都没了

    # Create a base/root name for all output files
    base_filename = dataset + '_' + str(captions_per_image) + '_cap_' + str(min_word_freq) + '_min_wf'

    # Save word map to a JSON

    # Sample captions for each image, save images to HDF5 file, and captions and their lengths to JSON files
    seed(123)
    for impaths, imcaps, split in [(train_image_paths, train_image_captions, 'TRAIN'),
                                   (val_image_paths, val_image_captions, 'VAL'),
                                   (test_image_paths, test_image_captions, 'TEST')]:
        print(os.path.join(output_folder, split + '_IMAGES_' + base_filename + '.hdf5'))
        print("image number", len(impaths))
        with h5py.File(os.path.join(output_folder, split + '_IMAGES_' + base_filename + '.hdf5'), 'a') as h:
            # Make a note of the number of captions we are sampling per image
            h.attrs['captions_per_image'] = captions_per_image

            # Create dataset inside HDF5 file to store images
            if image_type == "pixel":
                images = h.create_dataset('images', (len(impaths), 3, 256, 256), dtype='uint8')
            else:
                images = h.create_dataset('images', (len(impaths), svg_channel, max_element_number), dtype='float32')

            print("\nReading %s images and captions, storing to file...\n" % split)

            enc_captions = []
            caplens = []
            image_text = []

            for i, path in enumerate(tqdm(impaths)):
                # Sample captions
                if len(imcaps[i]) < captions_per_image:
                    captions = imcaps[i] + [choice(imcaps[i]) for _ in range(captions_per_image - len(imcaps[i]))]
                else:
                    captions = sample(imcaps[i], k=captions_per_image)
                # Sanity check
                assert len(captions) == captions_per_image
                img, img_text = svg_read(impaths[i], need_text = need_text, svg_number = max_element_number)
                
                img_text = [word.lower() for word in img_text]

                # print("Image from text", img_text)

                # print("original caption", captions)

                replace_token = {}

                for word in img_text:
                    # if word != '' and word not in word_map:
                    #     word_map[word] = len(word_map) 
                    if word != '' and word != "<pad>":

                        current_token = "<#" + str(len(replace_token)) + ">"
                        replace_token[word] = current_token
                        if current_token not in word_map:
                            word_map[current_token] = len(word_map)
                        
                for j, caption in enumerate(captions):
                    caption = " " + caption + " "
                    caption = caption.lower()
                    caption = caption.replace(",", " ,").replace('.', " .").replace(';', " ;").replace("  ", " ")
                    for word in replace_token:
                        caption = caption.replace(" " + word + " ", " " + replace_token[word] + " ")
                    caption = caption.strip()
                    captions[j] = caption.split(" ")

                # print("replace caption", captions)
                # print("replaced token", replace_token)

                encoded_image_text = [word_map.get(replace_token.get(word, ""), 0) for word in img_text]
                image_text.append(encoded_image_text)
                # 添加图形中出现的词汇
                

                # Save image to HDF5 file
                images[i] = img
                # print(img)
                # print(img)

                if i == 0:
                    print("see the situation of the first file")
                    # print(img)
                    # print(img[0])
                    print(encoded_image_text)

                for j, c in enumerate(captions):
                    # Encode captions 其实完全没有必要

                    enc_c = [word_map['<start>']] + [word_map.get(word, word_map['<unk>']) for word in c] + [
                        word_map['<end>']] + [word_map['<pad>']] * (max_len - len(c))
                    
                    # print("enc_c", enc_c)

                    # Find caption lengths
                    c_len = len(c) + 2

                    enc_captions.append(enc_c)
                    caplens.append(c_len)


            # print("word_map", word_map)
            # Sanity check
            assert images.shape[0] * captions_per_image == len(enc_captions) == len(caplens)

            # Save encoded captions and their lengths to JSON files
            with open(os.path.join(output_folder, split + '_CAPTIONS_' + base_filename + '.json'), 'w') as j:
                json.dump(enc_captions, j)

            with open(os.path.join(output_folder, split + '_CAPLENS_' + base_filename + '.json'), 'w') as j:
                json.dump(caplens, j)

            with open(os.path.join(output_folder, split + '_IMAGE_TEXT_' + base_filename + '.json'), 'w') as j:
                json.dump(image_text, j)

    with open(os.path.join(output_folder, 'WORDMAP_' + base_filename + '.json'), 'w') as j:
        json.dump(word_map, j)
    print("word_map", word_map)



def create_input_files(dataset, karpathy_json_path, image_folder, captions_per_image, min_word_freq, output_folder,
                       max_len=100, image_type = "pixel", need_text = False, max_element_number = 40):
    """
    Creates input files for training, validation, and test data.

    :param dataset: name of dataset, one of 'coco', 'flickr8k', 'flickr30k'
    :param karpathy_json_path: path of Karpathy JSON file with splits and captions
    :param image_folder: folder with downloaded images
    :param captions_per_image: number of captions to sample per image
    :param min_word_freq: words occuring less frequently than this threshold are binned as <unk>s
    :param output_folder: folder to save files
    :param max_len: don't sample captions longer than this length
    """
    print("min_freq", min_word_freq)

    make_sure_dir(output_folder)

    assert dataset in {'coco', 'flickr8k', 'flickr30k', "chart"}

    # Read Karpathy JSON
    with open(karpathy_json_path, 'r') as j:
        data = json.load(j)

    # Read image paths and captions for each image
    train_image_paths = []
    train_image_captions = []
    val_image_paths = []
    val_image_captions = []
    test_image_paths = []
    test_image_captions = []
    word_freq = Counter()


    # 此处需要进行修改，不能直接使用tokens，而对每个图像应该维护一个replace 列表。
    # 1、首先从图中构建一个replace 列表
    # 2、查找重复出现的可以替换的部分
    # 3、

    for img in data['images']:
        captions = []
        for c in img['sentences']:
            # Update word frequency
            word_freq.update(c['tokens'])
            if len(c['tokens']) <= max_len:
                captions.append(c['tokens'])

        if len(captions) == 0:
            continue

        # print(word_freq)

        path = os.path.join(image_folder, img['filepath'], img['filename']) if dataset == 'coco' else os.path.join(
            image_folder, img['filename'])

        if img['split'] in {'train', 'restval'}:
            train_image_paths.append(path)
            train_image_captions.append(captions)
        elif img['split'] in {'val'}:
            val_image_paths.append(path)
            val_image_captions.append(captions)
        elif img['split'] in {'test'}:
            test_image_paths.append(path)
            test_image_captions.append(captions)

    # Sanity check
    assert len(train_image_paths) == len(train_image_captions)
    assert len(val_image_paths) == len(val_image_captions)
    assert len(test_image_paths) == len(test_image_captions)

    # Create word map
    words = [w for w in word_freq.keys() if word_freq[w] > min_word_freq]
    word_map = {k: v + 1 for v, k in enumerate(words)}
    word_map['<unk>'] = len(word_map) + 1
    word_map['<start>'] = len(word_map) + 1
    word_map['<end>'] = len(word_map) + 1
    # word_map['<#1>'] = len(word_map) + 1
    word_map['<pad>'] = 0

    # Create a base/root name for all output files
    base_filename = dataset + '_' + str(captions_per_image) + '_cap_' + str(min_word_freq) + '_min_wf'

    # Save word map to a JSON

    # Sample captions for each image, save images to HDF5 file, and captions and their lengths to JSON files
    seed(123)
    for impaths, imcaps, split in [(train_image_paths, train_image_captions, 'TRAIN'),
                                   (val_image_paths, val_image_captions, 'VAL'),
                                   (test_image_paths, test_image_captions, 'TEST')]:
        print(os.path.join(output_folder, split + '_IMAGES_' + base_filename + '.hdf5'))
        print("image number", len(impaths))
        with h5py.File(os.path.join(output_folder, split + '_IMAGES_' + base_filename + '.hdf5'), 'a') as h:
            # Make a note of the number of captions we are sampling per image
            h.attrs['captions_per_image'] = captions_per_image

            # Create dataset inside HDF5 file to store images
            if image_type == "pixel":
                images = h.create_dataset('images', (len(impaths), 3, 256, 256), dtype='uint8')
            else:
                images = h.create_dataset('images', (len(impaths), svg_channel, max_element_number), dtype='float32')

            print("\nReading %s images and captions, storing to file...\n" % split)

            enc_captions = []
            caplens = []
            image_text = []

            for i, path in enumerate(tqdm(impaths)):

                # Sample captions
                if len(imcaps[i]) < captions_per_image:
                    captions = imcaps[i] + [choice(imcaps[i]) for _ in range(captions_per_image - len(imcaps[i]))]
                else:
                    captions = sample(imcaps[i], k=captions_per_image)

                # Sanity check
                assert len(captions) == captions_per_image

                # Read images
                if image_type == "pixel":
                    img = imread(impaths[i])
                    if len(img.shape) == 2:
                        img = img[:, :, np.newaxis]
                        img = np.concatenate([img, img, img], axis=2)
                    img = imresize(img, (256, 256))
                    img = img.transpose(2, 0, 1)
                    assert img.shape == (3, 256, 256)
                    assert np.max(img) <= 255
                else:
                    if need_text:
                        img, img_text = svg_read(impaths[i], need_text = need_text, svg_number = max_element_number)
                        # print("img_text: ", img_text)
                    else:
                        img = svg_read(impaths[i], need_text = need_text)


                if i == 0:
                    print("see the situation of the first file")
                    print(img)
                    print(img[0])
                    print(image_text)

                if need_text:
                    img_text = [word.lower() for word in img_text]
                    for word in img_text:
                        if word != '' and word not in word_map:
                            word_map[word] = len(word_map)    
                    encoded_image_text = [word_map.get(word, 0) for word in img_text]
                    image_text.append(encoded_image_text)
                                            # 添加图形中出现的词汇
                



                # Save image to HDF5 file
                images[i] = img
                # print(img)
                # print(img)

                for j, c in enumerate(captions):
                    # Encode captions
                    enc_c = [word_map['<start>']] + [word_map.get(word, word_map['<unk>']) for word in c] + [
                        word_map['<end>']] + [word_map['<pad>']] * (max_len - len(c))
                    
                    # print("enc_c", enc_c)

                    # Find caption lengths
                    c_len = len(c) + 2

                    enc_captions.append(enc_c)
                    caplens.append(c_len)


            # print("word_map", word_map)
            # Sanity check
            assert images.shape[0] * captions_per_image == len(enc_captions) == len(caplens)

            # Save encoded captions and their lengths to JSON files
            with open(os.path.join(output_folder, split + '_CAPTIONS_' + base_filename + '.json'), 'w') as j:
                json.dump(enc_captions, j)

            with open(os.path.join(output_folder, split + '_CAPLENS_' + base_filename + '.json'), 'w') as j:
                json.dump(caplens, j)

            with open(os.path.join(output_folder, split + '_IMAGE_TEXT_' + base_filename + '.json'), 'w') as j:
                json.dump(image_text, j)

    with open(os.path.join(output_folder, 'WORDMAP_' + base_filename + '.json'), 'w') as j:
        json.dump(word_map, j)



def init_embedding(embeddings):
    """
    Fills embedding tensor with values from the uniform distribution.

    :param embeddings: embedding tensor
    """
    bias = np.sqrt(3.0 / embeddings.size(1))
    torch.nn.init.uniform_(embeddings, -bias, bias)


def load_embeddings(emb_file, word_map):
    """
    Creates an embedding tensor for the specified word map, for loading into the model.

    :param emb_file: file containing embeddings (stored in GloVe format)
    :param word_map: word map
    :return: embeddings in the same order as the words in the word map, dimension of embeddings
    """

    # Find embedding dimension
    with open(emb_file, 'r') as f:
        emb_dim = len(f.readline().split(' ')) - 1

    vocab = set(word_map.keys())

    # Create tensor to hold embeddings, initialize
    embeddings = torch.FloatTensor(len(vocab), emb_dim)
    init_embedding(embeddings)

    # Read embedding file
    print("\nLoading embeddings...")
    for line in open(emb_file, 'r'):
        line = line.split(' ')

        emb_word = line[0]
        embedding = list(map(lambda t: float(t), filter(lambda n: n and not n.isspace(), line[1:])))

        # Ignore word if not in train_vocab
        if emb_word not in vocab:
            continue

        embeddings[word_map[emb_word]] = torch.FloatTensor(embedding)

    return embeddings, emb_dim


def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.

    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def save_checkpoint(data_name, epoch, epochs_since_improvement, encoder, decoder, encoder_optimizer, decoder_optimizer,
                    bleu4, is_best, checkpoint_path = "checkpoint/"):
    """
    Saves model checkpoint.

    :param data_name: base name of processed dataset
    :param epoch: epoch number
    :param epochs_since_improvement: number of epochs since last improvement in BLEU-4 score
    :param encoder: encoder model
    :param decoder: decoder model
    :param encoder_optimizer: optimizer to update encoder's weights, if fine-tuning
    :param decoder_optimizer: optimizer to update decoder's weights
    :param bleu4: validation BLEU-4 score for this epoch
    :param is_best: is this checkpoint the best so far?
    """

    # print("this epoch is best: ", is_best)
    state = {'epoch': epoch,
             'epochs_since_improvement': epochs_since_improvement,
             'bleu-4': bleu4,
             'encoder': encoder,
             'decoder': decoder,
             'encoder_optimizer': encoder_optimizer,
             'decoder_optimizer': decoder_optimizer}


        
    filename = f'{checkpoint_path}/epoch_{epoch}_bleu_{bleu4}.pth.tar'

    torch.save(state, filename)
    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    if is_best:
        filename = f'{checkpoint_path}/Best.pth.tar'
        torch.save(state, filename)


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, shrink_factor):
    """
    Shrinks learning rate by a specified factor.

    :param optimizer: optimizer whose learning rate must be shrunk.
    :param shrink_factor: factor in interval (0, 1) to multiply learning rate with.
    """

    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))


def accuracy(scores, targets, k):
    """
    Computes top-k accuracy, from predicted and true labels.

    :param scores: scores from the model
    :param targets: true labels
    :param k: k in top-k accuracy
    :return: top-k accuracy
    """

    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / batch_size)
