import time
import torch.backends.cudnn as cudnn
import torch
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from models import Encoder, DecoderWithAttention, SvgEncoder, SvgCompEncoder
from datasets import *
from utils import *
from nltk.translate.bleu_score import corpus_bleu
import os
import argparse

parser = argparse.ArgumentParser(description='Generate Caption for SVG')

parser.add_argument('--data_folder', type=str, default = "/home/can.liu/caption/data/karpathy_output/", help='folder with data files saved by create_input_files.py')
parser.add_argument('--data_name', type=str, default = 'coco_5_cap_per_img_5_min_word_freq', help='base name shared by data files')
parser.add_argument('--beam_size', '-b', default=5, type=int, help='beam size for beam search')
parser.add_argument('--dont_smooth', dest='smooth', action='store_false', help='do not smooth alpha overlay')
parser.add_argument('--image_type', type=str, default = 'pixel', help='image type as input')
parser.add_argument('--svg_channel', type=int, default = 75)
parser.add_argument('--input_nc', type=str, default = "", help='using svg list')
parser.add_argument('--output_nc', type=str, default = '', help='using svg output list')
parser.add_argument('--svg_element_number', type = int, default = 40)
parser.add_argument('--pretrained_model', type=str, default = "none")
parser.add_argument('--max_epoch', type = int, default = 120)
parser.add_argument("--emb_dim", type = int, default = 512)
parser.add_argument("--attention_dim", type = int, default = 512)
parser.add_argument('--decoder_dim', type = int, default = 512)
parser.add_argument('--encoder_dim', type = int, default = 2048)
parser.add_argument('--encode_word', type = str, default = "no", help = "using text as input") # In fact, this is useless
parser.add_argument('--need_text', action='store_true', help="decide whether need text")
parser.add_argument('--need_random', action='store_true', help="decide whether need text")

# [3, 2, 4, 3, 1], output_nc = [5, 5, 5, 5, 5])
# input_nc = "3,2,4,3,1"
# output_nc = "5,5,5,5,5"

args = parser.parse_args()

# os.system(f"mkdir -p checkpoint/{args.data_name}")

checkpoint_path = "checkpoint/" + args.data_name + time.strftime("-%Y-%m-%d-%H-%M", time.localtime())
os.mkdir(checkpoint_path)

args.checkpoint_path = checkpoint_path
# python train.py --data_folder /Users/tsunmac/vis/projects/autocaption/data/karpathy_output

# Data parameters
data_folder = args.data_folder # '/home/can.liu/caption/data/karpathy_output/'  # folder with data files saved by create_input_files.py
data_name = args.data_name #'coco_5_cap_per_img_5_min_word_freq'  # base name shared by data files

# Model parameters

emb_dim = args.emb_dim  # dimension of word embeddings 词汇 embed 
attention_dim = args.attention_dim  # dimension of attention linear layers
decoder_dim = args.decoder_dim  # dimension of decoder RNN
encoder_dim = args.encoder_dim # dimension of encoder dim, which is the output of the encoder of the image and the input of the decoder.

# Input encoder with words
encode_word = False
if args.encode_word == "yes":
    encode_word = True

dropout = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

# Training parameters
start_epoch = 0
epochs = args.max_epoch  # number of epochs to train for (if early stopping is not triggered)
print('max epochs', epochs)
epochs_since_improvement = 0  # keeps track of number of epochs since there's been an improvement in validation BLEU
batch_size = 32
workers = 1  # for data-loading; right now, only 1 works with h5py
encoder_lr = 1e-4  # learning rate for encoder if fine-tuning
decoder_lr = 4e-4  # learning rate for decoder
grad_clip = 5.  # clip gradients at an absolute value of
alpha_c = 1.  # regularization parameter for 'doubly stochastic attention', as in the paper
best_bleu4 = 0.  # BLEU-4 score right now
print_freq = 100  # print training/validation stats every __ batches
fine_tune_encoder = False  # fine-tune encoder?
checkpoint = args.pretrained_model  # path to checkpoint, None if none
if checkpoint == "none":
    checkpoint = None
else:
    # checkpoint = "checkpoint/" + data_name + "/" + checkpoint
    print("Load checkpoint", checkpoint)


def main():
    """
    Training and validation.
    """

    global best_bleu4, epochs_since_improvement, checkpoint, start_epoch, fine_tune_encoder, data_name, word_map

    # Read word map
    word_map_file = os.path.join(data_folder, 'WORDMAP_' + data_name + '.json')
    with open(word_map_file, 'r') as j:
        word_map = json.load(j)

    # Initialize / load checkpoint
    if checkpoint is None:
        # print("attention dim", attention_dim)
        # print("emb_dim", emb_dim)
        # print("decoder_dim", decoder_dim)
        decoder = DecoderWithAttention(attention_dim=attention_dim,
                                       embed_dim=emb_dim,
                                       decoder_dim=decoder_dim,
                                       vocab_size=len(word_map),
                                       encoder_dim = encoder_dim,
                                       dropout=dropout)
        decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                             lr=decoder_lr)

        if args.image_type == "svg":
            if args.input_nc == "":
                encoder = SvgEncoder(svg_channel = args.svg_channel, svg_element_number = args.svg_element_number)
            else:
                input_nc = [int(item) for item in args.input_nc.split(",")]
                output_nc = [int(item) for item in args.output_nc.split(",")]
                args.svg_channel = sum(input_nc)
                if args.need_text:
                    # 此处仅使用encoder - embed 的维度， 预留出embed 的dim 作为输入的自然语言的位置
                    encoder = SvgCompEncoder(svg_channel = args.svg_channel, input_nc = input_nc, output_nc = output_nc, svg_element_number = args.svg_element_number, image_encoder_num = encoder_dim - emb_dim) 
                else:
                    encoder = SvgCompEncoder(svg_channel = args.svg_channel, input_nc = input_nc, output_nc = output_nc, svg_element_number = args.svg_element_number, image_encoder_num = encoder_dim)

        else:
            encoder = Encoder()
            encoder.fine_tune(fine_tune_encoder)

        encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                             lr=encoder_lr) if fine_tune_encoder else None

    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        best_bleu4 = checkpoint['bleu-4']
        decoder = checkpoint['decoder']
        decoder_optimizer = checkpoint['decoder_optimizer']
        encoder = checkpoint['encoder']
        encoder_optimizer = checkpoint['encoder_optimizer']
        if fine_tune_encoder is True and encoder_optimizer is None:
            encoder.fine_tune(fine_tune_encoder)
            encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                                 lr=encoder_lr)

    # Move to GPU, if available

    print("device: ", device)
    decoder = decoder.to(device)
    encoder = encoder.to(device)

    # Loss function
    criterion = nn.CrossEntropyLoss().to(device)

    # Custom dataloaders
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    if args.image_type == "svg":
        my_transform = "svg"
    else:
        my_transform = transforms.Compose([normalize])

    train_loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, 'TRAIN', transform = my_transform, need_random = args.need_random),
        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, 'VAL', transform = my_transform),
        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)

    # Epochs
    for epoch in range(start_epoch, epochs):

        # Decay learning rate if there is no improvement for 8 consecutive epochs, and terminate training after 20
        # if epochs_since_improvement == 20:
        #     break
        if epochs_since_improvement > 0 and epochs_since_improvement % 8 == 0:
            adjust_learning_rate(decoder_optimizer, 0.8)
            if fine_tune_encoder:
                adjust_learning_rate(encoder_optimizer, 0.8)

        # One epoch's training
        train(train_loader=train_loader,
              encoder=encoder,
              decoder=decoder,
              criterion=criterion,
              encoder_optimizer=encoder_optimizer,
              decoder_optimizer=decoder_optimizer,
              epoch=epoch)

        # One epoch's validation
        recent_bleu4 = validate(val_loader=val_loader,
                                encoder=encoder,
                                decoder=decoder,
                                criterion=criterion)

        # Check if there was an improvement
        is_best = recent_bleu4 > best_bleu4
        # print("This checkpoint is best: ", is_best)
        best_bleu4 = max(recent_bleu4, best_bleu4)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        # Save checkpoint
        save_checkpoint(data_name, epoch, epochs_since_improvement, encoder, decoder, encoder_optimizer,
                        decoder_optimizer, recent_bleu4, is_best, checkpoint_path = checkpoint_path)


def train(train_loader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, epoch):
    """
    Performs one epoch's training.

    :param train_loader: DataLoader for training data
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    :param encoder_optimizer: optimizer to update encoder's weights (if fine-tuning)
    :param decoder_optimizer: optimizer to update decoder's weights
    :param epoch: epoch number
    """

    decoder.train()  # train mode (dropout and batchnorm is used)
    encoder.train()

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss (per word decoded)
    top5accs = AverageMeter()  # top5 accuracy

    start = time.time()

    # Batches
    for i, (imgs, caps, caplens, image_text) in enumerate(train_loader):
        data_time.update(time.time() - start)
        # print(image_text)
        # Move to GPU, if available
        imgs = imgs.to(device)
        caps = caps.to(device)
        caplens = caplens.to(device)
        image_text = image_text.to(device)

        # Forward prop.
        imgs = encoder(imgs)
        # print("encoded imgs shape: ", imgs.shape)
        
        if args.need_text:
            # print("image text shape", image_text.shape)
            encoded_image_text = decoder.embedding(image_text)
            # print("image text encoded", encoded_image_text.shape)
            imgs = torch.cat((imgs, encoded_image_text), 2)
            # print("img dimension after cating: ", imgs.shape)

        # print("imgs. shape", imgs.shape)
        scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens)

        # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
        targets = caps_sorted[:, 1:]

        # Remove timesteps that we didn't decode at, or are pads
        # pack_padded_sequence is an easy trick to do this
        scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
        targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data
        # scores = pack_padded_sequence(scores, decode_lengths, batch_

        # print("scores", scores)
        # print("targets", targets)

        # Calculate loss
        loss = criterion(scores, targets)

        # Add doubly stochastic attention regularization
        loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

        # Back prop.
        decoder_optimizer.zero_grad()
        if encoder_optimizer is not None:
            encoder_optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        if grad_clip is not None:
            clip_gradient(decoder_optimizer, grad_clip)
            if encoder_optimizer is not None:
                clip_gradient(encoder_optimizer, grad_clip)

        # Update weights
        decoder_optimizer.step()
        if encoder_optimizer is not None:
            encoder_optimizer.step()

        # Keep track of metrics
        top5 = accuracy(scores, targets, 5)
        losses.update(loss.item(), sum(decode_lengths))
        top5accs.update(top5, sum(decode_lengths))
        batch_time.update(time.time() - start)

        start = time.time()

        # Print status
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})'.format(epoch, i, len(train_loader),
                                                                          batch_time=batch_time,
                                                                          data_time=data_time, loss=losses,
                                                                          top5=top5accs))


def validate(val_loader, encoder, decoder, criterion):
    """
    Performs one epoch's validation.

    :param val_loader: DataLoader for validation data.
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    :return: BLEU-4 score
    """
    decoder.eval()  # eval mode (no dropout or batchnorm)
    if encoder is not None:
        encoder.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top5accs = AverageMeter()

    start = time.time()

    references = list()  # references (true captions) for calculating BLEU-4 score
    hypotheses = list()  # hypotheses (predictions)

    # explicitly disable gradient calculation to avoid CUDA memory error
    # solves the issue #57
    with torch.no_grad():
        # Batches
        for i, (imgs, caps, caplens, image_text, allcaps) in enumerate(val_loader):

            # Move to device, if available
            imgs = imgs.to(device)
            caps = caps.to(device)
            caplens = caplens.to(device)
            image_text = image_text.to(device)

            # Forward prop.
            if encoder is not None:
                imgs = encoder(imgs)
                if args.need_text:
                    # print("image text shape", image_text.shape)
                    encoded_image_text = decoder.embedding(image_text)
                    # print("image text encoded", encoded_image_text.shape)
                    imgs = torch.cat((imgs, encoded_image_text), 2)
                    # print("img dimension after cating: ", imgs.shape)


            scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens)

            # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
            targets = caps_sorted[:, 1:]

            # Remove timesteps that we didn't decode at, or are pads
            # pack_padded_sequence is an easy trick to do this
            scores_copy = scores.clone()
            scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
            targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data

            # Calculate loss
            loss = criterion(scores, targets)

            # Add doubly stochastic attention regularization
            loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

            # Keep track of metrics
            losses.update(loss.item(), sum(decode_lengths))
            top5 = accuracy(scores, targets, 5)
            top5accs.update(top5, sum(decode_lengths))
            batch_time.update(time.time() - start)

            start = time.time()

            if i % print_freq == 0:
                print('Validation: [{0}/{1}]\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})\t'.format(i, len(val_loader), batch_time=batch_time,
                                                                                loss=losses, top5=top5accs))

            # Store references (true captions), and hypothesis (prediction) for each image
            # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
            # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]

            # References
            allcaps = allcaps[sort_ind]  # because images were sorted in the decoder
            for j in range(allcaps.shape[0]):
                img_caps = allcaps[j].tolist()
                img_captions = list(
                    map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<pad>']}],
                        img_caps))  # remove <start> and pads
                references.append(img_captions)

            # Hypotheses
            _, preds = torch.max(scores_copy, dim=2)
            preds = preds.tolist()
            temp_preds = list()
            for j, p in enumerate(preds):
                temp_preds.append(preds[j][:decode_lengths[j]])  # remove pads
            preds = temp_preds
            hypotheses.extend(preds)

            assert len(references) == len(hypotheses)

        # Calculate BLEU-4 scores
        bleu4 = corpus_bleu(references, hypotheses)

        print(
            '\n * LOSS - {loss.avg:.3f}, TOP-5 ACCURACY - {top5.avg:.3f}, BLEU-4 - {bleu}\n'.format(
                loss=losses,
                top5=top5accs,
                bleu=bleu4))

    return bleu4


if __name__ == '__main__':
    main()
