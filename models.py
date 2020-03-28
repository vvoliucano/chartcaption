import torch
from torch import nn
import torchvision

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Encoder(nn.Module):
    """
    Encoder.
    """

    def __init__(self, encoded_image_size=14):
        super(Encoder, self).__init__()
        self.enc_image_size = encoded_image_size

        resnet = torchvision.models.resnet101(pretrained=False)  # pretrained ImageNet ResNet-101

        # Remove linear and pool layers (since we're not doing classification)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)

        # Resize image to fixed size to allow input images of variable size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))

        self.fine_tune()

    def forward(self, images):
        """
        Forward propagation.

        :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
        :return: encoded images
        """
        out = self.resnet(images)  # (batch_size, 2048, image_size/32, image_size/32)
        out = self.adaptive_pool(out)  # (batch_size, 2048, encoded_image_size, encoded_image_size)
        out = out.permute(0, 2, 3, 1)  # (batch_size, encoded_image_size, encoded_image_size, 2048)
        # print("encoder out shape", out.shape)
        return out

    def fine_tune(self, fine_tune=True):
        """
        Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.

        :param fine_tune: Allow?
        """
        for p in self.resnet.parameters():
            p.requires_grad = False
        # If fine-tuning, only fine-tune convolutional blocks 2 through 4
        for c in list(self.resnet.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune




class Attention(nn.Module):
    """
    Attention Network.
    """

    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        """
        :param encoder_dim: feature size of encoded images
        :param decoder_dim: size of decoder's RNN
        :param attention_dim: size of the attention network
        """
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)  # linear layer to transform encoded image
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)  # linear layer to transform decoder's output
        self.full_att = nn.Linear(attention_dim, 1)  # linear layer to calculate values to be softmax-ed
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, encoder_out, decoder_hidden):
        """
        Forward propagation.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)
        :return: attention weighted encoding, weights
        """
        att1 = self.encoder_att(encoder_out)  # (batch_size, num_pixels, attention_dim)
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)  # (batch_size, num_pixels)
        alpha = self.softmax(att)  # (batch_size, num_pixels)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim)

        return attention_weighted_encoding, alpha


class DecoderWithAttention(nn.Module):
    """
    Decoder.
    """
    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size, encoder_dim=2048, dropout=0.5):
        """
        :param attention_dim: size of attention network
        :param embed_dim: embedding size
        :param decoder_dim: size of decoder's RNN
        :param vocab_size: size of vocabulary
        :param encoder_dim: feature size of encoded images
        :param dropout: dropout
        """
        super(DecoderWithAttention, self).__init__()

        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout

        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)  # attention network

        self.embedding = nn.Embedding(vocab_size, embed_dim)  # embedding layer
        self.dropout = nn.Dropout(p=self.dropout)
        self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)  # decoding LSTMCell
        self.init_h = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial hidden state of LSTMCell
        self.init_c = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial cell state of LSTMCell
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)  # linear layer to create a sigmoid-activated gate
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(decoder_dim, vocab_size)  # linear layer to find scores over vocabulary
        self.init_weights()  # initialize some layers with the uniform distribution

    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence.
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def load_pretrained_embeddings(self, embeddings):
        """
        Loads embedding layer with pre-trained embeddings.

        :param embeddings: pre-trained embeddings
        """
        self.embedding.weight = nn.Parameter(embeddings)

    def fine_tune_embeddings(self, fine_tune=True):
        """
        Allow fine-tuning of embedding layer? (Only makes sense to not-allow if using pre-trained embeddings).

        :param fine_tune: Allow?
        """
        for p in self.embedding.parameters():
            p.requires_grad = fine_tune

    def init_hidden_state(self, encoder_out):
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :return: hidden state, cell state
        """
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out)
        return h, c

    def forward(self, encoder_out, encoded_captions, caption_lengths):
        """
        Forward propagation.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, enc_image_size, enc_image_size, encoder_dim)
        :param encoded_captions: encoded captions, a tensor of dimension (batch_size, max_caption_length)
        :param caption_lengths: caption lengths, a tensor of dimension (batch_size, 1)
        :return: scores for vocabulary, sorted encoded captions, decode lengths, weights, sort indices
        """

        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size

        # Flatten image
        # print("encoder_dim", encoder_dim)
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)
        num_pixels = encoder_out.size(1)

        # Sort input data by decreasing lengths; why? apparent below
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]

        # Embedding
        embeddings = self.embedding(encoded_captions)  # (batch_size, max_caption_length, embed_dim)

        # Initialize LSTM state
        h, c = self.init_hidden_state(encoder_out)  # (batch_size, decoder_dim)

        # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
        # So, decoding lengths are actual lengths - 1
        decode_lengths = (caption_lengths - 1).tolist()

        # Create tensors to hold word predicion scores and alphas
        # print("max(decode_lengths)", max(decode_lengths))
        # print("vocab_size", vocab_size)
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(device)
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(device)

        # At each time-step, decode by
        # attention-weighing the encoder's output based on the decoder's previous hidden state output
        # then generate a new word in the decoder with the previous word and the attention weighted encoding
        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t],
                                                                h[:batch_size_t])
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))  # gating scalar, (batch_size_t, encoder_dim)
            attention_weighted_encoding = gate * attention_weighted_encoding
            h, c = self.decode_step(
                torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1),
                (h[:batch_size_t], c[:batch_size_t]))  # (batch_size_t, decoder_dim)
            preds = self.fc(self.dropout(h))  # (batch_size_t, vocab_size)
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha

        return predictions, encoded_captions, decode_lengths, alphas, sort_ind


class SvgCompEncoder(nn.Module):
    """
    Encoder.
    this_type + position (size, position) + color + opacity
    """
    def __init__(self, svg_channel=20, svg_element_number = 10, use_bias=False, input_nc = [3, 2, 4, 3, 1], output_nc = [5, 5, 5, 5, 5]):
        super(SvgCompEncoder, self).__init__()
        print(input_nc)
        svg_channel = sum(input_nc)
        for i, input_channel in enumerate(input_nc):
            model = [nn.Conv1d(input_nc[i], output_nc[i], kernel_size = 1),
                    nn.BatchNorm1d(output_nc[i])
                    ]
            exec("self.embedding{} = nn.Sequential(*model)".format(i))
        total_nc = sum(output_nc)
        self.input_parts = len(input_nc)
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.total_nc = total_nc
        self.linear = nn.Conv1d(total_nc, total_nc, kernel_size = 1)
        # type_encoder = [nn.Conv1d(3, type_out_nc, kernel_size = 1),
                            # nn.BatchNorm1d(type_out_nc)]
        # 这不是一个简单的尝试

        modules = [nn.ReflectionPad1d(3),
                nn.Conv1d(total_nc, 256, kernel_size=7, padding=0, bias=use_bias),
                nn.BatchNorm1d(256),
                nn.ReLU(True),
                nn.ReflectionPad1d(3),
                nn.Conv1d(256, 2048, kernel_size=7, padding=0, bias=use_bias),
                nn.Tanh()
                ]

        self.model = nn.Sequential(*modules)

        # resnet = torchvision.models.resnet101(pretrained=False)  # pretrained ImageNet ResNet-101

        # # Remove linear and pool layers (since we're not doing classification)
        # modules = list(resnet.children())[:-2]
        # self.resnet = nn.Sequential(*modules)
        # self.

        # # Resize image to fixed size to allow input images of variable size
        # self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))

        # self.fine_tune()

    def forward(self, input):
        """
        Forward propagation.

        :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
        :return: encoded images
        """

        input_array = torch.split(input, self.input_nc, dim = 1) #  这边根据input——nc的数组对输入的数据进行分裂
        
        # print(input_)

        assert(len(input_array) == len(self.input_nc))
        embedding_results = []
        for i in range(self.input_parts):
            exec("embedding_results.append(self.embedding{}(input_array[i]))".format(i)) # 对每个模块输出一部分
        total = torch.cat(embedding_results, 1)
        # print("total shape: ", total.shape)
        # assert(self.total_nc == total.data.shape[1])
        # total = self.debug_embedding(input)
        out = self.linear(total)
        out = self.model(out)
     
        # print("images.shape", images.shape)
        # out = self.model(images)  # (batch_size, 2048, image_size/32, image_size/32)
        # print("out shape", out.shape)
        # out = self.adaptive_pool(out)  # (batch_size, 2048, encoded_image_size, encoded_image_size)
        out = out.permute(0, 2, 1)  # (batch_size, encoded_image_size, encoded_image_size, 2048)
        # print("encoder out shape", out.shape)
        return out

    def fine_tune(self, fine_tune=True):
        """
        Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.

        :param fine_tune: Allow?
        """
        for p in self.resnet.parameters():
            p.requires_grad = False
        # If fine-tuning, only fine-tune convolutional blocks 2 through 4
        for c in list(self.resnet.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune


class SvgEncoder(nn.Module):
    """
    Encoder.
    """
    def __init__(self, svg_channel=20, svg_element_number = 10, use_bias=False):
        super(SvgEncoder, self).__init__()

        # 这是一个简单的尝试

        modules = [nn.ReflectionPad1d(3),
                nn.Conv1d(svg_channel, 256, kernel_size=7, padding=0, bias=use_bias),
                nn.BatchNorm1d(256),
                nn.ReLU(True),
                nn.ReflectionPad1d(3),
                nn.Conv1d(256, 2048, kernel_size=7, padding=0, bias=use_bias),
                nn.Tanh()
                ]

        self.model = nn.Sequential(*modules)


        # resnet = torchvision.models.resnet101(pretrained=False)  # pretrained ImageNet ResNet-101

        # # Remove linear and pool layers (since we're not doing classification)
        # modules = list(resnet.children())[:-2]
        # self.resnet = nn.Sequential(*modules)
        # self.

        # # Resize image to fixed size to allow input images of variable size
        # self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))

        # self.fine_tune()

    def forward(self, images):
        """
        Forward propagation.

        :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
        :return: encoded images
        """

        # print("images.shape", images.shape)
        out = self.model(images)  # (batch_size, 2048, image_size/32, image_size/32)
        # print("out shape", out.shape)
        # out = self.adaptive_pool(out)  # (batch_size, 2048, encoded_image_size, encoded_image_size)
        out = out.permute(0, 2, 1)  # (batch_size, encoded_image_size, encoded_image_size, 2048)
        # print("encoder out shape", out.shape)
        return out

    def fine_tune(self, fine_tune=True):
        """
        Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.

        :param fine_tune: Allow?
        """
        for p in self.resnet.parameters():
            p.requires_grad = False
        # If fine-tuning, only fine-tune convolutional blocks 2 through 4
        for c in list(self.resnet.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune



class SvgResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm1d, use_dropout=False, n_blocks=6, padding_type='reflect', svgresnet_output=3, need_mid=False, resnet_layer = 12):
        super(SvgResnetGenerator, self).__init__()
        assert(len(input_nc) == len(output_nc))
        self.input_parts = len(input_nc)
        # exec("self.debug_embedding = nn.Conv1d(sum(input_nc), sum(output_nc), kernel_size = 1)")
        self.embedding = []
        self.input_nc = input_nc
        self.output_nc = output_nc
        for i in range(self.input_parts):
            model = [nn.Conv1d(input_nc[i], output_nc[i], kernel_size = 1),
                    nn.BatchNorm1d(output_nc[i])]
            exec("self.embedding{} = nn.Sequential(*model)".format(i))

        total_nc = sum(output_nc)
        self.total_nc = total_nc
        self.linear = nn.Conv1d(total_nc, total_nc, kernel_size = 1)
        self.resnet1d = Resnet1dGenerator(total_nc, svgresnet_output, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=resnet_layer) # 3 代表着生成一个句子。第一个维度表示是否关键部分、第二个维度表示是否被比较部分、第三个维度表示不重要。
            # type_encoder = [nn.Conv1d(3, type_out_nc, kernel_size = 1),
                            # nn.BatchNorm1d(type_out_nc)]
        self.need_mid = need_mid




    def forward(self, input):
        # print(input.data.shape)
        input_array = torch.split(input, self.input_nc, dim = 1) #  这边根据input——nc的数组对输入的数据进行分裂
        assert(len(input_array) == len(self.input_nc))
        embedding_results = []
        for i in range(self.input_parts):
            exec("embedding_results.append(self.embedding{}(input_array[i]))".format(i)) # 对每个模块输出一部分
        total = torch.cat(embedding_results, 1)
        # assert(self.total_nc == total.data.shape[1])
        # total = self.debug_embedding(input)
        mid = self.linear(total)
        output = self.resnet1d(mid)

        # print(output.data.shape)
        # print(self.need_mid)
        if self.need_mid:
            return output, mid
        return output

class Resnet1dGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm1d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        assert(n_blocks >= 0)
        super(Resnet1dGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm1d
        else:
            use_bias = norm_layer == nn.InstanceNorm1d

        model = [nn.ReflectionPad1d(3),
                 nn.Conv1d(input_nc, ngf, kernel_size=7, padding=0,
                           bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 0
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv1d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [Resnet1dBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose1d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad1d(3)]
        model += [nn.Conv1d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        output = self.model(input)
        return output


# Define a resnet block
class Resnet1dBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(Resnet1dBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad1d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad1d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv1d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad1d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad1d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv1d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


