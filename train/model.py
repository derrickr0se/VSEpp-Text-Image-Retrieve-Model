import torch
import torch.nn as nn
import torchvision.models as models
from param import args
import numpy as np
import torch.nn.functional as f
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_
from info_nce import InfoNCE
import gensim
import os

if args.gpu.lower() == 'cpu':
    DEVICE = torch.device('cpu')
elif args.gpu in ['0', '1', '2', '3']:
    DEVICE = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
else:
    raise ValueError('Invalid GPU ID')


class ImageEncoder(nn.Module):
    def __init__(self, embed_size, cnn_type, finetune=False):
        """
            ImageEncoder结构
                (1) CNN
                (2) ReLU
                (3) Linear Mapping
                (4) L2-Normalize
        """
        super(ImageEncoder, self).__init__()
        self.embed_size = embed_size
        # 对应ImageEncoder中的CNN层
        if cnn_type == 'VGG19':
            self.cnn = models.vgg19(pretrained=True).to(DEVICE)
            self.cnn.classifier = nn.Sequential(
                *list(self.cnn.classifier.children())[:-1]
            )
            """
            原始VGG19模型包括 卷积层 和 分类器部分
            首先使用self.cnn.classifier.children()，获取了原始VGG19模型 分类器部分 中的所有子层(0-6)
            然后，将其转为list，通过切片操作[:-1]将最后一层从子层列表中移除，即移除了原始分类器的最后一层Linear层，其用于输出1000个类别的预测结果
            *list将列表解包，将列表中的每个元素(0-5)作为nn.Sequential()的参数传递给函数。这种用法通常在需要将列表中的元素作为单独的参数传递给函数时使用。
            VGG19的classifier结构：
                (classifier): Sequential(
                    (0): Linear(in_features=25088, out_features=4096, bias=True)
                    (1): ReLU(inplace=True)
                    (2): Dropout(p=0.5, inplace=False)
                    (3): Linear(in_features=4096, out_features=4096, bias=True)
                    (4): ReLU(inplace=True)
                    (5): Dropout(p=0.5, inplace=False)
                    (6): Linear(in_features=4096, out_features=1000, bias=True) # 移除此层，将VGG19提取的特征映射到指定的嵌入向量大小，而不包含原始预测输出的类别信息
                )
            """
            # 在VGG19后面接一个映射层,输入为原最后一层的输入4096，输出为指定的嵌入向量大小embed_size
            self.mapping = nn.Linear(
                self.cnn.classifier[3].out_features,
                embed_size,
                bias=True
            )
            # 设置VGG19的卷积层是否参与训练
            for param in self.cnn.parameters():
                param.requires_grad = finetune
            """
                requires_grad属性是PyTorch张量（Tensor）对象的属性，用于指定是否在计算过程中对该张量的梯度进行自动计算。
                这个属性默认为True，表示需要计算梯度，以便在反向传播时更新张量的值。
                当然，你可以通过设置这个属性为False来禁止梯度的计算，这在一些情况下很有用，比如在模型参数冻结时或者在进行推理时。
            """
            # 设置VGG19的分类层作为映射层参与训练
            for param in self.cnn.classifier.parameters():
                param.requires_grad = True

        elif 'ResNet' in cnn_type:
            if cnn_type == 'ResNet101':
                self.cnn = models.resnet101(pretrained=True).to(DEVICE)
                """
                    ResNet101的classifier
                    (fc): Linear(in_features=2048, out_features=1000, bias=True)
                """
            elif cnn_type == 'ResNet152':
                self.cnn = models.resnet152(pretrained=True).to(DEVICE)
                """
                    ResNet152的classifier
                    (fc): Linear(in_features=2048, out_features=1000, bias=True)
                """
            # 在原ResNet后面接一个映射层,输入为原fc层的输入2048，输出为指定的嵌入向量大小embed_size
            self.mapping = nn.Linear(
                self.cnn.fc.in_features,
                embed_size,
                bias=True
            )

            # 清空原来的fc层，相当于用映射层替换了fc层
            self.cnn.fc = nn.Sequential()

            # 设置ResNet的卷积层是否参与训练
            for param in self.cnn.parameters():
                param.requires_grad = finetune

        else:
            raise ValueError('Invalid model name')

        if finetune:  # 如果上面选择了要预训练，需要在多个卡上并行，这个的话需要较大显存，我用的显卡(2080 11GB)上支持不了，
            """
                VGG19：共计需要约27GB显存
                ResNet152: 共计需要约34GB显存
            """
            self.cnn = nn.DataParallel(self.cnn)

        # 对应ImageEncoder中的ReLU层
        self.relu = nn.ReLU(inplace=True)

        # 对应ImageEncoder中的Linear Mapping层，使用Xavier初始化
        r = np.sqrt(6.) / np.sqrt(
            self.mapping.in_features + self.mapping.out_features)  # 对应函数torch.nn.init.xavier_uniform
        self.mapping.weight.data.uniform_(-r, r)  # 将权重初始化为在(-r,r)之间均匀分布的随机值
        self.mapping.bias.data.fill_(0)  # 将偏置初始化为0

    def forward(self, images):
        features = self.cnn(images)
        features = self.relu(features)
        features = self.mapping(features)
        features = f.normalize(features, p=2, dim=1)  # L2 Normalize

        return features


class TextEncoder(nn.Module):
    def __init__(self, vocab, word_dim, embed_size, num_layers, rnn_mean_pool, bidirection_rnn, use_word2vec):
        """
            TextEncoder结构(不使用Transformer)
                (1) Word Embedding
                (2) Bi-GRU
                (3) Average Pooling
                (4) Linear Mapping
                (5) L2-Normalize
        """
        super(TextEncoder, self).__init__()
        # 对应TextEncoder中的Word Embedding层
        self.embed_size = embed_size
        self.embed = nn.Embedding(len(vocab), word_dim)  # 生成len(vocab)个维度为word_dim的嵌入向量 (词嵌入矩阵)
        weights = torch.zeros(len(vocab), word_dim)  # 将权重初始化为全零张量
        if use_word2vec:
            if os.path.exists('../word2vec/google-news-300.pt'):
                weights = torch.load('../word2vec/google-news-300.pt')
            else:  # 重新从预训练词向量赋值embedding
                weights.uniform_(-0.1, 0.1)
                print("No existed word vectors,loading from word2vec-google-news-300")
                # 加载预训练词向量文件
                word_vectors = gensim.models.KeyedVectors.load_word2vec_format(
                    '../word2vec/GoogleNews-vectors-negative300.bin.gz', binary=True)
                for word, index in vocab.word2idx.items():
                    # 当前词为特殊标记'<start>'或'<end>'
                    if word == '<start>' or word == '<end>':
                        weights[index] = torch.FloatTensor(word_vectors['</s>'])
                        continue
                    # 当前词不存在于预训练词向量中，直接跳过
                    if word not in word_vectors.key_to_index:
                        continue
                    # 当前词存在于预训练词向量中，更新权重为预训练词向量中该词的词向量表示
                    weights[index] = torch.FloatTensor(word_vectors[word])
            print("loading completed")
        # torch.save(weights , '../word2vec/google-news-300.pt')
        self.embed.weight = nn.Parameter(weights)  # 更新嵌入层权重为weights

        # 对应TextEncoder中的Bi-GRU层和Linear Mapping层
        if bidirection_rnn:
            self.rnn = nn.GRU(word_dim, embed_size, num_layers, batch_first=True, bidirectional=True)
            self.linear = nn.Linear(2 * embed_size, embed_size)  # 双向GRU模型的输出维度是隐藏状态的维度乘以2（因为每个时间步有两个隐藏状态）
        else:
            self.rnn = nn.GRU(word_dim, embed_size, num_layers, batch_first=True, bidirectional=False)
            self.linear = nn.Linear(embed_size, embed_size)

        # 对应TextEncoder中的Average Pooling层
        self.rnn_mean_pool = rnn_mean_pool

    def forward(self, texts, lengths):  # lengths记录每个序列的实际长度(不包含填充)
        embeds = self.embed(texts)  # 转换文本为词嵌入向量
        packed = pack_padded_sequence(embeds, lengths, batch_first=True)  # 将填充后的序列转换成压缩形式，去除填充部分的数据
        output, _ = self.rnn(packed)
        output, _ = pad_packed_sequence(output,
                                        batch_first=True)  # 对RNN输出进行填充解压，恢复原来的形状 (batch_size, max_seq_length, embed_size)

        if self.rnn_mean_pool:
            lengths_tensor = torch.LongTensor(lengths).view(-1,
                                                            1)  # 将长度信息的列表lengths转换为张量，并将其形状调整为(batch_size, 1)，-1表示占位符,以便进行下面的除法
            output = torch.sum(output,
                               dim=1)  # 对输出进行求和操作，沿着序列长度的维度（dim=1）求和，得到每个样本的总和，同时将其形状调整为(batch_size, 1, embed_size)
            output = torch.div(output,
                               lengths_tensor.expand_as(output).to(DEVICE))  # 将长度张量lengths_tensor扩展成与output相同的形状,再相除

        output = self.linear(output)
        output = f.normalize(output, p=2, dim=1)  # L2 Normalize

        return output


class Attention_Textencoder(nn.Module):
    def __init__(self, vocab, word_dim, embed_size, num_heads, num_layers, use_word2vec, rnn_mean_pool):
        """
            TextEncoder结构(使用Transformer)
                (1) Word Embedding
                (2) Transformer Encoder
                (3) Average Pooling
                (4) Linear Mapping
                (5) L2-Normalize
        """
        super(Attention_Textencoder, self).__init__()
        # Word Embedding层同TextEncoder
        self.embed = nn.Embedding(len(vocab), word_dim)
        weights = torch.zeros(len(vocab), word_dim)
        if use_word2vec:
            if os.path.exists('../word2vec/google-news-300.pt'):
                weights = torch.load('../word2vec/google-news-300.pt')
            else:
                weights.uniform_(-0.1, 0.1)
                print("No existed word vectors,loading from word2vec-google-news-300")
                word_vectors = gensim.models.KeyedVectors.load_word2vec_format(
                    '../word2vec/GoogleNews-vectors-negative300.bin.gz', binary=True)
                for word, index in vocab.word2idx.items():
                    if word == '<start>' or word == '<end>':
                        weights[index] = torch.FloatTensor(word_vectors['</s>'])
                        continue
                    if word not in word_vectors.key_to_index:
                        continue
                    weights[index] = torch.FloatTensor(word_vectors[word])
            print("loading completed")
        self.embed.weight = nn.Parameter(weights)

        # 对应Transformer Encoder层,输入和输出数据的形状都为(seq_length, batch_size, word_dim)
        transformer_layer = nn.TransformerEncoderLayer(d_model=word_dim, nhead=num_heads)  # 创建一个Transformer编码器层
        self.encoder = nn.TransformerEncoder(transformer_layer,
                                             num_layers=num_layers)  # 将单个编码器层传递给Encoder，构建一个堆叠了num_layers层的完整编码器模型

        # Linear Mapping层同TextEncoder
        self.linear = nn.Linear(word_dim, embed_size)

        # Average Pooling层同TextEncoder
        self.rnn_mean_pool = rnn_mean_pool

    def forward(self, x, lengths):  # seq_length是填充后的长度，lengths中是实际长度
        x = self.embed(x)  # (batch_size, seq_length, d_model)
        x = self.encoder(x.transpose(0, 1))  # (seq_length, batch_size, d_model) -> torch.Size([31, 128, 300])
        x = x.transpose(0, 1)  # (batch_size, seq_length, d_model) -> torch.Size([128, 31, 300])
        if self.rnn_mean_pool:
            lengths_tensor = torch.LongTensor(lengths).view(-1, 1)
            """
                创建掩码张量mask,x.shape[1]对应seq_length,torch.arange(x.shape[1])生成一个(0,seq_length-1)的一维张量,[None, :]使其变为(1,seq_length)的二维张量
                lengths_tensor[:, None]为(batch_size,1)的二维向量,二者做比较,生成一个(batch_size,seq_length)的二维bool张量
                float()使true变为1.0,false变为0.0,squeeze(1)挤压维度(实际上无影响)
            """
            mask = (torch.arange(x.shape[1])[None, :] < lengths_tensor[:, None]).float().squeeze(1).to(DEVICE)
            masked_output = x * mask.unsqueeze(
                -1)  # unsqueeze(-1)在mask的最后一维上添加一个维度，将形状从(batch_size, seq_length)变为(batch_size, seq_length, 1),方便与x相乘
            x = torch.sum(masked_output, dim=1)
            x = torch.div(x, lengths_tensor.expand_as(x).to(DEVICE))
            x = self.linear(x)
        else:
            x = self.linear(x[:, 0, :])  # 不进行平均池化，则直接取第一个代表整个句子的含义送入映射层

        x = f.normalize(x, p=2, dim=1)

        return x


def cosine_sim(images, captions):  # 只有ContrastiveLoss类中使用
    """
        计算图片和文本之间的余弦相似度，最后返回一个(batch_size , batch_size)的相似度矩阵
        images : (batch_size , embed_dim)
        captions : (batch_size , embed_dim)
    """
    # 计算images沿着第一维的L2范数，得到一个形状为(batch_size, 1)的张量，然后使用unsqueeze(1)在第二维上插入一个新维度使其形状变为 (batch_size, 1),最后将其扩展为与images相同的形状
    images_norm = images.norm(dim=1).unsqueeze(1).expand_as(images) + 1e-6
    # 同images_norm
    captions_norm = captions.norm(dim=1).unsqueeze(1).expand_as(captions) + 1e-6
    # 得到归一化后的图片张量和文本张量
    normalized_images = images / images_norm
    normalized_captions = captions / captions_norm
    # 将两个张量相乘，得到形状为(batch_size, batch_size)的相似度矩阵
    sim_score = torch.mm(normalized_images, normalized_captions.t())

    return sim_score


class ContrastiveLoss(nn.Module):
    """
        计算对比损失hinge-loss
    """
    def __init__(self, margin, max_violation):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.max_violation = max_violation

    def forward(self, images, captions):
        """
            images : (batch_size , embed_dim)
            captions : (batch_size , embed_dim)
        """
        sim_score = cosine_sim(images, captions)
        # 由 batch的构造方式(collate_fn中的zip(*batch))得知，对角线上的图片和文本一定匹配,取其作为正样本,并将得到的对角线元素向量重新塑造为一个(batch_size, 1)的列向量
        positive = sim_score.diag().view(images.size(0), 1)
        positive_image = positive.expand_as(sim_score)  # 图片的正样本得分向量,扩展为sim_score相同形状(batch_size, batch_size)，方便计算
        positive_caption = positive.t().expand_as(sim_score)  # 文本的正样本得分向量,扩展为sim_score相同形状(batch_size, batch_size)，方便计算

        loss_cap2img = (self.margin + sim_score - positive_image).clamp(min=0)  # 以文搜图时的hinge-loss
        loss_img2cap = (self.margin + sim_score - positive_caption).clamp(min=0)  # 以图搜文时的hinge-loss

        mask = torch.eye(sim_score.size(0)) > 0.5  # 创建掩码矩阵，用于在计算损失时排除自身匹配的情况
        variable_mask = Variable(mask).to(DEVICE)  # 将mask转换为Variable对象

        # 将原loss中的值根据variable_mask进行填充.如果 variable_mask中对应位置的值为True,则将对应位置的损失值设为 0；如果为 False，则保持不变,以此排除自身匹配情况下的损失
        loss_cap2img = loss_cap2img.masked_fill_(variable_mask, 0)
        loss_img2cap = loss_img2cap.masked_fill_(variable_mask, 0)

        if self.max_violation:  # 试试先用一般训练，再后面使用max_violation(失败)
            loss_img2cap = loss_img2cap.max(1)[0]  # 以图搜文时，max(1)表示在行维度上查最大值，返回每行的最大值及其索引，[0]表示只要最大值，不要索引,形状变为(batch_size, 1)
            loss_cap2img = loss_cap2img.max(0)[0]  # 以文搜图时

        # if loss_img2cap.mean() + loss_cap2img.mean() == torch.FloatTensor([0.4000]).to(DEVICE):
        #     # 很容易最大的那一个直接与负样本拉开0.2了， 把sim_score和posi_for_cap学成0
        #     print(sim_score)
        #     print(positive_caption)

        return loss_img2cap.mean() + loss_cap2img.mean()


class InfoNCE_contrastiveLoss(nn.Module):
    """
        使用InfoNCE对比损失
    """
    def __init__(self, temperature=0.1, reduction='mean', negative_mode='unpaired'):
        super(InfoNCE_contrastiveLoss, self).__init__()
        self.loss_calcer = InfoNCE(temperature, reduction, negative_mode)

    def forward(self, images, captions):
        """
            images : (batch_size , embed_dim)
            captions : (batch_size , embed_dim)
        """
        all_loss = torch.zeros(images.shape[0] * 2)  # 长度为2 * batch_size,前半部分存储以图搜文的loss,后半部分存储以文搜图的loss
        # 以图搜文
        for index, image in enumerate(images):
            anchor_img = image.view(1, -1)  # 当前图片为锚样本
            positive_cap = captions[index].view(1, -1)  # 与当前图片索引相同的文本为正样本
            negative_caps = captions[torch.arange(captions.shape[0]) != index]  # 与当前索引不同的所有文本为当前图片的负样本
            all_loss[index] = self.loss_calcer(anchor_img, positive_cap, negative_caps)  # 分别传入锚样本,正样本,负样本计算loss
        # 以文搜图
        for index, caption in enumerate(captions):
            anchor_cap = caption.view(1, -1)  # 当前文本为锚样本
            positive_img = images[index].view(1, -1)  # 与当前文本索引相同的图片为正样本
            negative_imgs = images[torch.arange(images.shape[0]) != index]  # 与当前索引不同的所有图片为当前文本的负样本
            all_loss[captions.shape[0] + index] = self.loss_calcer(anchor_cap, positive_img, negative_imgs)  # 分别传入锚样本,正样本,负样本计算loss

        return all_loss.mean()


# 用CLIP中的伪代码重写了一个损失函数试试
# class InfoNCE_contrastiveLoss(nn.Module):
#     """
#         使用InfoNCE对比损失
#     """
#     def __init__(self , temperature=0.1 , reduction='mean'):
#         super(InfoNCE_contrastiveLoss , self).__init__()
#         self.temperature = temperature
#         self.reduction = reduction

#     def forward(self , im , cap):
#         # 模型forward中最后都是normalize
#         # Im_e = F.normalize(im , p=2 , dim=1)
#         # Cap_e = F.normalize(cap , p=2 , dim=1)
#         logits = torch.matmul(im , cap.T) * torch.exp(torch.tensor(self.temperature))
#         labels = torch.arange(im.shape[0]).to(DEVICE)
#         loss_im = F.cross_entropy(logits , labels , reduction=self.reduction)
#         loss_cap = F.cross_entropy(logits.T , labels , reduction=self.reduction)

#         return (loss_im + loss_cap) / 2

class VSE(object):
    def __init__(
            self, embed_size, finetune, word_dim, num_layers, vocab,
            margin, max_violation, grad_clip, use_InfoNCE_loss, rnn_mean_pool,
            bidirection_rnn, use_word2vec, cnn_type, use_attention_for_text, num_heads
    ):
        self.margin = margin
        self.max_violation = max_violation
        self.grad_clip = grad_clip
        self.use_InfoNCE_loss = use_InfoNCE_loss
        # 图像编码器
        self.image_encoder = ImageEncoder(embed_size, cnn_type, finetune).to(DEVICE)
        # 文本编码器
        if not use_attention_for_text:
            self.text_encoder = TextEncoder(vocab, word_dim, embed_size, num_layers, rnn_mean_pool, bidirection_rnn,
                                            use_word2vec).to(DEVICE)
        else:
            self.text_encoder = Attention_Textencoder(vocab, word_dim, embed_size, num_heads, num_layers, use_word2vec,
                                                      rnn_mean_pool).to(DEVICE)
        self.temperature = nn.Parameter(torch.FloatTensor([args.temperature]))  # 准备把温度系数加入训练
        self.params = list(self.image_encoder.parameters()) + list(
            self.text_encoder.parameters())  # 把图像编码器和文本编码器的参数合并成一个参数列表params
        self.params.append(self.temperature)  # 把temperature也加入params，让优化器进行调整

        self.optimizer = torch.optim.Adam(self.params, lr=args.lr)  # 使用Adam优化器
        self.whole_iters = 0  # 记录迭代次数

    def state_dict(self):
        """
            返回一个包含图像编码器和文本编码器状态字典的列表，分别表示它们当前的参数状态
        """
        return [
            self.image_encoder.state_dict(),
            self.text_encoder.state_dict()
        ]

    def load_state_dict(self, state_dict):
        """
            接受一个状态字典作为输入，然后分别加载其中第一个元素（图像编码器的状态字典）到图像编码器中，以及第二个元素（文本编码器的状态字典）到文本编码器中，从而还原模型的参数状态
        """
        self.image_encoder.load_state_dict(state_dict[0])
        self.text_encoder.load_state_dict(state_dict[1])

    def train_model(self):
        """
            将图像编码器和文本编码器设置为训练模式
        """
        self.image_encoder.train()
        self.text_encoder.train()

    def val_model(self):
        """
            将图像编码器和文本编码器设置为评估模式
        """
        self.image_encoder.eval()
        self.text_encoder.eval()

    def forward(self, images, captions, lengths):
        """
            计算图片和文本的位置目标
        """
        if not self.use_InfoNCE_loss:
            self.contrastive_loss = ContrastiveLoss(self.margin, self.max_violation)
        else:
            self.contrastive_loss = InfoNCE_contrastiveLoss(
                self.temperature.cpu().item(),
                args.reduction,
            )
        images = images.to(DEVICE)
        captions = captions.to(DEVICE)
        image_features = self.image_encoder(images)
        caption_features = self.text_encoder(captions, lengths)

        return image_features, caption_features

    def calc_loss(self, img_emb, cap_emb):
        """
            除了调用函数计算损失外，还得顺便记录logger
        """
        loss = self.contrastive_loss(img_emb, cap_emb)
        # 将损失值和批次中的样本数量(用于计算均值)一起记录到日志中
        self.logger.update('Loss', loss.item(), img_emb.size(0))  # image即img_emb : (batch_size , embed_dim)

        return loss

    def train(self, images, captions, lengths):
        """
            对当前批次进行模型训练
        """
        self.train_model()  # 设置模型为训练模式
        self.whole_iters += 1
        self.logger.update('Iteration', self.whole_iters)
        self.logger.update('lr', self.optimizer.param_groups[0]['lr'])  # 通过索引[0]取出第一个参数组，然后从该参数组的字典中获取学习率信息
        image_features, caption_features = self.forward(images, captions, lengths)
        self.optimizer.zero_grad()  # 梯度清零
        loss = self.calc_loss(image_features, caption_features)  # 计算损失值并更新logger
        loss.backward()  # 对损失值进行反向传播
        if self.grad_clip:  # 梯度裁剪
            clip_grad_norm_(self.params, self.grad_clip)
        self.optimizer.step()  # 更新参数
