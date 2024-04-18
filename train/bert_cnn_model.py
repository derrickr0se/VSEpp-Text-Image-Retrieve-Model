import torch
import torch.nn as nn
from param import args
from transformers import BertModel
import torch.nn.functional as f
from info_nce import InfoNCE
from torch.nn.utils import clip_grad_norm_
from torch.autograd import Variable
import torchvision.models as models
import numpy as np

if args.gpu.lower() == 'cpu':
    DEVICE = torch.device('cpu')
elif args.gpu in ['0', '1', '2', '3']:
    DEVICE = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
else:
    raise ValueError('Invalid GPU ID')


class ImageEncoder(nn.Module):
    def __init__(self, embed_size, cnn_type, finetune=False):
        super(ImageEncoder, self).__init__()
        self.embed_size = embed_size
        if cnn_type == 'VGG19':
            self.cnn = models.vgg19(pretrained=True).to(DEVICE)
            # 在VGG19后面接一个映射层
            self.mapping = nn.Linear(
                self.cnn.classifier[6].out_features,
                embed_size,
                bias=True
            )
            # 设置VGG19的classifier是否参与训练
            for param in self.cnn.parameters():
                param.requires_grad = finetune

            # 分类层作为映射层参与训练
            for param in self.cnn.classifier.parameters():
                param.requires_grad = True

        elif 'ResNet' in cnn_type:
            if cnn_type == 'ResNet101':
                self.cnn = models.resnet101(pretrained=True).to(DEVICE)
            elif cnn_type == 'ResNet152':
                self.cnn = models.resnet152(pretrained=True).to(DEVICE)
            # 在ResNet后面接一个映射层
            self.mapping = nn.Linear(
                self.cnn.fc.out_features,
                embed_size,
                bias=True
            )
            # 设置ResNet的classifier是否参与训练
            for param in self.cnn.parameters():
                param.requires_grad = finetune
            # 分类层作为映射层参与训练
            for param in self.cnn.fc.parameters():
                param.requires_grad = True
        else:
            raise ValueError('Invalid model name')
        if finetune:  # 要预训练这个的话需要较大显存，我用的显卡(2080 11GB)上支持不了，需要在多个卡上并行
            """
                VGG19：共计需要约27GB显存
                ResNet152: 共计需要约34GB显存
            """
            self.cnn = nn.DataParallel(self.cnn)

        self.relu = nn.ReLU(inplace=True)

        # 使用Xavier初始化fc层
        r = np.sqrt(6.) / np.sqrt(self.mapping.in_features + self.mapping.out_features)
        self.mapping.weight.data.uniform_(-r, r)
        self.mapping.bias.data.fill_(0)

    def forward(self, images):
        features = self.cnn(images)
        features = self.relu(features)
        features = self.mapping(features)
        features = f.normalize(features, p=2, dim=1)

        return features


class BertTextEncoder(nn.Module):
    """
        BertTextEncoder结构
            (1) BERT(内含Word Embedding)
            (2) ReLU
            (3) Linear Mapping
            (4) L2-Normalize
    """
    def __init__(self, embed_size, finetune=False):
        super(BertTextEncoder, self).__init__()
        self.encoder = BertModel.from_pretrained("bert-base-uncased")
        self.encoder = nn.DataParallel(self.encoder)
        self.relu = nn.ReLU(inplace=True)
        self.mapping = nn.Linear(768, embed_size)  # 经过BERT后输出的词向量维度d_model=768

        # 使用Xavier初始化fc层
        r = np.sqrt(6.) / np.sqrt(self.mapping.in_features + self.mapping.out_features)
        self.mapping.weight.data.uniform_(-r, r)
        self.mapping.bias.data.fill_(0)

    def forward(self, captions, lengths):
        # 只需要给出每个词的索引，BERT内部会先将其转化成Word Embedding,再送入Encoder,不需要自己计算Embedding
        captions = captions.input_ids.to(DEVICE)  # input_ids: Indices of input sequence tokens in the vocabulary.
        """
            last_hidden_state 获取最后一个隐藏层的隐藏状态，其保存了输入序列中每个token经处理后对应的词向量,形状为[batch_size, sequence_length, word_dim]
            [:, 0, :]提取每个序列的第一个token(即<cls>)对应的词向量,形状为[batch_size, word_dim],作为整句话的语义表示
        """
        outputs = self.encoder(captions).last_hidden_state[:, 0, :]
        outputs = self.relu(outputs)
        outputs = self.mapping(outputs)
        outputs = f.normalize(outputs, p=2, dim=1)

        return outputs


def cosine_sim(images, captions):  # 只有ContrastiveLoss类中使用
    """
        计算图片和文本的余弦相似度,同model.py
        im : (batch_size , embed_dim)
        cap : (batch_size , embed_dim)
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
        计算对比损失hinge-loss,同model.py
    """
    def __init__(self, margin, max_violation):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.max_violation = max_violation

    def forward(self, images, captions):
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
            loss_img2cap = loss_img2cap.max(1)[0]  # 以图搜文时，max(1)表示在行维度上查找最大值，返回每行的最大值及其索引，[0]表示只选择最大值，不需要索引,形状变为(batch_size, 1)
            loss_cap2img = loss_cap2img.max(0)[0]  # 以文搜图时

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


class BERT_CNN_VSE(object):
    def __init__(self, embed_size, finetune, margin, max_violation, grad_clip, use_InfoNCE_loss, cnn_type):
        self.margin = margin
        self.max_violation = max_violation
        self.grad_clip = grad_clip
        self.use_InfoNCE_loss = use_InfoNCE_loss
        # 图像编码器
        self.image_encoder = ImageEncoder(embed_size, cnn_type, finetune).to(DEVICE)
        # 文本编码器
        self.text_encoder = BertTextEncoder(embed_size, finetune).to(DEVICE)

        self.temperature = nn.Parameter(torch.FloatTensor([args.temperature]))  # 准备把这个系数加入训练
        self.params = list(self.image_encoder.parameters()) + list(self.text_encoder.parameters())
        self.params.append(self.temperature)  # 加入温度系数
        self.optimizer = torch.optim.Adam(self.params, lr=args.lr)
        self.whole_iters = 0

    def state_dict(self):
        return [
            self.image_encoder.state_dict(),
            self.text_encoder.state_dict()
        ]

    def load_state_dict(self, state_dict):
        self.image_encoder.load_state_dict(state_dict[0])
        self.text_encoder.load_state_dict(state_dict[1])

    def train_model(self):
        self.image_encoder.train()
        self.text_encoder.train()

    def val_model(self):
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
        # captions = captions.to(DEVICE)  此处captions由collate_fn得到,是dict类型，在模型中再传入CUDA
        image_features = self.image_encoder(images)
        caption_features = self.text_encoder(captions, lengths)

        return image_features, caption_features

    def calc_loss(self, img_emb, cap_emb):
        """
            除了调用函数计算损失外，还得顺便记录logger
        """
        loss = self.contrastive_loss(img_emb, cap_emb)
        self.logger.update('Loss', loss.item(), img_emb.size(0))
        return loss

    def train(self, images, captions, lengths):  # 此处captions是dict类型
        self.train_model()
        self.whole_iters += 1
        self.logger.update('Iteration', self.whole_iters)
        self.logger.update('lr', self.optimizer.param_groups[0]['lr'])
        im_features, cap_features = self.forward(images, captions, lengths)
        self.optimizer.zero_grad()
        loss = self.calc_loss(im_features, cap_features)
        loss.backward()
        if self.grad_clip:
            clip_grad_norm_(self.params, self.grad_clip)
        self.optimizer.step()
