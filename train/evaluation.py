import os
import pickle
import torch
from collections import OrderedDict
import numpy as np
import time
import data_process
import transformer_data_process
import bert_cnn_data_process
from model import VSE
from transformer_model import transformer_VSE
from bert_cnn_model import BERT_CNN_VSE
from pprint import pprint
from param import args

if args.gpu.lower() == 'cpu':
    DEVICE = torch.device('cpu')
elif args.gpu in ['0', '1', '2', '3']:
    DEVICE = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
else:
    raise ValueError('Invalid GPU ID')


class AverageMeter(object):
    """
        计算并且存储当前值和总计平均值(平滑)
    """

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (self.count + 1e-5)

    def __str__(self):  # 打印loss当前值和平均值
        if self.count == 0:
            return str(self.val)
        return '%.4f (average: %.4f)' % (self.val, self.avg)


class LogCollector(object):
    """
        记录train和val的logging的对象
    """

    def __init__(self):
        self.meters = OrderedDict()  # meters是一个有序字典

    def update(self, key, value, n=1):
        if key not in self.meters:  # 没有当前键，先创建
            self.meters[key] = AverageMeter()
        self.meters[key].update(value, n)  # 更新值

    def __str__(self):
        s = ''
        for index, (key, value) in enumerate(self.meters.items()):  # 拼接字符串
            if index > 0:
                s += ' '
            s += f' {key} : {value}'
        return s

    def tensorboard_log(self, tb_logger, prefix='', step=None):  # 将记录的指标写入TensorBoard日志
        for key, value in self.meters.items():
            tb_logger.log_value(prefix + key, value.val, step=step)


def encode_data(model, data_loader, log_step=10, logging=print):
    batch_time_meter = AverageMeter()
    val_logger = LogCollector()
    model.val_model()

    start_time = time.time()
    img_embs = None
    cap_embs = None
    isInit = False

    for index, (images, captions, lengths, ids) in enumerate(data_loader):
        model.logger = val_logger

        with torch.no_grad():  # 禁用梯度计算,加速推断过程
            img_emb, cap_emb = model.forward(images, captions, lengths)  # (batch_size , embed_dim)

            # 初始化
            if not isInit:
                img_embs = np.zeros((len(data_loader.dataset), img_emb.shape[1]))  # (batch_size , embed_dim)
                cap_embs = np.zeros((len(data_loader.dataset), cap_emb.shape[1]))
                isInit = True

            img_embs[ids] = img_emb.data.cpu().numpy().copy()  # 转移到cpu上以便进行numpy().copy()
            cap_embs[ids] = cap_emb.data.cpu().numpy().copy()

            model.calc_loss(img_emb, cap_emb)  # 不用接返回值，只需要其中记录到logger的功能

        batch_time_meter.update(time.time() - start_time)

        if index % log_step == 0:  # 打印日志
            logging(
                'Test: [{index}/{len}]\t'
                '{e_log}\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                .format(
                    index=index,
                    len=len(data_loader),
                    e_log=model.logger,
                    batch_time=batch_time_meter
                )
            )

            start_time = time.time()

    return img_embs, cap_embs


def r_img2cap(img_embs, cap_embs):
    """
        计算以图搜文的召回率指标
    """
    image_num = int(img_embs.shape[0] / 5)  # 一张图片对应五条文本
    # ranks[i] : 与第i张图片最匹配的正样本文本在对应第i张图片的sim_score_i中的下标
    # 例如,ranks[3] = 6，说明与第4张图片最匹配的正样本文本(对应的五条文本中相似度最大的那个)在其sim_score中的下标为6，即第7大相似的，那么算r_im2cap_1和r_im2cap_5时都不能算在内
    ranks = np.zeros(image_num)
    for index in range(image_num):
        # 获取查询图片，形状为(1 , embed_dim)
        image = img_embs[5 * index].reshape(1, img_embs.shape[1])
        # 计算每个图片对应的相似度矩阵，形状为(1 , batch_size)
        sim_score_i = np.dot(image, cap_embs.T)  # sim_score_i[5 * i] ~ sim_score_i[5 * i + 4]对应第 i 张图片与其匹配的五条文本(正样本)
        # 先对sim_score进行升序排序，然后利用[::-1]倒序排列,indices记录排序后的索引
        indices = np.argsort(sim_score_i[0])[::-1]  # 下标范围 : 0 ~ batch_size-1,下标越小相似度越大
        min_indice = 1000000000  # 记录最小索引
        """
            np.where(indices == i)以元组形式(array,dtype)返回所有满足indices == i元素的索引,第一个[0]取出其中的array部分,第二个[0]取出array中第一个元素
        """
        for i in range(5 * index, 5 * index + 5, 1):  # 在匹配的五条文本(正样本)中找到下标最靠前的(即相似度最高的)，保存到ranks中
            tmp = np.where(indices == i)[0][0]  # 第i条文本在indices中的下标
            if tmp < min_indice:
                min_indice = tmp
        ranks[index] = min_indice  # ranks的元素是可以重复的，因为每个图片的sim_score_i不一样

    # 计算召回率指标 : 前K个检索出的文本中至少有一个正样本的查询句子占比(即前K个检索出的文本中存在最匹配的那个正样本的查询句子占比)
    r_img2cap_1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r_img2cap_5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r_img2cap_10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    r_img2cap_median = np.floor(np.median(ranks)) + 1
    r_img2cap_mean = np.floor(np.mean(ranks)) + 1

    return r_img2cap_1, r_img2cap_5, r_img2cap_10, r_img2cap_median, r_img2cap_mean


def r_cap2img(img_embs, cap_embs):
    """
        计算以文搜图的召回率指标
    """
    image_num = int(img_embs.shape[0] / 5)
    imgs = np.array([img_embs[5 * i] for i in range(image_num)])  # (image_num , embed_dim)
    ranks = np.zeros(img_embs.shape[0])  # (batch_size , 1)

    for index in range(image_num):
        # 获取查询文本,形状为 (5 , embed_dim)
        caps = cap_embs[5 * index: 5 * index + 5]  # 对应索引为index的图片与其匹配的五条文本(正样本)
        # 计算相似度矩阵,形状为 (5 , image_num)
        sim_score = np.dot(caps, imgs.T)
        indices = np.zeros(sim_score.shape)

        for i in range(len(indices)):
            indices[i] = np.argsort(sim_score[i])[::-1]  # 对indices每一行进行降序排序
            ranks[5 * index + i] = np.where(indices[i] == index)[0][0]  # 文本对于图片的正样本是唯一的，不必再去找最匹配的那一个

    # 计算召回率指标
    r_cap2img_1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r_cap2img_5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r_cap2img_10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    r_cap2img_median = np.floor(np.median(ranks)) + 1
    r_cap2img_mean = np.floor(np.mean(ranks)) + 1

    return r_cap2img_1, r_cap2img_5, r_cap2img_10, r_cap2img_median, r_cap2img_mean


def evalrank(model_path):
    """
        使用测试集评估训练完毕的模型效果
    """
    checkpoint = torch.load(model_path)  # 加载模型检查点
    old_args = checkpoint['args']  # 获得外部参数(param.py)

    with open(os.path.join(old_args.vocab_path, 'flickr30k_vocab.pkl'), 'rb') as f:
        vocab = pickle.load(f)
    old_args.vocab_size = len(vocab)

    print("-----------------------------------------------")
    pprint(old_args)
    print("-----------------------------------------------")

    if old_args.model_class == 'CNN_and_GRU':
        # 构建模型
        model = VSE(
            old_args.embed_size,
            old_args.finetune,
            old_args.word_dim,
            old_args.num_layers,
            vocab,
            old_args.margin,
            old_args.max_violation,
            old_args.grad_clip,
            old_args.use_InfoNCE_loss,
            old_args.rnn_mean_pool,
            old_args.bidirection_rnn,
            old_args.use_word2vec,
            old_args.cnn_type,
            old_args.use_attention_for_text,
            old_args.num_heads
        )
    elif old_args.model_class == 'ViT_and_BERT':
        # 构建模型
        model = transformer_VSE(
            old_args.embed_size,
            old_args.finetune,
            old_args.margin,
            old_args.max_violation,
            old_args.grad_clip,
            old_args.use_InfoNCE_loss
        )
    elif old_args.model_class == 'CNN_and_BERT':
        # 构建模型
        model = BERT_CNN_VSE(
            old_args.embed_size,
            old_args.finetune,
            old_args.margin,
            old_args.max_violation,
            old_args.grad_clip,
            old_args.use_InfoNCE_loss,
            old_args.cnn_type
        )
    else:
        raise ValueError('Wrong model class !! It must be in (CNN_and_GRU , ViT_and_BERT , CNN_and_BERT)')

    model.load_state_dict(checkpoint['model'])  # 加载模型内部参数

    print('Loading dataset')
    # 构建测试集的DataLoader
    if old_args.model_class == 'CNN_and_GRU':
        data_loader = data_process.get_test_loader(
            vocab,
            old_args.batch_size,
            old_args.workers
        )
    elif old_args.model_class == 'ViT_and_BERT':
        data_loader = transformer_data_process.get_test_loader(
            old_args.batch_size,
            old_args.workers
        )
    elif old_args.model_class == 'CNN_and_BERT':
        data_loader = bert_cnn_data_process.get_test_loader(
            old_args.batch_size,
            old_args.workers
        )
    else:
        raise ValueError('Wrong model class !! It must be in (CNN_and_GRU , ViT_and_BERT , CNN_and_BERT)')

    print('Computing result....')
    img_embs, cap_embs = encode_data(model, data_loader)
    print(f'Image num: {img_embs.shape[0] / 5} , Caption num: {cap_embs.shape[0]}')

    # 获得召回率指标
    r_img2caps = r_img2cap(img_embs, cap_embs)
    r_cap2imgs = r_cap2img(img_embs, cap_embs)

    # 计算召回率指标均值 (r_1 + r_5 + r_10) / 3
    avg_r_img2cap = (r_img2caps[0] + r_img2caps[1] + r_img2caps[2]) / 3
    avg_r_cap2img = (r_cap2imgs[0] + r_cap2imgs[1] + r_cap2imgs[2]) / 3

    # 计算总体召回率指标
    r_sum = r_img2caps[0] + r_img2caps[1] + r_img2caps[2] + r_cap2imgs[0] + r_cap2imgs[1] + r_cap2imgs[2]

    print("----------------------------------------")
    print("r_sum : %.1f" % r_sum)
    print("Average img2cap Recall: %.1f" % avg_r_img2cap)
    print("Image to text: %.1f %.1f %.1f %.1f %.1f" % r_img2caps)
    print("Average cap2img Recall: %.1f" % avg_r_cap2img)
    print("Text to image: %.1f %.1f %.1f %.1f %.1f" % r_cap2imgs)
