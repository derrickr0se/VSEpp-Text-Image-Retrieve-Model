import torch
from torch.utils.data import Dataset, DataLoader
import os
import torchvision.transforms as transforms
import nltk
from PIL import Image
import json


class FlickrDataset(Dataset):
    def __init__(self, root, json_path, split, vocab, transform=None):
        """
            对Flickr30k中读取到的数据进行预处理并打包为Dataset
        """
        self.root = root
        self.vocab = vocab
        self.transform = transform
        self.dataset = json.load(open(json_path, 'r'))['images']
        self.ids = []
        for img_idx, data in enumerate(self.dataset):
            if data['split'] == split:
                # 一张图片形成五个图文对的元组 [(i , 0) , (i , 1) , (i , 2) , (i , 3) , (i , 4)] 并加入ids中
                self.ids += [(img_idx, cap_idx) for cap_idx in range(len(data['sentences']))]

    def __getitem__(self, index):
        """
            根据索引，从Dataset中抽取对应元素的张量数据，为collate_fn服务
        """
        # 从dataset中取出图片和描述文本的索引
        pair_ids = self.ids[index]
        img_id = pair_ids[0]
        caption_id = pair_ids[1]
        # 获取图片
        img_path = self.dataset[img_id]['filename']
        image = Image.open(os.path.join(self.root, img_path)).convert('RGB')
        # 获取描述文本
        caption = self.dataset[img_id]['sentences'][caption_id]['raw']
        # 预处理图片信息,这里将图片转换成了张量
        if self.transform is not None:
            image = self.transform(image)
        # 预处理文本信息,将文本(字符串)转换为词汇表中索引
        tokens = nltk.tokenize.word_tokenize(str(caption).lower())  # 将文本转换为小写并分词
        caption_ids = []
        caption_ids.append(self.vocab('<start>'))  # 加入<start>在词汇表中对应的索引
        caption_ids.extend([self.vocab(token) for token in tokens])  # 加入所有token在词汇表中对应的索引
        caption_ids.append(self.vocab('<end>'))  # 加入<end>在词汇表中对应的索引
        caption_ids_tensor = torch.Tensor(caption_ids)  # 转换为张量

        return image, caption_ids_tensor, index

    def __len__(self):
        """
            返回图文对的数量
        """
        return len(self.ids)


def collate_fn(batch):
    """
        默认的collate_fn有一个很大的限制:批数据必须处于同一维度
        这里由于文本不等长，必须自定义一个collate_fn函数

        Args:
            batch: list of (image, caption_ids_tensor, index) tuple.(长度为 batch_size)
                - image: torch tensor of shape (3, 256, 256).
                - caption_ids_tensor: torch tensor of shape (?); variable length.

        Returns:
            images: torch tensor of shape (batch_size, 3, 256, 256).
            caption_ids_tensors: torch tensor of shape (batch_size, padded_length).
            lengths: list; valid length for each padded caption.
    """
    batch.sort(key=lambda x: len(x[1]), reverse=True)  # 根据每个数据项的第二个元素（即描述文本）的长度降序排序
    # 这里保证了对于第i张图片，第i个文本一定是与其匹配的正样本，但由于一张图片对应五个文本，可能存在<i,j>(i!=j)也是匹配的正样本
    images, captions, ids = zip(*batch)  # 对batch进行解压缩操作，将数据集中的每个元组中的元素按位置解压并分配给images、captions和ids
    # 整理图像
    images = torch.stack(images, dim=0)  # 将所有图像张量堆叠成一个张量，其中新增维度位于新张量的第一个位置，堆叠后形状为(batch_size, channel, height, width)
    # 整理文本
    lengths = [len(cap) for cap in captions]  # 记录每个文本的实际长度(不含填充)
    caption_ids_tensors = torch.zeros(len(captions), max(lengths), dtype=torch.long)  # 创建一个形状为 (batch_size, max_seq_length) 的全零张量，用于存储所有描述文本的张量
    # 填充张量
    for index, caption in enumerate(captions):
        real_len = lengths[index]  # 当前文本的实际长度
        caption_ids_tensors[index, :real_len] = caption[:real_len]

    return images, caption_ids_tensors, lengths, list(ids)


def FlickrDataLoader(split, root, json_path, vocab, transform, batch_size, shuffle, num_workers, collate_fn=collate_fn):
    """
        获取Flickr30k Dataset对应的DataLoader
    """
    dataset = FlickrDataset(
        root=root,
        json_path=json_path,
        split=split,
        vocab=vocab,
        transform=transform
    )

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=num_workers,
        collate_fn=collate_fn
    )

    return dataloader


def get_transform(split):
    """
        对数据进行预处理
    """
    transform_list = []  # 创建一个空的图像变换操作列表
    # 变换训练集
    if split == 'train':
        transform_list.append(transforms.RandomResizedCrop(224))  # 随机裁剪，并将裁剪后的图像大小调整为224x224
        transform_list.append(transforms.RandomHorizontalFlip())  # 随机水平翻转
    # 变换验证集或测试集
    elif split == 'val' or split == 'test':
        transform_list.append(transforms.Resize(256))  # 将图像大小调整至256x256
        transform_list.append(transforms.CenterCrop(224))  # 将图像在中心位置进行裁剪，裁剪后的图像大小为224x224
    # 公共变换
    normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化，分别对应 RGB 三个通道
    transform_common = [transforms.ToTensor(), normalizer]  # 转换成张量形式并归一化
    transform = transforms.Compose(transform_list + transform_common)

    return transform


def get_path(path='../data/'):
    """
        获取图片和描述文本的路径
    """
    img_path = os.path.join(path, 'flickr30k-images')
    cap_path = os.path.join(path, 'dataset.json')
    return img_path, cap_path


def get_train_val_loader(vocab, batch_size, workers):
    """
        获取训练集或验证集的DataLoader
    """
    # 获取图片和文本路径
    img_path, cap_path = get_path()
    # 获取训练集的DataLoader
    train_transform = get_transform('train')
    train_loader = FlickrDataLoader(
        split='train',
        root=img_path,
        json_path=cap_path,
        vocab=vocab,
        transform=train_transform,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        collate_fn=collate_fn
    )
    # 获取验证集的DataLoader
    val_transform = get_transform('val')
    val_loader = FlickrDataLoader(
        split='val',
        root=img_path,
        json_path=cap_path,
        vocab=vocab,
        transform=val_transform,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        collate_fn=collate_fn
    )

    return train_loader, val_loader


def get_test_loader(vocab, batch_size, workers):
    """
        获取测试集的DataLoader
    """
    # 获取图片和文本路径
    img_path, cap_path = get_path()
    # 获取验证集的DataLoader
    transform = get_transform('test')
    test_loader = FlickrDataLoader(
        split='test',
        root=img_path,
        json_path=cap_path,
        vocab=vocab,
        transform=transform,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        collate_fn=collate_fn
    )

    return test_loader
