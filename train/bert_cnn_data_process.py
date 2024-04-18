import torch
from torch.utils.data import Dataset, DataLoader
import os
import torchvision.transforms as transforms
from PIL import Image
import json
from transformers import AutoTokenizer


class FlickrDataset(Dataset):
    """
        对Flickr30k中读取到的数据进行预处理并打包为Dataset类型
    """
    def __init__(self, root, json_path, split, transform=None):
        self.root = root
        self.transform = transform
        self.dataset = json.load(open(json_path, 'r'))['images']
        self.ids = []
        for idx, dataItem in enumerate(self.dataset):
            if dataItem['split'] == split:
                self.ids += [(idx, captionIdx) for captionIdx in range(len(dataItem['sentences']))]  # 一张图片形成5个图文对
                # [(i , x0) , (i , x1) , (i , x2) , (i , x3) , (i , x4)]

    def __getitem__(self, index):
        """
            根据索引，从dataset中抽取对应元素的张量数据，为collate_fn服务
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
        # 预处理图片信息，这里将图片转换成了张量
        if self.transform is not None:
            image = self.transform(image)
        # 预处理文本信息，这里不做任何处理，在collate_fn中处理
        pass

        return image, caption, index

    def __len__(self):
        """
            返回图文对的数量
        """
        return len(self.ids)


def collate_fn(batch):
    """
        为DataLoader服务，将从Dataset中抓取的数据进行打包
    """
    # 对batch进行解压缩操作，将数据集中的每个元组中的元素按位置解压并分配给images、captions和ids
    images, captions, ids = zip(*batch)
    # 整理图像
    images = torch.stack(images, dim=0)
    # 整理文本
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    lengths = [len(cap.split()) for cap in captions]
    captions = tokenizer(
        captions,
        padding=True,  # 不足max_length就填充
        truncation=True,  # 超出max_length就截断
        return_tensors="pt",  # 返回tensor类型
        max_length=max(lengths)
    )
    """
        返回一个字典类型，形如
        {
            'input_ids': tensor([[ 101,  872, 1962, 2769, 1373, 2510,  754, 3234,  102], ---> 对应词汇表中的索引
                [ 101, 2510,  754, 3234, 4696, 2358,  102,    0,    0]]), 
            'token_type_ids': tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0], ---> 属于哪个句子
                [0, 0, 0, 0, 0, 0, 0, 0, 0]]), 
            'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1], ---> 掩码
                [1, 1, 1, 1, 1, 1, 1, 0, 0]])
        }
    """

    return images, captions, lengths, list(ids)


def FlickrDataLoader(split, root, json_path, transform, batch_size, shuffle, num_workers, collate_fn=collate_fn):
    """
        获取DataLoader
    """
    dataset = FlickrDataset(root, json_path, split, transform)
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    return data_loader


def get_transform(split):
    """
        对数据进行预处理
    """
    transform_list = []
    # 变换训练集
    if split == 'train':
        transform_list.append(transforms.RandomResizedCrop(224))
        transform_list.append(transforms.RandomHorizontalFlip())
    # 变换验证集或测试集
    elif split == 'val' or split == 'test':
        transform_list.append(transforms.Resize(256))
        transform_list.append(transforms.CenterCrop(224))
    # 公共变换
    normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform_end = [transforms.ToTensor(), normalizer]
    transform = transforms.Compose(transform_list + transform_end)

    return transform


def get_path(path='../data/'):
    img_path = os.path.join(path, 'flickr30k-images')
    cap_path = os.path.join(path, 'dataset.json')

    return img_path, cap_path


def get_train_dev_loader(batch_size, workers):
    """
        获取训练集或验证集的DataLoader
    """
    # 获取图片和文本路径
    img_path, cap_path = get_path()
    # 获取训练集的DataLoader
    transform = get_transform('train')
    train_loader = FlickrDataLoader(
        split='train',
        root=img_path,
        json_path=cap_path,
        transform=transform,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        collate_fn=collate_fn
    )
    # 获取验证集的DataLoader
    transform = get_transform('val')
    val_loader = FlickrDataLoader(
        split='val',
        root=img_path,
        json_path=cap_path,
        transform=transform,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        collate_fn=collate_fn
    )

    return train_loader, val_loader


def get_test_loader(batch_size, workers):
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
        transform=transform,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        collate_fn=collate_fn
    )

    return test_loader
