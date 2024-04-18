from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import json
from param import args
from transformers import AutoTokenizer
from transformers import AutoFeatureExtractor


class FlickrDataset(Dataset):
    """
        对Flickr30k中读取到的数据进行预处理并打包为Dataset类型
    """
    def __init__(self, root, json_path, split):
        self.root = root
        self.dataset = json.load(open(json_path, 'r'))['images']
        self.ids = []
        for idx, dataItem in enumerate(self.dataset):
            if dataItem['split'] == split:
                self.ids += [(idx, capIdx) for capIdx in range(len(dataItem['sentences']))]  # 一张图片形成5个图文对
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
        # 预处理图片信息，这里不做任何处理，在collate_fn中处理
        pass
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
    """
        查阅官方文档 https://huggingface.co/docs/transformers/main/en/model_doc/auto 中的 AutoFeatureExtractor部分
        发现vit-msn 对应的AutoFeatureExtractor类为 ViTFeatureExtractor(现在已被废弃,应该使用ViTImageProcessor)
        ViTImageProcessor继承自BaseImageProcessor,其__call__调用preprocess(images, **kwargs)方法进行预处理
        ViTImageProcessor中的preprocess(images, **kwargs)方法最后生成一个数据字典data = {"pixel_values": images} 
        最后 return BatchFeature(data=data, tensor_type=return_tensors) 返回images的张量形式(令return_tensors="pt")
        
        BatchFeature源码:
            class BatchFeature(BaseBatchFeature):

            Holds the output of the image processor specific `__call__` methods.

            This class is derived from a python dictionary and can be used as a dictionary.

            Args:
                data (`dict`):
                    Dictionary of lists/arrays/tensors returned by the __call__ method ('pixel_values', etc.).
                tensor_type (`Union[None, str, TensorType]`, *optional*):
                    You can give a tensor_type here to convert the lists of integers in PyTorch/TensorFlow/Numpy Tensors at
                    initialization.
        
    """
    image_processor = AutoFeatureExtractor.from_pretrained("facebook/vit-msn-small")  # 获得一个ViTImageProcessor,用于对图片进行预处理以满足ViT的格式
    images = image_processor(images=images, return_tensors="pt")  # 获得一个字典,格式为{"pixel_values": images}，其中images为张量形式
    images = images['pixel_values']  # 获得图片的张量
    # 整理文本
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    lengths = [len(cap.split()) for cap in captions]
    captions = tokenizer(
        captions,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=max(lengths)
    )

    return images, captions, lengths, list(ids)


def FlickrDataLoader(split, root, json_path, batch_size, shuffle, num_workers, collate_fn=collate_fn):
    """
        获取DataLoader
    """
    dataset = FlickrDataset(root, json_path, split)
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    return data_loader


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
    train_loader = FlickrDataLoader(
        split='train',
        root=img_path,
        json_path=cap_path,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        collate_fn=collate_fn
    )
    # 获取验证集的DataLoader
    val_loader = FlickrDataLoader(
        split='val',
        root=img_path,
        json_path=cap_path,
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
    test_loader = FlickrDataLoader(
        split='test',
        root=img_path,
        json_path=cap_path,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        collate_fn=collate_fn
    )

    return test_loader


if __name__ == "__main__":
    train_loader, val_loader = get_train_dev_loader(
        args.batch_size,
        args.workers
    )
    for idx, train_data in enumerate(train_loader):
        images, captions, lengths, _ = train_data
        if idx in [0, 1, 2, 3]:
            print(images, images.shape)  # tensor (batch_size , 3 , 224 , 224)
            print(captions)  # dict: {input_ids: ... , attention_mask: ...}
            print(lengths)  # list (batch_size)
        else:
            break
