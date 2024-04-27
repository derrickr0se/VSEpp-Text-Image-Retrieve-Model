import nltk
import pickle
from collections import Counter
import json
import argparse
import os
from tqdm import tqdm

"""
nltk.download('punkt')
dataset.json中的数据形式：
    keys: ['images' , 'dataset']

    images: {
        "sentids": [0, 1, 2, 3, 4],
        "imgid": 0, 
        "sentences": [
            {"tokens": ["two", "young", "guys", "with", "shaggy", "hair", "look", "at", "their", "hands", "while", "hanging", "out", "in", "the", "yard"], 
            "raw": "Two young guys with shaggy hair look at their hands while hanging out in the yard.", "imgid": 0, "sentid": 0}, 
            {"tokens": ["two", "young", "white", "males", "are", "outside", "near", "many", "bushes"], 
            "raw": "Two young, White males are outside near many bushes.", "imgid": 0, "sentid": 1}, 
            {"tokens": ["two", "men", "in", "green", "shirts", "are", "standing", "in", "a", "yard"],
            "raw": "Two men in green shirts are standing in a yard.", "imgid": 0, "sentid": 2}, 
            {"tokens": ["a", "man", "in", "a", "blue", "shirt", "standing", "in", "a", "garden"], 
            "raw": "A man in a blue shirt standing in a garden.", "imgid": 0, "sentid": 3}, 
            {"tokens": ["two", "friends", "enjoy", "time", "spent", "together"], "raw": "Two friends enjoy time spent together.", "imgid": 0, "sentid": 4}
        ],  
        "split": "train", 
        "filename": "1000092795.jpg"
    }

    dataset: 'flickr30K'
"""


class Vocabulary(object):
    """
        创建和管理词汇表
    """

    def __init__(self):
        """
            创建将单词映射到索引和将索引映射到单词的两个字典，初始化索引下标
        """
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        """
            如果单词不存在于 word2idx 字典中，则将单词添加到词汇表中，并为其分配一个唯一的索引
        """
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        """
            给定一个单词，从 word2idx 字典中返回对应的索引。如果找不到该单词，则返回未知标记 '<unk>'
        """
        if word not in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        """
            返回词汇表的长度，即存储在 word2idx 字典中的唯一单词数目
        """
        return len(self.word2idx)


def get_captions(json_path):
    """
        从Flickr30k的json文件中加载数据集，并从中提取文本
    """
    dataset = json.load(open(json_path, 'r'))['images']
    captions = []
    for index, data in enumerate(dataset):
        captions += [str(sentence['raw']) for sentence in data['sentences']]
    return captions


def build_vocab(data_path, data_name, threshold):
    """
        构建词汇表包装器
    """
    counter = Counter()  # 用于统计词频并丢弃出现次数低于阈值的词
    captions = get_captions(os.path.join(data_path, data_name))

    for index, caption in tqdm(enumerate(captions)):
        tokens = nltk.tokenize.word_tokenize(caption.lower())
        counter.update(tokens)  # 更新计数器
        if index % 1000 == 0:
            print("[%d/%d] tokenized the captions." % (index, len(captions)))

    # 使用items()方法获取所有键值对，将词频不低于阈值的词添加到words中
    words = [word for word, cnt in counter.items() if cnt >= threshold]

    # 初始化词汇表，添加特殊标记
    vocab = Vocabulary()
    vocab.add_word('<pad>')  # 用于填充长度不足的张量，索引为0
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    # 将文本中出现次数高于阈值的词添加到词汇表中
    for index, word in enumerate(words):
        vocab.add_word(word)

    return vocab


def initiate(data_path, data_name):
    """
        初始化词汇表并将其保存到指定路径下
    """
    vocab = build_vocab(data_path, data_name, threshold=3)
    print(f"length of vocabulary : {vocab.__len__()}")
    # 将构建好的词汇表对象保存到pickle文件中
    with open('../vocab/flickr30k_vocab.pkl', 'wb') as file:
        pickle.dump(vocab, file, pickle.HIGHEST_PROTOCOL)
    print("Successfully save vocabulary file to ", '../vocab/flickr30k_vocab.pkl')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='../data/', help="directory path for Flickr30k images")
    parser.add_argument('--data_name', type=str, default='dataset.json', help="name of Flickr30k json file")
    args = parser.parse_args()
    initiate(args.data_path, args.data_name)
