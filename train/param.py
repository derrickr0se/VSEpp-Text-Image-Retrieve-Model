import argparse

# 记录超参数
parser = argparse.ArgumentParser()

parser.add_argument("--gpu", type=str, default="0",
                    help="choose the index of the gpu used to run")
parser.add_argument("--lr", type=float, default=0.0002,
                    help="initial learning rate")
parser.add_argument("--log_dir", type=str, default="./tensorboard_logs/",
                    help="directory path for generate logging information")
parser.add_argument("--vocab_path", type=str, default="../vocab/",
                    help="Path to saved vocabulary pickle files")
parser.add_argument("--batch_size", type=int, default=32,
                    help="size of a training mini-batch")
parser.add_argument("--workers", type=int, default=5,
                    help="number of data loader workers")
parser.add_argument("--embed_size", type=int, default=1024,
                    help="dimensionality of the joint embedding")
parser.add_argument("--finetune", action='store_true', default=False,
                    help="whether finetune pretrained cnn-net embedding")
parser.add_argument("--word_dim", type=int, default=300,
                    help="dimensionality of the word embedding")
parser.add_argument("--num_layers", type=int, default=2,
                    help="number of layers for GRU or attentionBlocks")
parser.add_argument("--bidirection_rnn", action='store_true', default=False,
                    help="whether use bidirection for GRU")
parser.add_argument("--rnn_mean_pool", action='store_true', default=False,
                    help='whether to use mean pool for the end of GRU net and attention net')
parser.add_argument("--margin", type=float, default=0.2,
                    help="rank loss margin")
parser.add_argument("--max_violation", action='store_true', default=False,
                    help="use max instead of mean in the rank loss")
'''
对于每个anchor样本，选择最难以区分的positive样本和最容易区分的negative样本，然后通过一定的margin来确保它们之间的差异足够大。
具体来说，对于每个anchor样本，我们会选择一个positive样本  和一个negative样本 ，使得它们与anchor样本的相似度之间的差异最大化
'''
parser.add_argument("--grad_clip", type=float, default=2.0,
                    help="gradient clipping threshold")
'''
梯度裁剪的目的是防止梯度爆炸的问题，即当梯度值过大时可能导致模型不稳定甚至无法收敛的情况。
通过设定一个阈值（如最大梯度范数），如果计算得到的梯度的范数超过了这个阈值，就会对所有的梯度进行缩放，使其范数不超过设定的阈值。这样可以保证梯度的大小在一个合理的范围内，避免参数更新过大而导致模型不稳定。
'''
parser.add_argument("--num_epochs", type=int, default=30,
                    help="number of training epochs")  # 基本训练到16，17就开始过拟合，然后下降了
parser.add_argument("--log_step", type=int, default=10,
                    help="number of steps to print and record the log")  # 每训练 log_step 个batch就会记录并输出一次训练指标
parser.add_argument("--val_step", type=int, default=500,
                    help="number of steps to run validation")  # 每训练 val_step 个batch就会使用验证集对模型进行评估
parser.add_argument("--lr_decay_step", type=int, default=15,
                    help='number of epochs to decay the learning rate')  # 每经过 lr_decay_step 个epoch就衰减一次学习率
parser.add_argument("--use_InfoNCE_loss", action='store_true', default=False,
                    help='select whether to use InfoNCE as the loss function')
parser.add_argument("--temperature", type=float, default=0.1,
                    help='float 0-1, One of the parameters of the loss function——InfoNCE')
parser.add_argument("--reduction", type=str, default='mean',
                    help='str, One of the parameters of the loss function——InfoNCE')
parser.add_argument("--negative_mode", type=str, default='unpaired',
                    help='str, One of the parameters of the loss function——InfoNCE')
parser.add_argument("--cnn_type", type=str, default='VGG19',
                    help='choose the cnn backbone for VSE ( VGG19 , ResNet101 , ResNet152 )')
parser.add_argument("--use_attention_for_text", action='store_true', default=False,
                    help='choose to use GRU or self-attention for textEncoding')
parser.add_argument("--num_heads", type=int, default=3,
                    help='number of heads of multi-head attention')
parser.add_argument("--use_word2vec", action='store_true', default=False,
                    help='whether to use Gensim to initialize word embedding layer')
parser.add_argument("--model_class", type=str, default='ViT_and_BERT',
                    help='choose what kind of model to use (CNN_and_GRU , ViT_and_BERT , CNN_and_BERT)')
parser.add_argument("--max_violation_in_middle", action='store_true', default=False,
                    help='choose whether to use max_violation loss in the middle of training')

args = parser.parse_args()
