import os
import torch
import time
import data_process
import transformer_data_process
import bert_cnn_data_process
from model import VSE
from transformer_model import transformer_VSE
from bert_cnn_model import BERT_CNN_VSE
from param import args
import logging
import tensorboard_logger as tb_logger
import numpy as np
from evaluation import r_img2cap, r_cap2img, AverageMeter, LogCollector, encode_data
import pickle
import shutil
from pprint import pprint


def main():
    """
        %(asctime)s: 打印日志的时间
        %(message)s: 打印日志信息
        level分为五级：debug info warning error critical
    """
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)  # 配置日志记录器
    tb_logger.configure(os.path.join(args.log_dir, f'GPU{args.gpu}'), flush_secs=5)  # 配置TensorBoard 日志记录器

    if args.model_class == 'CNN_and_GRU':
        # 加载词汇表
        vocab = pickle.load(open(os.path.join(args.vocab_path, 'flickr30k_vocab.pkl'), 'rb'))
        # 加载DataLoader
        train_loader, val_loader = data_process.get_train_dev_loader(
            vocab,
            args.batch_size,
            args.workers
        )
        # 构建模型
        model = VSE(
            args.embed_size,
            args.finetune,
            args.word_dim,
            args.num_layers,
            vocab,
            args.margin,
            args.max_violation,
            args.grad_clip,
            args.use_InfoNCE_loss,
            args.rnn_mean_pool,
            args.bidirection_rnn,
            args.use_word2vec,
            args.cnn_type,
            args.use_attention_for_text,
            args.num_heads
        )
    elif args.model_class == 'ViT_and_BERT':
        # 加载DataLoader
        train_loader, val_loader = transformer_data_process.get_train_dev_loader(
            args.batch_size,
            args.workers
        )
        # 构建模型
        model = transformer_VSE(
            args.embed_size,
            args.finetune,
            args.margin,
            args.max_violation,
            args.grad_clip,
            args.use_InfoNCE_loss
        )
    elif args.model_class == 'CNN_and_BERT':
        # 加载DataLoader
        train_loader, val_loader = bert_cnn_data_process.get_train_dev_loader(
            args.batch_size,
            args.workers
        )
        # 构建模型
        model = BERT_CNN_VSE(
            args.embed_size,
            args.finetune,
            args.margin,
            args.max_violation,
            args.grad_clip,
            args.use_InfoNCE_loss,
            args.cnn_type
        )
    else:
        raise ValueError('Wrong model class. It must be in (CNN_and_GRU , ViT_and_BERT , CNN_and_BERT)')

    # 训练模型
    best_rsum = 0
    for epoch in range(args.num_epochs):
        # 调整学习率
        adjust_learning_rate(
            args.lr,
            model.optimizer,
            epoch
        )
        # 每个epoch训练模型
        main_train(
            args.log_step,
            args.val_step,
            train_loader,
            val_loader,
            model,
            epoch
        )

        # 计算此次epoch的r_sum
        r_sum = validate(args.log_step, val_loader, model)

        is_best = r_sum > best_rsum  # 确定当前的 r_sum 是否是历史最佳值
        best_rsum = max(r_sum, best_rsum)
        save_checkpoint(
            {
                'epoch': epoch + 1,  # epoch从0开始
                'model': model.state_dict(),  # 保存模型参数
                'best_rsum': best_rsum,  # 目前为止最好的R_sum
                'args': args,
                'whole_iters': model.whole_iters,  # 保存迭代轮数
            },
            is_best,
            prefix=args.log_dir + '/'
        )


def adjust_learning_rate(old_lr, optimizer, epoch):
    """
        动态调整优化器的学习率
    """
    lr = old_lr * (0.5 ** (epoch // args.lr_decay_step))  # 每经过lr_decay_step个epoch，学习率将衰减为原来的一半
    # 另一种衰减方式
    # lr = old_lr
    # if epoch > (2/3) * args.num_epochs:
    #     lr = old_lr * 0.1

    for param_group in optimizer.param_groups:  # 更新每个参数组中的学习率
        param_group['lr'] = lr


def main_train(log_step, val_step, train_loader, val_loader, model, epoch):
    """
        一个epoch内训练模型
    """
    batch_time_meter = AverageMeter()
    data_time_meter = AverageMeter()

    train_logger = LogCollector()

    model.train_model()  # 设置模型为训练模式
    start_time = time.time()  # 记录初始时间
    # 训练充足再使用max_violation
    if args.max_violation_in_middle and epoch > (1 / 3) * args.num_epochs:  # 尽量在过拟合之前切换max_violation
        model.use_InfoNCE_loss = True

    for index, train_data in enumerate(train_loader):
        data_time_meter.update(time.time() - start_time)  # 将data_time_meter.val的值更新为当前批次数据加载所花费的时间
        model.logger = train_logger
        images, captions, lengths, _ = train_data  # 从collate_fn中取出当前批次的数据
        model.train(images, captions, lengths)  # 将数据送入模型中进行本批次的训练
        batch_time_meter.update(time.time() - start_time)  # 将batch_time_meter.val的值更新为当前批次数据训练所花费的时间
        start_time = time.time()

        # 打印日志到控制台
        if model.whole_iters % log_step == 0:  # 每训练log_step个batch就打印一次日志
            logging.info(
                'Epoch: [{epoch}][{index}/{length}]\t'
                'Loss: {e_log}\t'
                'Batch_time: {batch_time_meter.val:.3f} (average: {batch_time_meter.avg:.3f})\t'
                'Data_time: {data_time_meter.val:.3f} (average: {data_time_meter.avg:.3f})\t'
                .format(
                    epoch=epoch,  # 当前的epoch
                    index=index,  # 当前batch的索引
                    length=len(train_loader),  # 总共的批次数量
                    e_log=str(model.logger.meters["Loss"]),  # AverageMeter.str()返回打印信息,logger里调用AverageMeter，在model.calc_loss()中记录,而model.train()又调用了calc_loss()
                    batch_time_meter=batch_time_meter,
                    data_time_meter=data_time_meter
                )
            )

        # 记录到tensorboard
        tb_logger.log_value('batch_time_meter', batch_time_meter.val, model.whole_iters)
        tb_logger.log_value('data_time_meter', data_time_meter.val, model.whole_iters)
        model.logger.tensorboard_log(tb_logger, step=model.whole_iters)

        # 每经过val_step个批次就验证一次
        if model.whole_iters % val_step == 0:
            validate(log_step, val_loader, model)


def validate(log_step, val_loader, model):
    img_embs, cap_embs = encode_data(
        model,
        val_loader,
        log_step,
        logging.info
    )
    #  以图搜文
    r_im2cap_1, r_im2cap_5, r_im2cap_10, r_im2cap_medi, r_im2cap_mean = r_img2cap(
        img_embs,
        cap_embs
    )
    logging.info("Image to caption: %.1f, %.1f, %.1f, %.1f, %.1f" % (
        r_im2cap_1, r_im2cap_5, r_im2cap_10, r_im2cap_medi, r_im2cap_mean))
    # 以文搜图
    r_cap2im_1, r_cap2im_5, r_cap2im_10, r_cap2im_medi, r_cap2im_mean = r_cap2img(
        img_embs,
        cap_embs
    )
    logging.info("Caption to image: %.1f, %.1f, %.1f, %.1f, %.1f" % (
        r_cap2im_1, r_cap2im_5, r_cap2im_10, r_cap2im_medi, r_cap2im_mean))

    # r_sum
    r_sum = r_im2cap_1 + r_im2cap_5 + r_im2cap_10 + r_cap2im_1 + r_cap2im_5 + r_cap2im_10

    # 记录到tensorboard上
    tb_logger.log_value('r_im2cap_1', r_im2cap_1, model.whole_iters)
    tb_logger.log_value('r_im2cap_5', r_im2cap_5, model.whole_iters)
    tb_logger.log_value('r_im2cap_10', r_im2cap_10, model.whole_iters)
    tb_logger.log_value('r_im2cap_medi', r_im2cap_medi, model.whole_iters)
    tb_logger.log_value('r_im2cap_mean', r_im2cap_mean, model.whole_iters)
    tb_logger.log_value('r_cap2im_1', r_cap2im_1, model.whole_iters)
    tb_logger.log_value('r_cap2im_5', r_cap2im_5, model.whole_iters)
    tb_logger.log_value('r_cap2im_10', r_cap2im_10, model.whole_iters)
    tb_logger.log_value('r_cap2im_medi', r_cap2im_medi, model.whole_iters)
    tb_logger.log_value('r_cap2im_mean', r_cap2im_mean, model.whole_iters)
    tb_logger.log_value('r_sum', r_sum, model.whole_iters)

    return r_sum


def save_checkpoint(state, is_best, file_name='checkpoint.pth.tar', prefix=''):
    """
        保存当前模型检查点及最佳模型
    """
    torch.save(state, os.path.join(os.path.join(prefix, f'GPU{args.gpu}'), file_name))  # 保存当前状态下的模型检查点
    if is_best:  # 若当前模型为历史上的最佳模型，则将复制一份模型文件并重命名，用于记录历史上的最佳模型检查点
        shutil.copyfile(
            os.path.join(os.path.join(prefix, f'GPU{args.gpu}'), file_name),
            os.path.join(os.path.join(prefix, f'GPU{args.gpu}'), 'model_best.pth.tar')
        )


if __name__ == '__main__':
    # 固定随机种子，使生成的随机数序列都将是相同的，以确保实验的可重复性和稳定性
    torch.manual_seed(16)  # 设置cpu随机数种子
    torch.cuda.manual_seed(16)  # 设置cuda随机数种子
    np.random.seed(16)  # 设置numpy随机数种子
    # 开始训练
    print("-----------------------------------------------")
    pprint(args)
    print("-----------------------------------------------")

    main()

    print("-----------------------------------------------")
    pprint(args)
    print("-----------------------------------------------")
