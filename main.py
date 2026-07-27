import os
import torch
import random
import argparse
import numpy as np

from stenet import STENet
from train import Trainer
from utils import Report
from data import get_dataset
from config import Config


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def resolve_resume_plan(config):
    finetuning = bool(getattr(config, 'finetuning', False))
    resume_mode = getattr(config, 'resume_mode', None)
    resume_epoch = getattr(config, 'resume_epoch', None)

    if not finetuning:
        return 'off', None

    if resume_mode is None:
        return 'off', None

    resume_mode = str(resume_mode).lower()
    if resume_mode == 'off':
        return 'off', None
    if resume_mode == 'latest':
        return 'latest', None
    if resume_mode == 'epoch':
        if resume_epoch is None or int(resume_epoch) <= 0:
            raise ValueError(f'resume_epoch must be positive when resume_mode=epoch, got {resume_epoch}')
        return 'epoch', int(resume_epoch)
    raise ValueError(f'unsupported resume_mode: {resume_mode}')


def train(config):
    global_step = 0
    train_log = Report(config.save_dir, type='train', stage=config.stage)
    val_log = Report(config.save_dir, type='val', stage=config.stage)

    train_dataloader = get_dataset(config, type='train')
    valid_dataloader = get_dataset(config, type='val')

    model = STENet(config=config)
    trainer = Trainer(config=config, model=model)

    print(f'num parameters: {count_parameters(model)}')

    # if config.stage == 2:
    trainer.load_best_stage1_model()

    best_psnr = 0
    last_epoch = 0
    resume_mode, resume_epoch = resolve_resume_plan(config)
    print(f'resume plan: mode={resume_mode}, epoch={resume_epoch}')
    if resume_mode == 'latest':
        last_epoch = trainer.load_checkpoint()
    elif resume_mode == 'epoch':
        last_epoch = trainer.load_checkpoint(epoch=resume_epoch)

    for epoch in range(last_epoch, config.num_epochs):
        train_log.write(f'========= Epoch {epoch+1} of {config.num_epochs} =========')
        global_step = trainer.train(train_dataloader, train_log, global_step)

        if (epoch + 1) % config.val_period == 0 or epoch == config.num_epochs - 1:
            psnr = trainer.validate(valid_dataloader, val_log, epoch+1)
            trainer.save_checkpoint(epoch + 1)
            if psnr > best_psnr:
                best_psnr = psnr
                trainer.save_best_model(epoch + 1)


def test(config):
    test_dataloader = get_dataset(config, type='test')
    model = STENet(config=config)
    trainer = Trainer(config=config, model=model)
    trainer.load_best_model()
    trainer.test(test_dataloader)
    trainer.test_quantitative_result(gt_dir=os.path.join(config.dataset_path, 'val_sharp/val/val_sharp'),
                                     output_dir=os.path.join(config.save_dir, 'test'), image_border=config.num_seq//2)


def test_custom(config):
    from data import Custom_Dataset

    data = Custom_Dataset(config)
    test_dataloader = torch.utils.data.DataLoader(data, batch_size=1, drop_last=False, shuffle=False, num_workers=int(config.nThreads), pin_memory=True)
    model = STENet(config=config)
    trainer = Trainer(config=config, model=model)
    trainer.load_best_model()
    trainer.test(test_dataloader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='train STENet on REDS')
    parser.add_argument('--test', action='store_true', help='test STENet on REDS4')
    parser.add_argument('--test_custom', action='store_true', help='test STENet on custom dataset')
    parser.add_argument('--config_path', type=str, default='./experiment.cfg', help='path to config file with hyperparameters, etc.')
    args = parser.parse_args()

    config = Config(args.config_path)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu

    if args.train:
        train(config)

    if args.test:
        test(config)

    if args.test_custom:
        test_custom(config)
