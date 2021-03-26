"""
Training script. Should be pretty adaptable to whatever.
"""
# -*- coding: UTF-8 -*-
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
# f = os.popen("python train.py -params multiatt/default.json -folder saves/flagship_answer")
import sys

PYTHON_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PYTHON_PATH)
import argparse
import shutil

import multiprocessing
import numpy as np
import pandas as pd
import torch
from allennlp.common.params import Params
from allennlp.training.learning_rate_schedulers import LearningRateScheduler
from allennlp.training.optimizers import Optimizer
from torch.nn import DataParallel
from tqdm import tqdm
from torch.utils.data import DataLoader

from dataloaders.vcr_attribute import VCR, VCRLoader
from utils.pytorch_misc import time_batch, save_checkpoint, clip_grad_norm, \
    restore_checkpoint, print_para, restore_best_checkpoint

import logging
from tensorboardX import SummaryWriter
import json

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', level=logging.DEBUG)

# This is needed to make the imports work
from allennlp.models import Model
import models

torch.backends.cudnn.enabled = False
torch.set_printoptions(threshold=500000000, linewidth=8000000)
#################################
#################################
######## Data loading stuff
#################################
#################################

parser = argparse.ArgumentParser(description='train')
parser.add_argument(
    '-params',
    dest='params',
    help='Params location',
    type=str,
)
parser.add_argument(
    '-rationale',
    action="store_true",
    help='use rationale',
)
parser.add_argument(
    '-output',
    type=str
)
parser.add_argument(
    '-no_tqdm',
    dest='no_tqdm',
    action='store_true',
)
parser.add_argument(
    '-batch_size',
    dest='batch_size',
    type=int,
    default=96
)
parser.add_argument(
    '-records',
    type=str,
    default='records.json'
)
parser.add_argument(
    '-describe',
    type=str,
    default=''
)

args = parser.parse_args()

seed = 1111
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
batch_size = args.batch_size
only_use_relevant_dets = False
# args.rationale = True

# args.params = 'models/multiatt/default2.json'
folder = f'saves/{args.output}'
writer = SummaryWriter(f'/home/share/wangkejie/vcr1/runs/{args.output}')
params = Params.from_file(args.params)
train, val, test = VCR.splits(embs_to_load=params['dataset_reader'].get('embs', 'bert_da'),
                              only_use_relevant_dets=params['dataset_reader'].get('only_use_relevant_dets',
                                                                                  only_use_relevant_dets))  ########################这个地方我改成了false，使用全部的box
# NUM_GPUS = torch.cuda.device_count()  # NUM_GPUS = 4
NUM_GPUS = 1
NUM_CPUS = multiprocessing.cpu_count()  # NUM_CPUS = 32
if NUM_GPUS == 0:
    raise ValueError("you need gpus!")


def _to_gpu(td):
    if NUM_GPUS > 1:
        return td
    for k in td:
        if k != 'metadata':
            td[k] = {k2: v.cuda(non_blocking=True) for k2, v in td[k].items()} if isinstance(td[k], dict) else td[
                k].cuda(
                non_blocking=True)
    return td


# num_workers = (8 * NUM_GPUS if NUM_CPUS == 32 else 2 * NUM_GPUS)
num_workers = 8
print(f"Using {num_workers} workers out of {NUM_CPUS} possible", flush=True)
loader_params = {'batch_size': batch_size, 'num_workers': num_workers, "pin_memory": True}
# train_loader = DataLoader(train, shuffle=True, collate_fn=collate_fn, drop_last=True, batch_size=batch_size, num_workers=num_workers)
# val_loader = DataLoader(val, shuffle=False, collate_fn=collate_fn, batch_size=batch_size, num_workers=num_workers)
# test_loader = DataLoader(test, shuffle=False, collate_fn=collate_fn, batch_size=batch_size, num_workers=num_workers)
train_loader = VCRLoader.from_dataset(train, **loader_params)
val_loader = VCRLoader.from_dataset(val, **loader_params)
test_loader = VCRLoader.from_dataset(test, **loader_params)

# train_loader = CudaDataLoader(train_loader, device='cuda', queue_size=4)
# val_loader = CudaDataLoader(val_loader, device='cuda', queue_size=4)
# test_loader = CudaDataLoader(test_loader, device='cuda', queue_size=4)

ARGS_RESET_EVERY = 600
print("Loading {} for {}".format(params['model'].get('type', 'WTF?'), 'rationales' if args.rationale else 'answer'),
      flush=True)
model = Model.from_params(vocab=train.vocab, params=params['model'])

model = DataParallel(model).cuda() if NUM_GPUS > 1 else model.cuda()
optimizer = Optimizer.from_params(model_parameters=[x for x in model.named_parameters() if x[1].requires_grad],
                                  params=params['trainer']['optimizer'])

lr_scheduler_params = params['trainer'].pop("learning_rate_scheduler", None)
scheduler = LearningRateScheduler.from_params(optimizer=optimizer,
                                              params=lr_scheduler_params) if lr_scheduler_params else None

if os.path.exists(folder):
    print("Found folder! restoring", flush=True)
    start_epoch, val_metric_per_epoch = restore_checkpoint(model, optimizer, serialization_dir=folder,
                                                           learning_rate_scheduler=scheduler)
    # start_epoch, val_metric_per_epoch = 0, []
    print(start_epoch)
    print(val_metric_per_epoch)
else:
    print("Making directories")
    os.makedirs(folder, exist_ok=True)
    start_epoch, val_metric_per_epoch = 0, []
    shutil.copy2(args.params, folder)

with open(os.path.join(folder, 'describe.txt'), 'a') as fp:
    fp.write(args.describe)
    fp.write('\n--------------------------\n')

logger = open(f'saves/{args.output}/log.txt', mode='a', encoding='utf8')

# store best performance of all models in a file

param_shapes = print_para(model)
num_batches = 0
for epoch_num in range(start_epoch, params['trainer']['num_epochs'] + start_epoch + 10):
    train_results = []
    norms = []
    model.train()
    for b, (time_per_batch, batch) in enumerate(
            time_batch(train_loader if args.no_tqdm else tqdm(train_loader, ncols=80), reset_every=ARGS_RESET_EVERY)):
        batch = _to_gpu(batch)
        optimizer.zero_grad()
        output_dict = model(**batch)
        loss = output_dict['loss'].mean()
        loss.backward()

        num_batches += 1
        if scheduler:
            scheduler.step_batch(num_batches)

        norms.append(
            clip_grad_norm(model.named_parameters(), max_norm=params['trainer']['grad_norm'], clip=True, verbose=False)
        )
        optimizer.step()

        train_results.append(pd.Series({'loss': output_dict['loss'].mean().item(),
                                        **(model.module if NUM_GPUS > 1 else model).get_metrics(
                                            reset=(b % ARGS_RESET_EVERY) == 0),
                                        'sec_per_batch': time_per_batch,
                                        'hr_per_epoch': len(train_loader) * time_per_batch / 3600,
                                        }))
        if b % ARGS_RESET_EVERY == 0 and b > 0:
            norms_df = pd.DataFrame(pd.DataFrame(norms[-ARGS_RESET_EVERY:]).mean(), columns=['norm']).join(
                param_shapes[['shape', 'size']]).sort_values('norm', ascending=False)

            print("e{:2d}b{:5d}/{:5d}. norms: \n{}\nsumm:\n{}\n~~~~~~~~~~~~~~~~~~\n".format(
                epoch_num, b, len(train_loader),
                norms_df.to_string(formatters={'norm': '{:.7f}'.format}),
                pd.DataFrame(train_results[-ARGS_RESET_EVERY:]).mean(),
            ), flush=True)
            logger.write("e{:2d}b{:5d}/{:5d}. norms: \n{}\nsumm:\n{}\n~~~~~~~~~~~~~~~~~~\n".format(
                epoch_num, b, len(train_loader),
                norms_df.to_string(formatters={'norm': '{:.7f}'.format}),
                pd.DataFrame(train_results[-ARGS_RESET_EVERY:]).mean(),
            ))
            writer.add_scalar('training_loss', pd.DataFrame(train_results[-ARGS_RESET_EVERY:]).mean()['loss'],
                              global_step=num_batches)
            writer.add_scalar('training_accuracy', pd.DataFrame(train_results[-ARGS_RESET_EVERY:]).mean()['accuracy'],
                              global_step=num_batches)

    print("---\nTRAIN EPOCH {:2d}:\n{}\n----".format(epoch_num, pd.DataFrame(train_results).mean()))
    logger.write("---\nTRAIN EPOCH {:2d}:\n{}\n----".format(epoch_num, pd.DataFrame(train_results).mean()))
    val_probs = []
    val_labels = []
    val_loss_sum = 0.0

    q_att1 = []
    a_att1 = []
    q_att2 = []
    a_att2 = []

    model.eval()
    for b, (time_per_batch, batch) in enumerate(time_batch(val_loader)):
        with torch.no_grad():
            batch = _to_gpu(batch)
            output_dict = model(**batch)
            val_probs.append(output_dict['label_probs'].detach().cpu().numpy())
            val_labels.append(batch['label'].detach().cpu().numpy())
            val_loss_sum += output_dict['loss'].mean().item() * batch['label'].shape[0]

            q_att1.append(output_dict['q_att1'])
            a_att1.append(output_dict['a_att1'])
            q_att2.append(output_dict['q_att2'])
            a_att2.append(output_dict['a_att2'])

    val_labels = np.concatenate(val_labels, 0)
    val_probs = np.concatenate(val_probs, 0)
    val_loss_avg = val_loss_sum / val_labels.shape[0]

    val_metric_per_epoch.append(float(np.mean(val_labels == val_probs.argmax(1))))
    if scheduler:
        scheduler.step(val_metric_per_epoch[-1])

    print("Val epoch {} has acc {:.4f} and loss {:.4f}".format(epoch_num, val_metric_per_epoch[-1], val_loss_avg),
          flush=True)
    logger.write(
        "Val epoch {} has acc {:.4f} and loss {:.4f}".format(epoch_num, val_metric_per_epoch[-1], val_loss_avg))
    if int(np.argmax(val_metric_per_epoch)) < (len(val_metric_per_epoch) - 1 - params['trainer']['patience']):
        print("Stopping at epoch {:2d}".format(epoch_num))
        logger.write("Stopping at epoch {:2d}".format(epoch_num))
        break
    save_checkpoint(model, optimizer, folder, epoch_num, val_metric_per_epoch,
                    is_best=int(np.argmax(val_metric_per_epoch)) == (len(val_metric_per_epoch) - 1),
                     q_att1=q_att1, a_att1=a_att1, q_att2=q_att2, a_att2=a_att2)
    writer.add_scalar('val_loss', val_loss_avg, global_step=epoch_num)
    writer.add_scalar('val_accuracy', val_metric_per_epoch[-1], global_step=epoch_num)

print("STOPPING. now running the best model on the validation set", flush=True)
logger.write("STOPPING. now running the best model on the validation set")
# Load best
restore_best_checkpoint(model, folder)
model.eval()
val_probs = []
val_labels = []
for b, (time_per_batch, batch) in enumerate(time_batch(val_loader)):
    with torch.no_grad():
        batch = _to_gpu(batch)
        output_dict = model(**batch)
        val_probs.append(output_dict['label_probs'].detach().cpu().numpy())
        val_labels.append(batch['label'].detach().cpu().numpy())
val_labels = np.concatenate(val_labels, 0)
val_probs = np.concatenate(val_probs, 0)
acc = float(np.mean(val_labels == val_probs.argmax(1)))
print("Final val accuracy is {:.4f}".format(acc))
logger.write("Final val accuracy is {:.4f}".format(acc))
np.save(os.path.join(folder, f'valpreds.npy'), val_probs)
