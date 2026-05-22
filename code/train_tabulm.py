# TabuLM — pre-training script
# Extends train_exploratory_distributed_model.py for tabular data.
# Logs: STEM | AFSET | AFFIX | MCR | CTP losses separately.

from __future__ import print_function, division

import gc
import math
import os
import random
from datetime import datetime
from shutil import copyfile

import numpy as np
import progressbar
import psutil
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader


def time_now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def date_now():
    return datetime.now().strftime("%Y-%m-%d")

def set_random_seeds(seed=0):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def train_loop(args, rank, scaler, device, data_loader,
               model, optimizer, lr_scheduler, save_file_path,
               accumulation_steps, loop, num_loops, bar,
               total_steps, total_loss,
               stem_loss_acc, afset_loss_acc, affix_loss_acc,
               mcr_loss_acc, ctp_loss_acc,
               save_every=50):

    from tabular_data_loaders import tabulm_model_forward

    for batch_idx, data_item in enumerate(data_loader):
        if scaler is not None:
            with torch.cuda.amp.autocast():
                loss, sl, al, fxl, ml, cl = tabulm_model_forward(
                    args, data_item, model, device,
                    model.module.encoder.tot_num_affixes if hasattr(model, 'module')
                    else model.encoder.tot_num_affixes,
                )
                loss = loss / accumulation_steps
            scaler.scale(loss).backward()
        else:
            loss, sl, al, fxl, ml, cl = tabulm_model_forward(
                args, data_item, model, device,
                model.module.encoder.tot_num_affixes if hasattr(model, 'module')
                else model.encoder.tot_num_affixes,
            )
            loss = loss / accumulation_steps
            loss.backward()

        total_loss      += loss.item()
        stem_loss_acc   += sl.item()  / accumulation_steps
        afset_loss_acc  += al.item()  / accumulation_steps
        affix_loss_acc  += fxl.item() / accumulation_steps
        mcr_loss_acc    += ml.item()  / accumulation_steps
        ctp_loss_acc    += cl.item()  / accumulation_steps
        total_steps     += 1

        if (total_steps % accumulation_steps) == 0:
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()

            if rank == 0:
                print(
                    time_now(),
                    f'Loop:{loop}/{num_loops}',
                    f'Batch:{batch_idx+1}/{len(data_loader)}',
                    f'TOTAL:{total_loss:.4f}',
                    f'STEM:{stem_loss_acc:.4f}',
                    f'AFSET:{afset_loss_acc:.4f}',
                    f'AFFIX:{affix_loss_acc:.4f}',
                    f'MCR:{mcr_loss_acc:.4f}',
                    f'CTP:{ctp_loss_acc:.4f}',
                    f'LR:{lr_scheduler.get_lr():.8f}',
                    f'iter:{lr_scheduler.num_iters}',
                )
                bar.update(lr_scheduler.num_iters)

            total_loss = stem_loss_acc = afset_loss_acc = 0.0
            affix_loss_acc = mcr_loss_acc = ctp_loss_acc = 0.0

    if rank == 0 and (((loop + 1) % save_every) == 0 or loop == num_loops - 1):
        if os.path.exists(save_file_path):
            copyfile(save_file_path, save_file_path + '_prev_checkpoint.pt')

        _model = model.module if hasattr(model, 'module') else model
        _model.eval()
        torch.save({
            'iter': total_steps,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler_state_dict': lr_scheduler.state_dict(),
            'loop': loop,
            'num_loops': num_loops,
        }, save_file_path)
        _model.train()

    return (total_steps, total_loss,
            stem_loss_acc, afset_loss_acc, affix_loss_acc,
            mcr_loss_acc, ctp_loss_acc)


def train_fn(rank, args):
    import youtokentome as yttm
    from morpho_learning_rates import AnnealingLR
    from morpho_data_loaders import KBVocab, AffixSetVocab
    from tabular_data_loaders import TabularKBCorpusDataset, tabular_collate_wrapper
    from tabulm_model import tabulm_base

    USE_GPU = args.gpus > 0 and torch.cuda.is_available()

    device = torch.device('cuda' if USE_GPU else 'cpu')

    if USE_GPU:
        dist.init_process_group('nccl', init_method='env://',
                                world_size=args.world_size, rank=rank)
        torch.cuda.set_device(rank)
        scaler = torch.cuda.amp.GradScaler()
    else:
        dist.init_process_group('gloo', init_method='env://',
                                world_size=args.world_size, rank=rank)
        scaler = None

    home = args.home_path

    bpe = yttm.BPE(model=home + 'data/BPE-30k.mdl')

    kb_vocab = KBVocab()
    kb_vocab.load_state_dict(torch.load(home + 'data/kb_vocab_state_dict_2021-02-07.pt'))

    affix_set_vocab = None
    if args.use_afsets:
        affix_set_vocab = AffixSetVocab(
            reduced_affix_dict_file=home + f'data/reduced_affix_dict_{args.afset_dict_size}.csv',
            reduced_affix_dict_map_file=home + f'data/reduced_affix_dict_map_{args.afset_dict_size}.csv',
        )

    morpho_rel_pos_dict = None
    morpho_rel_pos_dmax = 5
    if args.use_pos_aware_rel_pos_bias:
        rel_pos_file = home + 'data/morpho_rel_pos_dict_2021-03-24.pt'
        if os.path.exists(rel_pos_file):
            saved = torch.load(rel_pos_file)
            morpho_rel_pos_dict = saved['morpho_rel_pos_dict']
            morpho_rel_pos_dmax = saved['morpho_rel_pos_dmax']
        else:
            print(f'[WARN] morpho_rel_pos_dict not found, disabling pos_aware_rel_pos_bias')
            args.use_pos_aware_rel_pos_bias = False
            args.use_pos_aware_rel = False

    num_iters   = args.num_iters
    warmup_iter = args.warmup_iter
    peak_lr     = args.peak_lr
    wd          = args.wd

    if rank == 0:
        print(time_now(), 'Building TabuLM model ...')

    model = tabulm_base(kb_vocab, affix_set_vocab, morpho_rel_pos_dict,
                        device, args, saved_model_file=args.exploratory_model_load)

    if USE_GPU:
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)
        try:
            import apex
            optimizer = apex.optimizers.FusedLAMB(
                model.parameters(), lr=peak_lr, betas=(0.9, 0.98),
                eps=1e-06, weight_decay=wd,
            )
        except ImportError:
            from lamb import Lamb
            optimizer = Lamb(model.parameters(), lr=peak_lr, betas=(0.9, 0.98),
                             eps=1e-06, weight_decay=wd)
    else:
        from lamb import Lamb
        model = DDP(model, device_ids=[])
        optimizer = Lamb(model.parameters(), lr=peak_lr, betas=(0.9, 0.98),
                         eps=1e-06, weight_decay=wd)

    lr_scheduler = AnnealingLR(optimizer,
                               start_lr=peak_lr,
                               warmup_iter=warmup_iter,
                               num_iters=num_iters,
                               decay_style='linear',
                               last_iter=0)

    # ── Resume from checkpoint if provided ────────────────────────────────────
    resume_file = getattr(args, 'resume_checkpoint', None)
    curr_loops = 0
    total_steps = 0
    total_loss  = stem_loss_acc = afset_loss_acc = 0.0
    affix_loss_acc = mcr_loss_acc = ctp_loss_acc = 0.0

    if resume_file and os.path.exists(resume_file):
        if rank == 0:
            print(f'[RESUME] Loading checkpoint from {resume_file}')
        ckpt = torch.load(resume_file, map_location=device)
        # Strip DDP 'module.' prefix if present
        state = ckpt['model_state_dict']
        if all(k.startswith('module.') for k in state):
            state = {k[len('module.'):]: v for k, v in state.items()}
        _model = model.module if hasattr(model, 'module') else model
        _model.load_state_dict(state, strict=False)
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        lr_scheduler.load_state_dict(ckpt['lr_scheduler_state_dict'])
        curr_loops  = ckpt.get('loop', 0) + 1
        total_steps = ckpt.get('iter', 0)
        if rank == 0:
            print(f'[RESUME] Resuming from loop {curr_loops}, iter {total_steps}')

    csv_dir = args.tabulm_csv_dir if hasattr(args, 'tabulm_csv_dir') and args.tabulm_csv_dir \
        else home + 'data/tables/'

    num_train_loops = math.ceil(
        num_iters * args.accumulation_steps / args.number_of_load_batches
    )

    save_path = (
        home + f'data/tabulm_model_{date_now()}'
        f'_pos@{args.num_pos_m_embeddings}'
        f'_stem@{args.num_stem_m_embeddings}'
        f'_afsets@{args.use_afsets}@{args.afset_dict_size}'
        f'{getattr(args, "ablation_tag", "")}.pt'
    )

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if rank == 0:
        print('─' * 60)
        print(f'Total params: {total_params:,}  Trainable: {trainable_params:,}')
        print(f'Saving to: {save_path}')
        print(f'CSV tables from: {csv_dir}')
        print(f'num_iters={num_iters}  warmup={warmup_iter}  loops={num_train_loops}')
        print(f'batch_size={args.batch_size}  accum={args.accumulation_steps}')
        print(f'peak_lr={peak_lr}  wd={wd}')
        print('─' * 60)

    model.train()
    model.zero_grad()

    with progressbar.ProgressBar(
        initial_value=lr_scheduler.num_iters,
        max_value=lr_scheduler.end_iter,
        redirect_stdout=True,
    ) as bar:
        if rank == 0:
            bar.update(lr_scheduler.num_iters)

        for loop in range(curr_loops, num_train_loops):
            if rank == 0:
                print(time_now(), 'Loading tabular dataset ...')

            dataset = TabularKBCorpusDataset(
                args, kb_vocab, affix_set_vocab, bpe,
                csv_dir=csv_dir,
                max_batch_items=args.number_of_load_batches * args.batch_size,
                max_seq_len=512,
                rank=rank,
            )

            data_loader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                collate_fn=tabular_collate_wrapper,
                shuffle=True,
            )

            if rank == 0:
                print(time_now(), 'Memory:', psutil.virtual_memory())

            (total_steps, total_loss,
             stem_loss_acc, afset_loss_acc, affix_loss_acc,
             mcr_loss_acc, ctp_loss_acc) = train_loop(
                args, rank, scaler, device, data_loader,
                model, optimizer, lr_scheduler, save_path,
                args.accumulation_steps, loop, num_train_loops, bar,
                total_steps, total_loss,
                stem_loss_acc, afset_loss_acc, affix_loss_acc,
                mcr_loss_acc, ctp_loss_acc,
                save_every=getattr(args, 'save_every', 50),
            )

            if rank == 0:
                print(time_now(), f'{loop+1}/{num_train_loops} loops complete')

            del data_loader, dataset
            gc.collect()


def main():
    import argparse
    from morpho_common import setup_common_args

    # Pull out --resume-checkpoint and ablation flags before setup_common_args sees sys.argv
    import sys
    resume_checkpoint = None
    no_mcr = False
    no_ctp = False
    no_tabular_emb = False
    no_bias = False
    ablation_tag = ''
    filtered = []
    i = 0
    while i < len(sys.argv[1:]):
        arg = sys.argv[1:][i]
        if arg == '--resume-checkpoint':
            resume_checkpoint = sys.argv[1:][i + 1]
            i += 2
        elif arg.startswith('--resume-checkpoint='):
            resume_checkpoint = arg.split('=', 1)[1]
            i += 1
        elif arg == '--no-mcr':
            no_mcr = True
            ablation_tag += '_noMCR'
            i += 1
        elif arg == '--no-ctp':
            no_ctp = True
            ablation_tag += '_noCTP'
            i += 1
        elif arg == '--no-tabular-emb':
            no_tabular_emb = True
            ablation_tag += '_noTabEmb'
            i += 1
        elif arg == '--no-bias':
            no_bias = True
            ablation_tag += '_noBias'
            i += 1
        else:
            filtered.append(arg)
            i += 1
    sys.argv = [sys.argv[0]] + filtered

    args = setup_common_args()
    args.resume_checkpoint = resume_checkpoint
    args.no_mcr = no_mcr
    args.no_ctp = no_ctp
    args.no_tabular_emb = no_tabular_emb
    args.no_bias = no_bias
    args.ablation_tag = ablation_tag

    # Extra args not in morpho_common.setup_common_args
    if not hasattr(args, 'tabulm_csv_dir'):
        args.tabulm_csv_dir = os.environ.get('TABULM_CSV_DIR', None)
    if not hasattr(args, 'resume_checkpoint'):
        args.resume_checkpoint = None

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '29602')

    if args.gpus == 0:
        args.world_size = 1

    mp.spawn(train_fn, nprocs=args.world_size, args=(args,))


if __name__ == '__main__':
    main()
