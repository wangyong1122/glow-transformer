import torch
import numpy as np
import math
import time
import copy

from collections import defaultdict
from torch.autograd import Variable
from tqdm import tqdm, trange
from utils import *
from data_loader import split_batch
from optimizer import Adam

import torch.distributed as dist


def get_learning_rate(args, i):

    return args.lr_glow


def train_glow(args, watcher, model, train, dev, save_path=None, maxsteps=None, decoding_path=None, names=None):
    # optimizer
    if args.world_size > 1:
        opt_all = [Adam(param, betas=(0.9, 0.98), eps=1e-9, weight_decay=args.weight_decay) for param in
                   model.module.trainable_parameters_glow()]
    else:
        opt_all = [Adam(param, betas=(0.9, 0.98), eps=1e-9, weight_decay=args.weight_decay) for param in
                   model.trainable_parameters_glow()]

    opt = opt_all[0]

    # if resume training
    if (args.load_from != 'none') and (args.resume):
        with torch.cuda.device(args.local_rank):  # very important.
            offset, opt_states = torch.load(args.workspace_prefix + '/models/' + args.load_from + '.pt.states',
                                            map_location=lambda storage, loc: storage.cuda())
            opt.load_state_dict(opt_states)
    else:
        offset = 0

    iters = offset
    # confirm the saving path
    if save_path is None:
        save_path = args.model_name

    # setup a watcher
    # param_to_watch = ['corpus_bleu']
    watcher.set_progress_bar(args.eval_every)
    # watcher.set_best_tracker(model, opt, save_path, args.local_rank, *param_to_watch)
    if args.tensorboard and (not args.debug):
        watcher.set_tensorboard('{}/runs/{}'.format(args.workspace_prefix, args.prefix + args.hp_str))
    dev_loss_best = math.inf
    train_iter = [iter(t) for t in train]
    while True:

        def check(every=0, k=0):
            if every <= 0:
                return False
            return iters % every == k

        # --- saving --- #
        # if check(args.save_every) and (args.local_rank == 0):  # saving only works for local-rank=0
        #     watcher.info('save (back-up) checkpoints at iter={}'.format(iters))
        #     with torch.cuda.device(args.local_rank):
        #         torch.save(watcher.best_tracker.model.state_dict(), '{}_iter={}.pt'.format(args.model_name, iters))
        #         torch.save([iters, watcher.best_tracker.opt.state_dict()],
        #                    '{}_iter={}.pt.states'.format(args.model_name, iters))

        if maxsteps is None:
            maxsteps = args.maximum_steps

        if iters > maxsteps:
            watcher.info('reach the maximum updating steps.')
            break

        # --- training  --- #
        iters += 1
        # model.train()

        info_str = 'training step = {}, lr={:.7f}, '.format(iters, opt.param_groups[0]['lr'])
        info = defaultdict(lambda: [])

        with Timer() as train_timer:

            opt.param_groups[0]['lr'] = get_learning_rate(args, iters)  # (args.model == 'AutoTransformer2'))
            opt.zero_grad()
            model.train()
            # prepare the data
            for inter_step in range(args.inter_size):

                def sample_a_training_set(train, prob):
                    if (prob is None) or (len(prob) == 0):  # not providing probability, sample dataset uniformly.
                        prob = [1 / len(train) for _ in train]
                    train_idx = np.random.choice(np.arange(len(train)), p=prob)
                    return next(train[train_idx])

                if len(train) == 1:  # single-pair MT:
                    batch = next(train_iter[0])  # load the next batch of training data.
                else:
                    if (args.inter_size % len(train) == 0):  # systematic sampling
                        batch = next(train_iter[inter_step % len(train_iter)])
                    else:
                        batch = sample_a_training_set(train_iter, args.sample_prob)

                # --- attention visualization --- #
                if (check(args.att_plot_every, 1) and (inter_step == 0) and (args.local_rank == 0)):
                    if args.world_size > 1:
                        model.module.attention_flag = True
                    else:
                        model.attention_flag = True
                # -- search optimal paths -- #
                if ((args.order == 'random') or (args.order == 'optimal')) and (iters >= args.esteps):
                    # DIV = args.inter_size * args.sub_inter_size
                    #
                    # if args.search_with_dropout:
                    #     model.train()
                    # else:
                    #     model.eval()  # searching path should turn-off drop-out ?? (less noise.)
                    #
                    # # model.train()
                    # with torch.no_grad():
                    #     infob_ = model(batch, mode='path', dataflow=['src', 'src'], step=iters)
                    #     for t in infob_:
                    #         info[t] += [item(infob_[t])]
                    #
                    # model.train()  # open drop-out
                    #
                    # for batch_ in split_batch(batch, args.sub_inter_size):
                    #     mode = 'search_train' if args.order == 'search_optimal' else 'train'
                    #     info_ = model(batch_, mode=mode, dataflow=['src', 'src'], step=iters)
                    #
                    #     info_['loss'] = info_['loss'] / DIV
                    #     info_['loss'].backward()
                    #
                    #     pairs.append(batch.dataset.task + batch.message)
                    #     for t in info_:
                    #         info[t] += [item(info_[t])]
                    raise NotImplementedError
                else:
                    DIV = args.inter_size

                      # open drop-out
                    mode = 'train'
                    info_ = model.glow_forward(batch, mode=mode, dataflow=['src', 'src'], step=iters)

                    info_['loss'] = info_['loss'] / DIV
                    info_['loss'].backward()

                    for t in info_:
                        info[t] += [item(info_[t])]

            # multiple steps, one update
            grad_norm = opt.clip_grad_norm(args.grad_clip)
            opt.step()

            if args.distributed:  # gather information from other workers.
                gather_dict(info)

            for t in info:
                try:
                    if t == 'max_att':
                        info[t] = max(info[t])
                    else:
                        info[t] = sum(info[t])
                except TypeError:
                    continue


        info_str += '#token={}, #sentence={}, #maxtt={}, gn={:.4f}, speed={} t/s | {} | '.format(
            format(info['tokens'], 'k'), int(info['sents']), format(info['max_att'], 'm'), grad_norm,
            format(info['tokens'] / train_timer.elapsed_secs, 'k'), batch.dataset.task)

        for keyword in info:
            if keyword == 'loss':
                info_str += '{}={:.3f}, '.format(keyword, info[keyword] / args.world_size / DIV)
                if args.tensorboard and (not args.debug):
                    watcher.add_tensorboard('train_glow/{}'.format(keyword), info[keyword] / args.world_size / DIV, iters)

        # --- validation --- #
        if check(args.eval_every) and (not args.no_valid):  # and (args.local_rank == 0):

            model.eval()

            watcher.close_progress_bar()
            info_dev = defaultdict(lambda: [])

            with torch.no_grad():

                for j, dev_batch in enumerate(dev[0]):

                    info_dev_ = model.glow_forward(dev_batch, mode='train', dataflow=['src', 'src'], step=iters)
                    # gather from all workers:
                    if args.distributed:
                        gather_dict(info_dev_)
                    for t in info_dev_:
                        info_dev[t] += [info_dev_[t].item()]
                for t in info_dev:
                    info_dev[t] = np.mean(info_dev[t])

                if info_dev['loss'] <=dev_loss_best:

                    dev_loss_best = info_dev['loss']
                    if args.local_rank == 0:  # saving only works for local-rank=0
                        watcher.info('save (back-up) checkpoints at iter={}'.format(iters))
                        # with torch.cuda.device(args.local_rank):
                        #     torch.save(model.state_dict(),
                        #                '{}_iter={}.pt'.format(args.model_name, iters))
                        #     torch.save([iters, opt.state_dict()],
                        #                '{}_iter={}.pt.states'.format(args.model_name, iters))

                if args.tensorboard and (not args.debug):
                    for keyword in info_dev:
                        if keyword == 'loss':
                            watcher.add_tensorboard('dev_glow/{}'.format(keyword), info_dev[keyword] / args.world_size, iters)

                outputs_data = model.nat_sampling()
            if not args.debug:
                # output the best translation for record #
                if decoding_path is not None:
                    name_suffix = 'b={}_a={}.txt'.format(args.beam_size, args.alpha)
                    name = '{}.dec.{}'.format(args.test_set, name_suffix)

                    handle = open(os.path.join(decoding_path, name), 'a')
                    for d in sorted(outputs_data):
                        d = d.replace('@@ ', '')
                        print(d, file=handle, flush=True)
                        print(d)

                    handle.close()

            watcher.info('model:' + args.prefix + args.hp_str)

            # ---set-up a new progressor---
            watcher.set_progress_bar(args.eval_every)

        watcher.step_progress_bar(info_str=info_str)

