from dataset import BCIIV2a_data_gdf
from dataset import OpenBMI_data_mat
from model.NeuroGuideNet import NeuroGuideNet
from utils import StepLR2, ConfusionMatrix, center_loss, FocalLoss, CrossEntropyLoss_CosineSim, move2device
from transform import SegmentReassemble, Compose, RandomCrop, RandomFlipBCIIV2a, RandomErasing, CutMix, \
    RepeatedTrialAugmentation, AddNoise, RandomFlipOpenBMI
from utils import StandardScaler, EmbeddingRunningMean, CE_Label_Smooth_Loss, CenterLoss

from torchinfo import summary
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import torch.nn.functional as F
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import time
import random
from glob import glob
import pickle
import gc


def set_seed(seed_n):
    print('seed is ' + str(seed_n))
    g = torch.Generator()
    g.manual_seed(seed_n)
    random.seed(seed_n)
    np.random.seed(seed_n)
    torch.manual_seed(seed_n)
    torch.cuda.manual_seed(seed_n)
    torch.cuda.manual_seed_all(seed_n)
    os.environ['PYTHONHASHSEED'] = str(seed_n)


def get_dataset(cfg, subject):
    if cfg['dataset']['name'] == 'BCIIV2a':
        path_T = glob(os.path.join(cfg['dataset']['root'], '*T.gdf'))
        path_E = glob(os.path.join(cfg['dataset']['root'], '*E.gdf'))
        train_path = [i for i in path_T if str(int(i.split(f'{os.sep}A')[-1].split('T.')[0])) in subject]
        test_path = [i for i in path_E if str(int(i.split(f'{os.sep}A')[-1].split('E.')[0])) in subject]
        train_data_dict = BCIIV2a_data_gdf.read_gdf_data(
            cfg['dataset']['root'],
            train_path,
            filter_order=cfg['dataset']['filter_order'],
            data_len=cfg['dataset']['data_len'],
            window_size=cfg['dataset']['window_size'],
            stride=cfg['dataset']['window_stride'],
            fs=cfg['dataset']['fs'],
            f_start=cfg['dataset']['filter_start'],
            f_end=cfg['dataset']['filter_end'],
            factor_new=cfg['dataset']['factor_new']
        )
        test_data_dict = BCIIV2a_data_gdf.read_gdf_data(
            cfg['dataset']['root'],
            test_path,
            filter_order=cfg['dataset']['filter_order'],
            data_len=cfg['dataset']['data_len'],
            window_size=cfg['dataset']['window_size'],
            stride=cfg['dataset']['window_stride'],
            fs=cfg['dataset']['fs'],
            f_start=cfg['dataset']['filter_start'],
            f_end=cfg['dataset']['filter_end'],
            mat_label='true_labels/',
            factor_new=cfg['dataset']['factor_new']
        )
        train_ds = BCIIV2a_data_gdf.BCIIV2ADataset(train_data_dict)
        test_ds = BCIIV2a_data_gdf.BCIIV2ADataset(test_data_dict)
    elif cfg['dataset']['name'] == 'OpenBMI':
        train_data_dict = OpenBMI_data_mat.read_mat_data(
            root=cfg['dataset']['root'],
            subject_select=subject,
            session=[1, 2],
            filter_order=cfg['dataset']['filter_order'],
            data_len=cfg['dataset']['data_len'],
            fs=cfg['dataset']['fs'],
            f_start=cfg['dataset']['filter_start'],
            f_end=cfg['dataset']['filter_end'],
            factor_new=cfg['dataset']['factor_new'],
            is_train=True
        )
        test_data_dict = OpenBMI_data_mat.read_mat_data(
            root=cfg['dataset']['root'],
            subject_select=subject,
            session=[1, 2],
            filter_order=cfg['dataset']['filter_order'],
            data_len=cfg['dataset']['data_len'],
            fs=cfg['dataset']['fs'],
            f_start=cfg['dataset']['filter_start'],
            f_end=cfg['dataset']['filter_end'],
            factor_new=cfg['dataset']['factor_new'],
            is_train=False
        )

        train_ds = OpenBMI_data_mat.OpenBMIDataset(train_data_dict)
        test_ds = OpenBMI_data_mat.OpenBMIDataset(test_data_dict)
    else:
        raise 'Unsupported dataset'
    return train_ds, test_ds


def train_one_epoch(model, data_loader, criterion, kl_criterion, optimizer, rta, max_grad_norm,
                    use_R_Drop, R_Drop_alpha, use_center_loss, center_loss_lambda, device,
                    num_classes, label_name):
    model.train()
    total_loss = 0
    running_loss_dict = {'mi': 0, 'R_Drop_kl': 0, 'center': 0}
    loss_cnt = 0
    total_acc = 0
    acc_cnt = 0
    dl = tqdm(data_loader)

    confusion = ConfusionMatrix(num_classes=num_classes, labels=label_name)
    for idx, batch in enumerate(dl):
        optimizer.zero_grad()

        x, y, extras = batch

        x = x.to(device)
        y = y.to(device)
        # extras = extras.to(device)

        batch_size = x.shape[0]

        x, y = rta(x, y)

        if use_R_Drop:
            x = x.repeat(2, 1, 1, 1)
            y = y.repeat(2)

        y_pred, output_dict = model(x)

        mi_loss = criterion(y_pred, y)
        running_loss_dict['mi'] += mi_loss.item()

        loss = mi_loss

        if use_center_loss:
            center_loss = model.get_center_loss(output_dict['mi_emb'], y)
            loss = loss + center_loss * center_loss_lambda
            running_loss_dict['center'] += center_loss.item()

        if use_R_Drop:
            y_pred_temp = y_pred.reshape(2, batch_size, -1)
            kl_loss = kl_criterion(F.log_softmax(y_pred_temp[0], dim=-1),
                                   F.softmax(y_pred_temp[1], dim=-1)) + \
                      kl_criterion(F.log_softmax(y_pred_temp[1], dim=-1),
                                   F.softmax(y_pred_temp[0], dim=-1))

            running_loss_dict['R_Drop_kl'] += kl_loss.item()
            loss = loss + kl_loss * R_Drop_alpha * 0.5

        loss.backward()
        clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        y_pred_cls = torch.argmax(y_pred, dim=-1)
        total_loss += loss.item()
        loss_cnt += 1
        total_acc += torch.sum(y_pred_cls == y.data).item()
        acc_cnt += x.shape[0]
        confusion.update(y_pred_cls.cpu().detach().numpy(), y.cpu().detach().numpy())

        dl.set_postfix_str('Loss: {:.2f}'.format(loss.item()))

    return (total_loss / loss_cnt, total_acc / acc_cnt * 100, confusion.summary(),
            {k: running_loss_dict[k] / loss_cnt for k in running_loss_dict})


def evaluate_model(model, data_loader, criterion, use_center_loss, center_loss_lambda,
                   device, num_classes, label_name, log_dir,
                   window_size=None, window_stride=None,
                   plot_matrix=False, eval_name=None,
                   ):
    model.eval()
    total_loss = 0
    running_loss_dict = {'mi': 0, 'center': 0}
    loss_cnt = 0
    total_acc = 0
    acc_cnt = 0
    dl = tqdm(data_loader)

    confusion = ConfusionMatrix(num_classes=num_classes, labels=label_name)
    for idx, batch in enumerate(dl):
        x, y, extras = batch
        # x, y = batch

        # x = x.to(device)
        y = y.to(device)

        with torch.no_grad():
            x_i = 0
            y_pred_list = []
            while x_i + window_size <= x.shape[2]:
                y_pred, output_dict = model(x[:, :, x_i: x_i + window_size].to(device))

                mi_loss = criterion(y_pred, y)

                loss = mi_loss
                running_loss_dict['mi'] += mi_loss.item()

                if use_center_loss:
                    center_loss = model.get_center_loss(output_dict['mi_emb'], y)
                    loss = loss + center_loss * center_loss_lambda
                    running_loss_dict['center'] += center_loss.item()
                x_i += window_stride

                total_loss += loss.item()
                loss_cnt += 1

                y_pred_cls = torch.argmax(y_pred, dim=-1)
                total_acc += torch.sum(y_pred_cls == y.data).item()
                acc_cnt += x.shape[0]
                y_pred_list.append(y_pred)

                dl.set_postfix_str('Loss: {:.2f}'.format(loss.item()))

            y_pred = torch.stack(y_pred_list, dim=1)
            y_pred_cls = torch.argmax(y_pred, dim=-1)
            y_pred_cls = torch.stack([torch.argmax(torch.bincount(cls)) for cls in y_pred_cls], dim=0)
            confusion.update(y_pred_cls.cpu().detach().numpy(), y.cpu().detach().numpy())
    if plot_matrix:
        if eval_name is None:
            eval_name = 'unknown'
        confusion.plot(os.path.join(log_dir, eval_name + '.png'))
    return (total_loss / loss_cnt, total_acc / acc_cnt * 100,
            confusion.summary(), {k: running_loss_dict[k] / loss_cnt for k in running_loss_dict})


def train(cfg, logger, log_dir, subject=None, state_dict=None, state_file=None,
          repeat=5):
    set_seed(42)

    device = cfg['device']
    os.makedirs(log_dir, exist_ok=True)

    if cfg['dataset']['name'] == 'BCIIV2a':
        random_flip = RandomFlipBCIIV2a(cfg['dataset']['label_name'],
                                        p1=cfg['dataset']['random_flip_p1'])
    elif cfg['dataset']['name'] == 'ULMD':
        random_flip = None
    elif cfg['dataset']['name'] == 'OpenBMI':
        random_flip = RandomFlipOpenBMI(cfg['dataset']['label_name'],
                                        p1=cfg['dataset']['random_flip_p1'])
    else:
        raise 'Unsupported dataset'

    transforms = Compose([
        random_flip,
        SegmentReassemble(cfg['dataset']['segment_reassemble_p1'],
                          cfg['dataset']['segment_reassemble_split_num']),
    ])
    repeat_trail_argument = RepeatedTrialAugmentation(transforms, m=cfg['dataset']['rta_num'])

    # 5-fold
    if state_dict is not None:
        if state_dict['result_df'] is None:
            result_df = {'fold': [],
                         'train loss': [], 'train acc': [], 'train kappa': [],
                         'test loss': [], 'test acc': [], 'test kappa': []}
            state_dict['result_df'] = result_df
        else:
            result_df = state_dict['result_df']
        loader = state_dict['subject_dict'][subject[0]].copy()
    else:
        result_df = {'fold': [],
                     'train loss': [], 'train acc': [], 'train kappa': [],
                     'test loss': [], 'test acc': [], 'test kappa': []}
        loader = range(repeat)
    logger.info(f'loader {loader}')
    for i in loader:
        cv_dir = os.path.join(log_dir, str(i))
        if not os.path.exists(cv_dir):
            os.mkdir(cv_dir)

        train_ds, test_ds = get_dataset(cfg, subject)

        train_dl = DataLoader(train_ds,
                              batch_size=cfg['dataset']['batch_size'],
                              shuffle=True,
                              num_workers=cfg['dataset']['num_workers'])
        test_dl = DataLoader(test_ds,
                             batch_size=cfg['dataset']['test_batch_size'],
                             shuffle=False,
                             num_workers=cfg['dataset']['num_workers'])

        model = NeuroGuideNet(cfg['model'], cfg['dataset']['name'])
        model.to(device)

        total_param = sum([param.nelement() for param in model.parameters()])
        logger.info("Number of parameter: {:.4f}M".format(total_param / 1e6))

        filter_num = len(cfg['dataset']['filter_start']) \
            if isinstance(cfg['dataset']['filter_start'], list) else 1
        summary(model=model,
                input_size=(
                    cfg['dataset']['batch_size'],
                    cfg['model']['n_channels'],
                    int(cfg['dataset']['window_size'] * cfg['dataset']['fs']),
                    filter_num
                ),
                depth=3,
                device=device)

        criterion = nn.CrossEntropyLoss()
        kl_criterion = nn.KLDivLoss(reduction='batchmean')

        optimizer = optim.AdamW(list(model.parameters()) + list(criterion.parameters()),
                                lr=cfg['train']['base_lr'],
                                eps=cfg['train']['epsilon'],
                                betas=(cfg['train'].get('b1', 0.9),
                                       cfg['train'].get('b2', 0.999)),
                                weight_decay=cfg['train']['weight_decay'])
        scheduler_name = cfg['train'].get('scheduler', 'cosine')
        if scheduler_name == 'cosine':
            scheduler = CosineAnnealingLR(optimizer,
                                          T_max=cfg['train']['t_max'],
                                          eta_min=cfg['train']['min_learning_rate'])
        elif scheduler_name == 'ReduceLROnPlateau':
            scheduler = ReduceLROnPlateau(
                optimizer, mode='min',
                factor=cfg['train']['lr_factor'],
                patience=cfg['train']['patience'],
                min_lr=cfg['train']['min_learning_rate']
            )
        elif scheduler_name == 'None':
            scheduler = None
        else:
            raise 'Unsupported scheduler'
        max_grad_norm = cfg['train']['max_grad_norm']

        last_test_acc = 0
        patience_cnt = 0

        metrics_dict = {'train_loss': [],
                        'train_acc': [],
                        'train_kappa': [],
                        'test_loss': [],
                        'test_acc': [],
                        'test_kappa': []}

        for epoch in range(cfg['train']['epoch']):
            begin_time = time.perf_counter()

            train_loss, train_acc, train_kappa, train_loss_dict = train_one_epoch(
                model=model,
                data_loader=train_dl,
                criterion=criterion,
                kl_criterion=kl_criterion,
                optimizer=optimizer,
                rta=repeat_trail_argument,
                max_grad_norm=max_grad_norm,
                use_R_Drop=cfg['train'].get('use_R_Drop', False),
                R_Drop_alpha=cfg['train'].get('R_Drop_alpha', 1.),
                use_center_loss=cfg['train'].get('use_center_loss', False),
                center_loss_lambda=cfg['train'].get('center_loss_lambda', 1.),
                device=device,
                num_classes=cfg['model']['num_class'],
                label_name=cfg['dataset']['label_name']
            )
            test_loss, test_acc, test_kappa, test_loss_dict = evaluate_model(
                model=model,
                data_loader=test_dl,
                criterion=criterion,
                use_center_loss=cfg['train'].get('use_center_loss', False),
                center_loss_lambda=cfg['train'].get('center_loss_lambda', 1.),
                device=device,
                num_classes=cfg['model']['num_class'],
                label_name=cfg['dataset']['label_name'],
                log_dir=cv_dir,
                window_size=int(cfg['dataset']['window_size'] * cfg['dataset']['fs']),
                window_stride=int(cfg['dataset']['window_stride'] * cfg['dataset']['fs']),
                eval_name='test'
            )

            metrics_dict['train_loss'].append(train_loss)
            metrics_dict['train_acc'].append(train_acc)
            metrics_dict['train_kappa'].append(train_kappa)
            metrics_dict['test_loss'].append(test_loss)
            metrics_dict['test_acc'].append(test_acc)
            metrics_dict['test_kappa'].append(test_kappa)

            if (epoch + 1) % cfg['train']['save_every_n_epochs'] == 0:
                save_dir = cv_dir
                if not os.path.exists(save_dir):
                    os.mkdir(save_dir)
                epoch_path = os.path.join(save_dir,
                                          'epoch-{}.pt'.format(epoch + 1))
                torch.save(model.state_dict(), epoch_path)

            if last_test_acc < test_acc:
                if not os.path.exists(cv_dir):
                    os.mkdir(cv_dir)
                patience_cnt = 0
                logger.info('Test acc from {:.2f} to {:.2f}'.format(
                    last_test_acc,
                    test_acc))
                last_test_acc = test_acc
                torch.save(model.state_dict(), os.path.join(cv_dir, 'best.pt'))
            else:
                patience_cnt += 1

            if patience_cnt >= cfg['train']['patience']:
                break

            if scheduler is not None:
                if isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step(train_loss)
                else:
                    scheduler.step()

            logger.info(
                'Epoch:{}, train loss:{:.6f}, train acc:{:.2f}%, train kappa {:.2f} \n'
                'test loss:{:.6f}, test acc:{:.2f}%, test kappa {:.2f}, \n'
                'train loss dict: {} \n'
                'test loss dict: {} \n'
                ', lr={}, patience {}'.format(
                    epoch,
                    train_loss,
                    train_acc,
                    train_kappa,
                    test_loss,
                    test_acc,
                    test_kappa,
                    str(train_loss_dict),
                    str(test_loss_dict),
                    str(optimizer.state_dict()['param_groups'][0]['lr']),
                    patience_cnt)
            )

            time_elapsed = time.perf_counter() - begin_time
            logger.info('time elapsed:{:.2f}'.format(time_elapsed))

        model.load_state_dict(torch.load(os.path.join(cv_dir, 'best.pt')))
        logger.info('Load best params')

        train_loss, train_acc, train_kappa, train_loss_dict = evaluate_model(
            model=model,
            data_loader=train_dl,
            criterion=criterion,
            use_center_loss=cfg['train'].get('use_center_loss', False),
            center_loss_lambda=cfg['train'].get('center_loss_lambda', 1.),
            device=device,
            num_classes=cfg['model']['num_class'],
            label_name=cfg['dataset']['label_name'],
            log_dir=cv_dir,
            window_size=int(cfg['dataset']['window_size'] * cfg['dataset']['fs']),
            window_stride=int(cfg['dataset']['window_stride'] * cfg['dataset']['fs']),
            plot_matrix=True,
            eval_name='train'
        )
        test_loss, test_acc, test_kappa, test_loss_dict = evaluate_model(
            model=model,
            data_loader=test_dl,
            criterion=criterion,
            use_center_loss=cfg['train'].get('use_center_loss', False),
            center_loss_lambda=cfg['train'].get('center_loss_lambda', 1.),
            device=device,
            num_classes=cfg['model']['num_class'],
            label_name=cfg['dataset']['label_name'],
            log_dir=cv_dir,
            window_size=int(cfg['dataset']['window_size'] * cfg['dataset']['fs']),
            window_stride=int(cfg['dataset']['window_stride'] * cfg['dataset']['fs']),
            plot_matrix=True,
            eval_name='test'
        )
        logger.info('Test Loss {:.6f} | Test Acc {:.2f} | Test Kappa {:.2f}'.format(
            test_loss, test_acc, test_kappa))

        result_df['fold'].append(i)
        result_df['train loss'].append(train_loss)
        result_df['train acc'].append(train_acc)
        result_df['train kappa'].append(train_kappa)
        result_df['test loss'].append(test_loss)
        result_df['test acc'].append(test_acc)
        result_df['test kappa'].append(test_kappa)

        plt.figure()
        plt.plot(metrics_dict['train_loss'], label='train')
        plt.plot(metrics_dict['test_loss'], label='test')
        plt.title('Train {:.4f} | Test {:.4f}'.format(
            min(metrics_dict['train_loss']),
            min(metrics_dict['test_loss']))
        )
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(cv_dir, 'loss.png'))
        plt.clf()

        plt.figure()
        plt.plot(metrics_dict['train_acc'], label='train')
        plt.plot(metrics_dict['test_acc'], label='test')
        plt.title('Train {:.2f}% | Test {:.2f}%'.format(
            max(metrics_dict['train_acc']),
            max(metrics_dict['test_acc'])
        ))
        plt.xlabel('Epoch')
        plt.ylabel('Acc')
        plt.legend()
        plt.savefig(os.path.join(cv_dir, 'acc.png'))
        plt.clf()

        plt.figure()
        plt.plot(metrics_dict['train_kappa'], label='train')
        plt.plot(metrics_dict['test_kappa'], label='test')
        plt.title('Train {:.2f} | Test {:.2f}'.format(
            max(metrics_dict['train_kappa']),
            max(metrics_dict['test_kappa']))
        )
        plt.xlabel('Epoch')
        plt.ylabel('Kappa')
        plt.legend()
        plt.savefig(os.path.join(cv_dir, 'kappa.png'))
        plt.clf()

        if state_dict is not None:
            state_dict['subject_dict'][subject[0]].remove(i)
            state_dict['result_df'] = result_df
            with open(state_file, 'wb') as f:
                pickle.dump(state_dict, f)
        gc.collect()

    result_df = pd.DataFrame(result_df)
    result_df.to_csv(os.path.join(log_dir, 'result.csv'))
    return result_df
