import pandas as pd

from trainer import train
from utils import get_logger

import yaml
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'


if __name__ == '__main__':
    config_file = 'config/BCIIV2a_NeuroGuideNet.yaml'
    # config_file = 'config/OpenBMI_NeuroGuideNet.yaml'
    with open(config_file, "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    cfg['device'] = 'cpu'
    # cfg['device'] = 'cuda'
    cfg['model']['device'] = cfg['device']

    # log_dir = f'./log/EX1_{model_name}'
    log_dir = './log/Temp'
    logger = get_logger(log_dir)
    logger.info(cfg)

    repeat = 1

    if cfg['dataset']['name'] == 'BCIIV2a':
        subject_list = [str(i + 1) for i in range(9)]
    elif cfg['dataset']['name'] == 'OpenBMI':
        subject_list = [str(i + 1) for i in range(54)]
    else:
        raise ValueError(f'Dataset {cfg["dataset"]["name"]} not recognized')
    logger.info(f'subject_list: {subject_list}')

    avg_df = {'subject': [],
              'train loss': [], 'train acc': [], 'train kappa': [],
              'test loss': [], 'test acc': [], 'test kappa': []}
    for s in subject_list:
        logger.info(s)

        s_dir = os.path.join(log_dir, s)

        result_df = train(
            cfg, logger, s_dir, subject=[s],
            repeat=repeat
        )

        avg_df['subject'].append(s)
        avg_df['train loss'].append(result_df['train loss'].mean())
        avg_df['train acc'].append(result_df['train acc'].mean())
        avg_df['train kappa'].append(result_df['train kappa'].mean())
        avg_df['test loss'].append(result_df['test loss'].mean())
        avg_df['test acc'].append(result_df['test acc'].mean())
        avg_df['test kappa'].append(result_df['test kappa'].mean())

    avg_df['subject'].append('_avg')
    for idx, k in enumerate(avg_df.keys()):
        if k != 'subject':
            data = sum(avg_df[k]) / len(avg_df[k])
            avg_df[k].append(data)

    avg_df = pd.DataFrame(avg_df)
    # avg_df.sort_values(by='index', inplace=True)
    avg_df.to_csv(os.path.join(log_dir, f'{cfg["dataset"]["name"]}_avg_df.csv'))
