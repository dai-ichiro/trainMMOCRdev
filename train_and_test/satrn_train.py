import os
from mmengine.config import Config
from mmengine.registry import RUNNERS
from mmengine.runner import Runner

from mmocr.utils import register_all_modules

def main():
    cfg = Config.fromfile('satrn_japanese_cfg.py')

    os.makedirs('satrn_output', exist_ok=True)

    ####
    ## modify configuration file
    ####

    # set output dir
    cfg.work_dir = 'satrn_output'

    # Path to dictionary file
    cfg.dictionary.dict_file = 'dicts.txt'
    cfg.model.decoder.dictionary = cfg.dictionary
    
    
    # Path to annotation file and image folder
    cfg.train.data_prefix.img_path = 'train'
    cfg.train.ann_file = 'train_labels.json'
    cfg.train_dataset.datasets = [cfg.train]
    cfg.train_dataloader.dataset = cfg.train_dataset

    cfg.test.data_prefix.img_path = 'test'
    cfg.test.ann_file = 'test_labels.json'
    cfg.test_dataset.datasets = [cfg.test]
    cfg.test_dataloader.dataset = cfg.test_dataset

    cfg.val_dataloader.dataset = cfg.test_dataset

    # Modify cuda setting
    cfg.gpu_ids = range(1)
    cfg.device = 'cuda'

    # Others
    cfg.train_dataloader.batch_size = 32
    cfg.train_dataloader.num_workers = 8
    cfg.model.decoder.max_seq_len = 35
    cfg.train_cfg.max_epochs = 1 # default 5 
    
    # Build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    # Start training
    runner.train()
    
if __name__ == '__main__':
    main()
