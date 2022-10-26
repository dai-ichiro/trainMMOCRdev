import os
from mmengine.config import Config

def main():
    cfg = Config.fromfile('satrn_japanese_cfg.py')

    #os.makedirs('satrn_output', exist_ok=True)

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
    cfg.test.ann_file = 'train_labels.json'
    cfg.test_dataset.datasets = [cfg.test]
    cfg.test_dataloader.dataset = cfg.test_dataset

    cfg.val_dataloader.dataset = cfg.test_dataset


    # modify cuda setting
    cfg.gpu_ids = range(1)
    cfg.device = 'cuda'

    # Others
    cfg.model.decoder.max_seq_len = 35
    cfg.train_cfg.max_epochs = 1 # default 5 

    cfg.dump('new_SATRN_cfg.py')
    '''
    # Build dataset
    datasets = [build_dataset(cfg.data.train)]

    model = build_detector(cfg.model)
    model.CLASSES = datasets[0].CLASSES
    model.init_weights()

    train_detector(model, datasets, cfg, validate=True)
    '''
if __name__ == '__main__':
    main()