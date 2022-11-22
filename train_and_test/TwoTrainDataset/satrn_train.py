import os
from mmengine.config import Config
from mmengine.registry import RUNNERS
from mmengine.runner import Runner

#from mmocr.utils import register_all_modules

def main():
    cfg = Config.fromfile('satrn_japanese_cfg.py')

    os.makedirs('satrn_output', exist_ok=True)

    ####
    ## modify configuration file
    ####

    # Set output dir
    cfg.work_dir = 'satrn_output'
    
    # Modify cuda setting
    cfg.gpu_ids = range(1)
    cfg.device = 'cuda'

    # Others
    cfg.train_cfg.max_epochs = 3 # default 5 
    cfg.default_hooks.logger.interval = 2000
    
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
