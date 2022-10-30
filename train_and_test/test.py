import warnings
warnings.filterwarnings('ignore')

import sys
from mmocr.ocr import MMOCR

img = sys.argv[1]

cfg = 'satrn_output/satrn_japanese_cfg.py'
checkpoint = 'satrn_output/epoch_3.pth'

ocr_model = MMOCR(
    det = None,
    recog_config = cfg, 
    recog_ckpt = checkpoint,
    device = 'cuda'
    )

results = ocr_model.readtext(img, print_result=True, show=False)