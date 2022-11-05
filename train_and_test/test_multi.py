import warnings
warnings.filterwarnings('ignore')

import sys
import os
import glob
from mmocr.ocr import MMOCR

img_dir = sys.argv[1]

imgs = glob.glob(os.path.join(img_dir, '*.jpg'))

cfg = 'satrn_output/satrn_japanese_cfg.py'
checkpoint = 'satrn_output/epoch_3.pth'

ocr_model = MMOCR(
    det = None,
    recog_config = cfg, 
    recog_ckpt = checkpoint,
    device = 'cuda'
    )

for img in imgs:
    results = ocr_model.readtext(img, print_result=False, show=False)
    print(results['rec_texts'])