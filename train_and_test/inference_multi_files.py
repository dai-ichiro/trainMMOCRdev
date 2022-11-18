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

results = ocr_model.readtext(imgs, print_result=False, show=False)
texts_list = results[0]['rec_texts']
for text in texts_list:
    print(text)
