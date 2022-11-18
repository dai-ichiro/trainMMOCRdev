from mmocr.ocr import MMOCR
from mim.commands.download import download
import os
import sys
import numpy as np
import cv2

img = sys.argv[1]

os.makedirs('models', exist_ok=True)

det_checkpoint_name = 'textsnake_resnet50_fpn-unet_1200e_ctw1500'
checkpoint = download(package='mmocr', configs=[det_checkpoint_name], dest_root="models")
config_paths =os.path.join('models', det_checkpoint_name + '.py')
checkpoint_paths = os.path.join('models', checkpoint[0])

det_model = MMOCR(
    det_config = config_paths, 
    det_ckpt = checkpoint_paths,
    recog = None,
    device = 'cuda'
    )

recog_cfg = 'satrn_japanese_cfg.py'
recog_checkpoint = 'epoch_1.pth'

recog_model = MMOCR(
    det = None,
    recog_config = recog_cfg, 
    recog_ckpt = recog_checkpoint,
    device = 'cuda'
    )

det_result = det_model.readtext(img) # -> dict(key:['det_polygons', 'det_scores'])

polygons = det_result['det_polygons']              # -> list (len: number of bboxes)

original_image = cv2.imread(img)

for each_array in polygons:
    poly = np.array(each_array).reshape(-1, 1, 2).astype(np.float32) 

    x, y, width, height = cv2.boundingRect(poly)
    trim_image = original_image[y:y+height, x:x+width, :]

    recog_result = recog_model.readtext(trim_image, print_result=False, show=False)
    score = np.mean(np.array(recog_result['rec_scores']))
    if score > 0.95:
        print(recog_result['rec_texts'][0])