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

ocr_model = MMOCR(
    det_config = config_paths, 
    det_ckpt = checkpoint_paths,
    recog = None,
    device = 'cuda'
    )

result = ocr_model.readtext(img, imshow=False) # -> dict(key:['det_polygons', 'det_scores'])

polygons = result['det_polygons']      # -> list (len: number of bboxes)

os.makedirs('trim', exist_ok=True)
original_image = cv2.imread(img)

for i, each_array in enumerate(polygons):
    poly = np.array(each_array).reshape(-1, 1, 2).astype(np.float32) 

    x, y, width, height = cv2.boundingRect(poly)
    trim_image = original_image[y:y+height, x:x+width, :]

    cv2.imwrite(os.path.join('trim', f'{i}.jpg'), trim_image)