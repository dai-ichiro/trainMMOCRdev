import os
from mmocr.ocr import MMOCR
from mim.commands.download import download
from torchvision.datasets.utils import download_url

os.makedirs('models', exist_ok=True)

############ Build OCR model ############
# Detection: textsnake
det_checkpoint_name = 'textsnake_resnet50_fpn-unet_1200e_ctw1500'

checkpoint = download(package='mmocr', configs=[det_checkpoint_name], dest_root="models")
config_paths =os.path.join('models', det_checkpoint_name + '.py')
checkpoint_paths = os.path.join('models', checkpoint[0])

ocr_model = MMOCR(
    det_config = config_paths, 
    det_ckpt = checkpoint_paths,
    recog_config = 'satrn_japanese_cfg.py', 
    recog_ckpt = 'epoch_1.pth',
    device = 'cuda'
    )
############ Build OCR model ############

results = ocr_model.readtext('sampleimage/test1.png', print_result=True, show=True)
