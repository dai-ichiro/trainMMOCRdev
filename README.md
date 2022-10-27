# trainMMOCRdev

## Requirements
~~~
pip install torch==1.12.1 torchvision==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip install openmim==0.3.2
pip install mmengine==0.2.0
pip install mmcv==2.0.0rc1 -f https://download.openmmlab.com/mmcv/dist/cu116/torch1.12.0/index.html
pip install mmdet==3.0.0rc2
pip install mmocr==1.0.0rc2
pip install openpyxl==3.0.10
pip install trdg==1.8.0
~~~

## How to make datasets
### for train
trdg -l ja -c 30000 -k 1 -rk -bl 1 -rbl -fd fonts -dt text.txt -na 2 --output_dir train

### for test
trdg -l ja -c 100 -k 1 -rk -bl 1 -rbl -fd fonts -dt text.txt -na 2 --output_dir test
