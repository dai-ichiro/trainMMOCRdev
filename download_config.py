import os
from mim.commands.download import download

os.makedirs('models', exist_ok=True)

# Recognition: satrn
recog_checkpoint_name = 'satrn_shallow_5e_st_mj'
checkpoints = download(package='mmocr', configs=[recog_checkpoint_name], dest_root="models")
