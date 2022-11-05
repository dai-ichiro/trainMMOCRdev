import time
import os
import glob
from PIL import Image
import numpy as np

import threading

thread_count = 6
save_dir = 'train_with_noise3'

os.makedirs(save_dir, exist_ok=True)

def add_noise(img_list):
    for img in img_list:
        fname = os.path.basename(img)
        original_img = np.array(Image.open(img))
        noise = np.random.normal(0, 3, original_img.shape)
        img_with_noise = Image.fromarray((original_img + noise).astype('uint8'))
        img_with_noise.save(os.path.join(save_dir, fname))
        
start_time = time.time()

imgs = glob.glob('train/*.jpg')

img_num = int(len(imgs) / thread_count)

thread_list = []
for i in range(thread_count):
    if i != thread_count -1:
        thread_list.append(threading.Thread(target=add_noise, args=(imgs[(img_num * i):(img_num* (i + 1))],)))
    else:
        thread_list.append(threading.Thread(target=add_noise, args=(imgs[(img_num * i):],)))
        
for each_thread in thread_list:
    each_thread.start()

for each_thread in thread_list:
    each_thread.join()

finish_time = time.time()

print(f'time: {finish_time - start_time} sec')