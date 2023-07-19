import os
import numpy as np
from PIL import Image
image_dir = "/home/wangy0k/Desktop/owntree/hyperacorncontinue_bak/dataset/bin5/10643_40_bin5"
# img = cv2.imread(os.path.join(image_dir, '0001.png'),cv2.IMREAD_GRAYSCALE)
# imagenames = np.loadtxt(os.path.join(image_dir,'images.txt' ), delimiter='\n',dtype=str)
masknames  = np.loadtxt(os.path.join(image_dir,'masks.txt' ), delimiter='\n',dtype=str)
imagesavedir = os.path.join(image_dir,'masks_all2')
os.makedirs(imagesavedir, exist_ok=True)
for idx in range(0,masknames.size):
    # imagename = '0040.tiff'
# idx = 9
#     image = Image.open(os.path.join(image_dir,'images' ,imagenames[idx]))
#     img = np.copy(image).astype(np.float32)
    mask = Image.open(os.path.join(image_dir,'maskmarker', masknames[idx]))
    mask = np.copy(mask).astype(np.float32)/255.

    mask2 = Image.open(os.path.join(image_dir,'masks_ploy',masknames[idx]))
    mask2 = np.copy(mask2).astype(np.float32)


    mask2 = mask2[:,:,0]
    mask2[mask2>0] = 1.
    # mask = np.ones_like(mask2)
    mask2 = 1 -mask2
    mask3 = mask * mask2

    image_new2 = np.array(mask3[:,:] * 255.).astype(np.uint8)
    image_tosave = Image.fromarray(image_new2)
    image_savefile = os.path.join(imagesavedir, masknames[idx])
    image_tosave.save(image_savefile)