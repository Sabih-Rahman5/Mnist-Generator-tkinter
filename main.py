import torch
import matplotlib.pyplot as plt
#from IPython.display import Image, display
from torchvision.utils import save_image

from PIL import Image

import numpy as np

import os
sample_dir = 'samples'
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)


def denorm(x):
    out = (x+1)/2
    return out.clamp(0, 1)


def save_fake_images(num, index, rows = 20):
    sample_vectors = torch.randn(num, 64)
    fake_images = Generator(sample_vectors)
    fake_images = fake_images.reshape(fake_images.size(0), 1,28,28)

    fake_fname = 'fake_images-{0:0=4d}.png'.format(index)
    print('saving:', fake_fname)
    save_image(denorm(fake_images), os.path.join(sample_dir, fake_fname), nrow = rows)


Generator = torch.load('models/Generator.pth', map_location='cpu')

y = Generator(torch.randn(2, 64))   #Generate 2 random vectors of latent_size(64)
print(y.shape)


gen_imgs = denorm(y.reshape((-1, 28,28)).detach())
plt.imshow(gen_imgs[0], cmap='gray')
plt.show()


plt.imshow(gen_imgs[1], cmap='gray')

plt.show()


save_fake_images(1200, 0, 50)
pil_img = Image.open(os.path.join(sample_dir, 'fake_images-0000.png'))
im_array = np.asarray(pil_img)
plt.imshow(im_array)
plt.show()

