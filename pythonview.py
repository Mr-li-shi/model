import numpy as np
import matplotlib.pyplot as plt
import torch.nn
from torch import nn

class view(nn.Module):
    def __init__(self, model, layer_name, index, row):
        super(view, self).__init__()
        self.model = model
        self.layer_name = layer_name
        self.row = row
        self.out = []
        self.a = []
        self.b = self.layer(self.a)

        self.b[index].register_forward_hook(hook=self.hook)
    def layer(self, a:list ):
        for index, layer in enumerate(self.model.modules()):
            print(layer)
            if isinstance(layer, self.layer_name):
                print('生成Conv2d层数，第{}层。'.format(index))
                a.append(layer)
        return a
    def g(self, images, row):
        row_num = images.shape[0] // row
        for i in range(row_num):
            img = images[i * row]
            img = (img - img.min()) / (img.max() - img.min())
            for j in range(1, row):
                img1 = images[i * row + j]
                img1 = (img1 - img1.min()) / (img1.max() - img1.min())
                img = np.hstack((img, img1))
            if i == 0:
                ans = img
            else:
                ans = np.vstack((ans, img))
        return ans
    def hook(self, module, f_in, f_out):
        print(f_out.size())
        self.out.append(f_out)
        img0 = f_out.cpu().detach().squeeze(0).numpy()
        img0 = self.g(img0, self.row)
        self.fig(img0, 'gray')
    def fig(self, img, cmp):
        plt.figure()
        plt.imshow(img, cmp)

