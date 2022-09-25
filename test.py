# from email.mime import image
from PIL import Image
import torch
import argparse
import os
import torchvision.transforms as transforms
import numpy as np
from models import GeneratorUNet
from datasets import ImageDataset
# image processing
# load network
# input conditional image to G network
# save image
# def is_image_file(filename):
#     return any(filename.endwith(extension) for extension in ['.png', '.jpg', '.jpeg'])

    # def load_image(filepath):
    #     img = Image.open(filepath).convert('RGB')
    #     img = img.resize((256, 256), Image.BICUBIC)
    #     return img

    # def save_image(image_tensor, filename):
    #     image_numpy = image_tensor.float().numpy()
    #     image_numpy = (np.transpose(image_numpy, (1, 2, 0)) +1 ) / 2.0 * 255.0
    #     image_numpy = image_numpy.clip(0, 255)
    #     image_numpy = image_numpy.astype(np.uint8)
    #     image_pil = Image.fromarray(image_numpy)
    #     image_pil.save(filename)
    #     print("Image saved as {}".format(filename))


    # parser = argparse.ArgumentParser(description="pix2pix-pytorch-implementation")
    # parser.add_argument("--cuda", action="store_true", help="use cuda")
    # parser.add_argument("--nepochs", type=int, default=200, help="number of epochs to use")
    # parser.add_argument("--dataset", type=str, default="facades", help="dataset to use")

    # opt = parser.parse_args()
    # print(opt)


    # device = torch.device("cuda" if opt.cuda else "cpu")

    # model_path = "./saved_models/{}/generator_last.pth".format(opt.dataset)

    # generator = torch.load(model_path).to(device)

    # image_dir = "./data/{}/test/".format(opt.dataset)

    # image_filenames = [x for x in os.listdir(image_dir)]
    # # image_filenames = [x for x in os.listdir(image_dir) if is_image_file(x)]

    # transform_list = [transforms.ToTensor(),
    #                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    # transform = transforms.Compose(transform_list)

    # for image_name in image_filenames:
    #     img = load_image(image_dir + image_name)
    #     img = transform(img)
    #     input = img.unsqueeze(0).to(device)
    #     out = generator(input)
    #     out_img = out.detach().squeeze(0).cpu()

    #     save_image(out_img, "result/{}/{}".format(opt.dataset,image_name))



import numpy as np
import matplotlib.pyplot as plt
import os

def plot_test_result(input, target, gen_image, epoch, training=True, save=False, save_dir='results/', show=False, fig_size=(5, 5)):
    if not training:
        fig_size = (input.size(2) * 3 / 100, input.size(3)/100)

    fig, axes = plt.subplots(1, 3, figsize=fig_size)
    imgs = [input, gen_image, target]
    for ax, img in zip(axes.flatten(), imgs):
        ax.axis('off')
        ax.set_adjustable('box')
        # Scale to 0-255
        img = (((img[0] - img[0].min()) * 255) / (img[0].max() - img[0].min())).numpy().transpose(1, 2, 0).astype(np.uint8)
        ax.imshow(img, cmap=None, aspect='equal')
    plt.subplots_adjust(wspace=0, hspace=0)

    if training:
        title = 'Epoch {0}'.format(epoch + 1)
        fig.text(0.5, 0.04, title, ha='center')

    # save figure
    if save:
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        if training:
            save_fn = save_dir + 'Result_epoch_{:d}'.format(epoch+1) + '.png'
        else:
            save_fn = save_dir + 'Test_result_{:d}'.format(epoch+1) + '.png'
            fig.subplots_adjust(bottom=0)
            fig.subplots_adjust(top=1)
            fig.subplots_adjust(right=1)
            fig.subplots_adjust(left=0)
        plt.savefig(save_fn)

    if show:
        plt.show()
    else:
        plt.close()


parser = argparse.ArgumentParser(description="test flags")
parser.add_argument("--dataset", type=str, default="facades")
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--cuda", action="store_true", help="use cuda")
parser.add_argument("--img_height", type=int, default=256, help="size of image height")
parser.add_argument("--img_width", type=int, default=256, help="size of image width")
opt = parser.parse_args()
# directories for loading data and saving results


data_dir = './data/' + opt.dataset + '/' 
save_dir = opt.dataset + "_test_results/"
# model_dir = './saved_models/' + opt.dataset + '/generator_last.pth'

if not os.path.exists(save_dir):
    os.mkdir(save_dir)
# if not os.path.exists(model_dir):
#     os.mkdir(model_dir)

# data pre-processing
# Configure dataloaders
transforms_ = [
    transforms.Resize((opt.img_height, opt.img_width), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),]

# test data
test_data = ImageDataset(data_dir, transforms_=transforms_, mode="test")
test_data_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=opt.batch_size)

# load model

# G = GeneratorUNet()
# # G.cuda()
# G.load(model_dir)
model_path = "./saved_models/{}/generator_last.pth".format(opt.dataset)
device = torch.device("cuda" if opt.cuda else "cpu")
# G = torch.load(model_dir).to(device)
generator = torch.load(model_path).to(device)

from torch.autograd import Variable
# cuda = True if torch.cuda.is_available() else False
Tensor = torch.FloatTensor 

# test
for i, batch in enumerate(test_data_loader): # real_B  是原图，real_A是彩色图conditional图
    # input & target image data
    # x_ = Variable(input.cuda())
    # y_ = Variable(target.cuda())
    # input = real_A.unsqueeze(0).to(device)
    real_A = Variable(batch["B"].type(Tensor)) # conditional image
    real_B = Variable(batch["A"].type(Tensor)) # original image
    gen_image = generator(real_A)
    gen_image = gen_image.data

    # show result for test data
    plot_test_result(real_B, real_A, gen_image, i, training=False, save=True, save_dir=save_dir)

    print('%d images are generated.' % (i + 1))
