
import os
import argparse
import jittor as jt
import jittor.transform as transforms
from PIL import Image

from models import GeneratorUNet
from datasets import ImageDataset
from utils import plot_test_result
jt.flags.use_cuda = 1

parser = argparse.ArgumentParser(description="test flags")
parser.add_argument("--dataset", type=str, default="facades")
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--img_height", type=int, default=256, help="size of image height")
parser.add_argument("--img_width", type=int, default=256, help="size of image width")
opt = parser.parse_args()
# directories for loading data and saving results


data_dir = './data/' + opt.dataset + '/' 
save_dir = opt.dataset + "_test_results/"
model_dir = './saved_models/' + opt.dataset + '/generator_last.pkl'

if not os.path.exists(save_dir):
    os.mkdir(save_dir)
if not os.path.exists(model_dir):
    os.mkdir(model_dir)

# data pre-processing
test_transforms = [
    transforms.Resize(size=(opt.img_height, opt.img_width), mode=Image.BICUBIC),
    transforms.ImageNormalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
]

# test data
test_data = ImageDataset(data_dir, transforms_=test_transforms, mode="test").set_attrs(
    batch_size=opt.batch_size,
    shuffle=False,)
# test_data_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=opt.batch_size)

# load model

G = GeneratorUNet()
# G.cuda()
G.load(model_dir)

# test
for i, (real_B, real_A) in enumerate(test_data): # real_B  是原图，real_A是彩色图conditional图
    # input & target image data
    # x_ = Variable(input.cuda())
    # y_ = Variable(target.cuda())

    gen_image = G(real_A)
    gen_image = gen_image.data

    # show result for test data
    plot_test_result(real_B, real_A, gen_image, i, training=False, save=True, save_dir=save_dir)

    print('%d images are generated.' % (i + 1))
