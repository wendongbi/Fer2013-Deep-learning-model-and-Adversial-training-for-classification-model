import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
torch.set_default_tensor_type(torch.FloatTensor)


class Datasets(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.csv_file = pd.read_csv(csv_file)
        self.transform = transform
        self.root_dir = root_dir

    def __len__(self):
        return len(self.csv_file)

    def __getitem__(self, index):
        # Sample of our dataset will be a tuple(image,cls)
        img_name = os.path.join(self.root_dir, self.csv_file.iloc[index, 0])
        image = Image.open(img_name)
        # print(image.shape)
        cls = self.csv_file.iloc[index, 1]
        # print(cls)
        cls = int(cls)
        sample = [image, cls]

        if self.transform:
            sample[0] = self.transform(sample[0])
        return tuple(sample)


# class Rescale(object):
#     """Rescale the image in a sample to a given size.

#     Args:
#         output_size (tuple or int): Desired output size. If tuple, output is
#             matched to output_size. If int, smaller of image edges is matched
#             to output_size keeping aspect ratio the same.
#     """

#     def __init__(self, output_size):
#         assert isinstance(output_size, (int, tuple))
#         self.output_size = output_size

#     def __call__(self, sample):
#         image, cls = sample[0], sample[1]

#         h, w = image.shape[0:2]
#         if isinstance(self.output_size, int):
#             if h > w:
#                 new_h, new_w = self.output_size * h / w, self.output_size
#             else:
#                 new_h, new_w = self.output_size, self.output_size * w / h
#         else:
#             new_h, new_w = self.output_size

#         new_h, new_w = int(new_h), int(new_w)
#         img = transform.resize(image, (new_h, new_w),mode='constant')

#         # h and w are swapped for landmarks because for images,
#         # x and y axes are axis 1 and 0 respectively

#         return (img, cls)


# class ToTensor(object):
#     """Convert ndarrays in sample to Tensors."""

#     def __call__(self, sample):
#         image, cls = sample[0], sample[1]

#         # swap color axis because
#         # numpy image: H x W x C
#         # torch image: C X H X W
#         image = image.transpose((2, 0, 1))
#         image = torch.tensor(image)
#         return (image.type(torch.FloatTensor), torch.tensor(cls).type(torch.LongTensor))


if __name__ == '__main__':

    transformed_dataset = Datasets(
        './Annotations/train.csv', './Image/Train/',
        transforms.Compose([
            transforms.Resize((64,64)),
            # transforms.RandomCrop(64, padding=4),  #先四周填充0，在吧图像随机裁剪成32*32
            # transforms.RandomHorizontalFlip(),  #图像一半的概率翻转，一半的概率不翻转
            transforms.ToTensor()
        ]))
    dataloader = DataLoader(transformed_dataset, 512, True, num_workers=1)
    for batch_idx, (data,target) in enumerate(dataloader):
        print(data.shape)