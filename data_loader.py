import torch
import torchvision.datasets as dsets
from torchvision import transforms
from data_mura import customDf, MURA_dataset


class Data_Loader():
    def __init__(self, train, dataset, mura_class, mura_type, image_path, image_size, batch_size, shuf=True):
        self.dataset = dataset
        self.path = image_path
        self.imsize = image_size
        self.batch = batch_size
        self.shuf = shuf
        self.train = train
        self.mura_class = mura_class
        self.mura_type = mura_type

    def transform(self, rotation, hflip, resize, totensor, normalize, centercrop, to_pil, gray):
        options = []
        if to_pil:
            options.append(transforms.ToPILImage())
        if gray:
            options.append(transforms.Grayscale())
        if rotation:
            options.append(transforms.RandomRotation(20))
        if hflip:
            options.append(transforms.RandomHorizontalFlip())
        if centercrop:
            options.append(transforms.CenterCrop(256))
        if resize:
            options.append(transforms.Resize((self.imsize,self.imsize)))
        if totensor:
            options.append(transforms.ToTensor())
        # if True:
        #     options.append(transforms.Lambda(lambda x: (x - x.min())/(x.max()-x.min())))
        if normalize:
            options.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        transform = transforms.Compose(options)
        return transform

    def load_lsun(self, classes='church_outdoor_train'):
        transforms = self.transform(True, True, True, False)
        dataset = dsets.LSUN(self.path, classes=[classes], transform=transforms)
        return dataset

    def load_celeb(self):
        transforms = self.transform(True, True, True, True)
        dataset = dsets.ImageFolder(self.path+'/CelebA', transform=transforms)
        return dataset

    def load_mura(self, studyClass, studyType):
        transforms = self.transform(False, True, True, True, True, True, True, False)
        mura_df = customDf(self.path+'/MURA-v1.1/train_image_paths.csv', studyClass, studyType)
        dataset = MURA_dataset(mura_df, self.path, transforms)
        # dataset = dsets.ImageFolder(self.path+'/MURA-v1.1/train/'+classes, transform=transforms)
        return dataset

    def loader(self):
        if self.dataset == 'lsun':
            dataset = self.load_lsun()
        elif self.dataset == 'celeb':
            dataset = self.load_celeb()
        elif self.dataset == 'mura':
            dataset = self.load_mura(self.mura_class, self.mura_type)

        print("Dataset lengh: ", len(dataset))

        loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=self.batch,
                                              shuffle=self.shuf,
                                              num_workers=4,
                                              drop_last=False)
        return loader

