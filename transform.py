import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import PIL
import datautils

class TransformsSimCLR:
    """
    A stochastic data augmentation module
    """

    def __init__(self, size):
        s = 1
        kernel_size=int(size*0.1)
        if kernel_size%2 == 0:
            kernel_size += 1
        

        self.train_transform = transforms.Compose([
                transforms.RandomResizedCrop(
                    size,
                    scale=(0.08, 1.0),
                    interpolation=PIL.Image.BICUBIC,
                ),
                transforms.RandomHorizontalFlip(0.5),
                datautils.get_color_distortion(s=1.0),
                transforms.ToTensor(),
                datautils.GaussianBlur(size // 10, 0.5),
                datautils.Clip(),
            ])
        
    def __call__(self, x):
        return self.train_transform(x), self.train_transform(x)    


def get_transformtraintest(imgsize):
    train_transform = transforms.Compose([
                    transforms.RandomResizedCrop(
                        imgsize,
                        scale=(0.08, 1.0),
                        interpolation=PIL.Image.BICUBIC,
                    ),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    lambda x: (255*x).byte(),
                ])
    test_transform = transforms.Compose([
                    datautils.CenterCropAndResize(proportion=0.875, size=imgsize),
                    transforms.ToTensor(),
                ])
    
    return train_transform,test_transform




if __name__ == '__main__':
    train_dataset = torchvision.datasets.STL10(
        "/dataset",
        split="unlabeled",
        download=True,
        transform=TransformsSimCLR(size=96),
    )
    
    train_loader = DataLoader(train_dataset, batch_size=128, num_workers=2)
    for batchnum, ((xis,xjs), _) in enumerate(train_loader):
        if batchnum==1:
            print(xis.size())
            break
        
        