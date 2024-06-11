import random
import PIL
import numpy as np
import torch
from torchvision.transforms.v2 import Compose, ToDtype, ToImage, RandomResizedCrop, RandomHorizontalFlip, Resize, Normalize, ColorJitter




class TemporalRandomCrop(torch.nn.Module):
    """Temporally crop the given frame indices at a random location.

    If the number of frames is less than the size,
    loop the indices as many times as necessary to satisfy the size.

    Args:
        size (int): Desired output size of the crop.
    """

    def __init__(self, size):
        super(TemporalRandomCrop, self).__init__()
        self.size = size

    def forward(self, frame_indices):
        """
        Args:
            frame_indices (list): frame indices to be cropped.
        Returns:
            list: Cropped frame indices.
        """

        rand_end = max(0, len(frame_indices) - self.size - 1)
        begin_index = random.randint(0, rand_end)
        end_index = min(begin_index + self.size, len(frame_indices))

        out = frame_indices[begin_index:end_index]

        for index in out:
            if len(out) >= self.size:
                break
            out.append(index)

        return out
    
    def randomize_parameters(self):
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.size})"


def SpatialTransform(mode, crop_size):
    assert mode in ('train', 'val', 'test')

    if mode == 'train':
        return Compose([
            ToImage(),
            ToDtype(torch.uint8, scale=True),
            RandomResizedCrop((crop_size, crop_size)),
            RandomHorizontalFlip(),
            ToDtype(torch.float32, scale=True),
        ])
    else:
        return Compose([
            ToImage(),
            ToDtype(torch.uint8, scale=True),
            Resize((crop_size, crop_size)),
            ToDtype(torch.float32, scale=True),
        ])
    
def SpatialTransform_norm(mean, std):
    return Compose([
        Normalize(mean, std),
    ])
    
if __name__ == "__main__":
    img = PIL.Image.fromarray(np.random.randint(0, 255, (256, 256, 3)).astype(np.uint8))
    t = SpatialTransform('train', 224)
    d = t([img, img, img, img])
    print(d[0].shape, d[1].shape, d[2].shape, d[3].shape)
