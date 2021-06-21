from torch.utils.data import Dataset
import cv2

class ImageDataset(Dataset):
    def __init__(self, file_names, labels=None, transform=None, return_dict=True):
        """
        Args:
            file_names
            labels
            transform
            return_dict  (bool): if True, __getitem__ returns dict type, else, returns tuple
        """
        self.file_names = file_names
        self.labels = labels
        self.transform = transform
        self.return_dcit = return_dict
        
    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        file_path = self.file_names[idx]
        image = cv2.imread(file_path)
        if self.transform:
            image = self.transform(image=image)['image']
        if self.labels is None:
            if self.return_dict:
                return {'image': image}
            else:
                return image
            
        else:
            label = self.labels[idx]
            if self.return_dict:
                return {'image': image, 'label': label}
            else:
                return image, label