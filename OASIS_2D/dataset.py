from torch.utils.data import Dataset
import re, os, SimpleITK
import random
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image

class OASIS_Dataset(Dataset):
    def __init__(
        self, root:str='OASIS_2D', 
        flag='train', transform=None, 
        seed=7, vit=False
    ) -> None:
        """
        Initializes an instance of the OASIS_Dataset class.

        Args:
            root (str, optional): The root directory of the dataset. Defaults to 'OASIS_2D'.
            flag (str, optional): The flag indicating whether to use the training, testing, or all data. Must be one of 'train', 'test', or 'all'. Defaults to 'train'.
            transform (callable, optional): A function/transform that takes in an PIL image and returns a transformed version. Defaults to None.
            seed (int, optional): The random seed. Defaults to 7.
            vit (bool, optional): Whether to use the Vision Transformer model. Defaults to False.

        Raises:
            AssertionError: If the flag is not one of 'train', 'test', or 'all'.

        Returns:
            None
        """
        self.root = root
        
        assert flag in ['train', 'test', 'all'], 'flag must be either train, test or all'
        self.flag = flag
        self.transform = transform
        
        self.seed = seed
        self.vit = vit
        
        patients = self.check_files()
        
        if self.flag == 'all':
            selected_patients = patients
        else:
            selected_patients = self.train_test_split(patients)
        
        self.images, self.labels, self.patient_ids, self.days = self._load(selected_patients)
        
    def check_files(self):
        patients = {}
        
        for subdir in ["disease", "healthy"]:
            label = (subdir == 'disease')
            
            for filename in os.listdir(os.path.join(self.root, subdir)):
                patient_id, day = re.findall(r'\d+', filename)
                
                key = (patient_id, day)
                # assuming same patient id and day can't be in both healthy and disease dir
                if key in patients: 
                    print(f'Duplicate case found {key} !')
                else: patients[key] = label
                
        return patients
    
    def train_test_split(self, patients):        
        patient_ids, labels = [], []
        for (patient_id, day) in patients:
            if patient_id in patient_ids: continue
            
            patient_ids.append(patient_id)
            # assuming same patient id and day can't be in both healthy and disease dir
            labels.append(patients[(patient_id, day)])
        
        train_ids, test_ids = train_test_split(
            patient_ids, test_size=0.2, shuffle=True, 
            random_state=self.seed, stratify=labels
        )
        
        # shuffling is inplace
        # random.Random(self.seed).shuffle(indices)
        if self.flag == 'train':
            return {
                key: patients[key]
                for key in patients if key[0] in train_ids
            }
        else:
            return {
                key: patients[key]
                for key in patients if key[0] in test_ids
            }
        
    def _load(self, selected_patients):
        # initialize
        images, labels = [], []
        patient_ids, days = [], []
        
        # load selected images
        for subdir, label in zip(["disease", "healthy"], [1, 0]):
            for filename in os.listdir(os.path.join(self.root, subdir)):
                patient_id, day = re.findall(r'\d+', filename)
                if (patient_id, day) not in selected_patients: 
                    continue

                image_path = os.path.join(self.root, subdir, filename)
                image = SimpleITK.GetArrayFromImage((SimpleITK.ReadImage(image_path)))
                images.append(image)
                labels.append(label)
                
                patient_ids.append(int(patient_id))
                days.append(int(day))
                
        total = len(labels)
        disease_patients = sum(labels)
        print(f'Total {total}, disease {disease_patients}, healthy {total - disease_patients}.')
        
        return images, labels, patient_ids, days
        
    def __len__(self):
        return len(self.labels)
    
    def _grayscale_to_rgb(self, x, channel_first=True):
        if channel_first: axis =0 # C x W X H
        else: axis = -1 # W x H x C
        
        return np.repeat(np.expand_dims(x, axis=axis), repeats=3, axis=axis)
    
    def __getitem__(self, index):
        # https://stackoverflow.com/questions/10965417/how-to-convert-a-numpy-array-to-pil-image-applying-matplotlib-colormap
        y = self.labels[index]
        
        if self.vit:
            x = Image.fromarray(
                np.int16(self.images[index]*255)
            ).convert('RGB')
        else:
            x = self._grayscale_to_rgb(self.images[index])
        
        if self.transform: x = self.transform(x)
        
        return x, y