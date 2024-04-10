from torch.utils.data import Dataset
import re, os, SimpleITK
import random
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image

class OASIS_Dataset(Dataset):
    def __init__(
        self, root:str='OASIS_2D', 
        train:bool=False, transform=None, seed=7,
        vit=False
    ) -> None:
        self.root = root
        self.train = train
        self.transform = transform
        self.seed = seed
        self.vit = vit
        
        patients = self.unique_patients()
        selected_patients = self.train_test_split(patients)
        
        self._load(selected_patients)
        
    def unique_patients(self):
        patients = {}
        
        for subdir in ["disease", "healthy"]:
            label = (subdir == 'disease')
            
            for filename in os.listdir(os.path.join(self.root, subdir)):
                patient_id, _ = re.findall(r'\d+', filename)
                
                # assuming same patient id can't be in both healthy and disease dir
                if patient_id not in patients:
                    patients[patient_id] = label
                
        return patients
    
    def train_test_split(self, patients):
        patient_ids = list(patients.keys())
        
        train_ids, test_ids = train_test_split(
            patient_ids, test_size=0.2, shuffle=True, 
            random_state=self.seed, stratify=list(patients.values())
        )
        
        # shuffling is inplace
        # random.Random(self.seed).shuffle(indices)
        
        if self.train:
            return train_ids
        else:
            return test_ids
        
    def _load(self, selected_patients):
        # initialize
        images, labels = [], []
        patient_ids, days = [], []
        
        # load selected images
        for subdir, label in zip(["disease", "healthy"], [1, 0]):
            for filename in os.listdir(os.path.join(self.root, subdir)):
                patient_id, day = re.findall(r'\d+', filename)
                if patient_id not in selected_patients: 
                    continue

                image_path = os.path.join(self.root, subdir, filename)
                image = SimpleITK.GetArrayFromImage((SimpleITK.ReadImage(image_path)))
                images.append(image)
                labels.append(label)
                
                patient_ids.append(int(patient_id))
                days.append(int(day))
                
        total = len(labels)
        disease_patients = sum(labels)
        print(f'Total {total}, disease {disease_patients}, healthy {total - disease_patients}. \
            Unique patients {len(selected_patients)}.')
        
        # save data
        self.labels = labels
        self.images = images
        self.patient_ids = patient_ids
        self.days = days
        
    def __len__(self):
        return len(self.labels)
    
    def _grayscale_to_rgb(self, x):
        return np.repeat(np.expand_dims(x, axis=0), repeats=3, axis=0)
    
    def __getitem__(self, index):
        # https://stackoverflow.com/questions/10965417/how-to-convert-a-numpy-array-to-pil-image-applying-matplotlib-colormap
        y = self.labels[index]
        
        if self.vit:
            x = Image.fromarray(
                np.int16(self.images[index]*255)
            ).convert('RGB')
        else:
            x = self._grayscale_to_rgb(self.images[index])
        
        
        if self.transform:
            x = self.transform(x)
        
        return x, y