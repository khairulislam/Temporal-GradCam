from torch.utils.data import Dataset
import re, os, SimpleITK
import random, torch
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image

def collate_fn(data, max_len=None):
    """Build mini-batch tensors from a list of (X, mask) tuples. Mask input. Create
    Args:
        data: len(batch_size) list of tuples (X, y).
            - X: torch tensor of shape (seq_length, feat_dim); variable seq_length.
            - y: torch tensor of shape (num_labels,) : class indices or numerical targets
                (for classification or regression, respectively). num_labels > 1 for multi-task models
        max_len: global fixed sequence length. Used for architectures requiring fixed length input,
            where the batch length cannot vary dynamically. Longer sequences are clipped, shorter are padded with 0s
    Returns:
        X: (batch_size, padded_length, feat_dim) torch tensor of masked features (input)
        targets: (batch_size, padded_length, feat_dim) torch tensor of unmasked features (output)
        target_masks: (batch_size, padded_length, feat_dim) boolean torch tensor
            0 indicates masked values to be predicted, 1 indicates unaffected/"active" feature values
        padding: (batch_size, padded_length) boolean tensor, 1 means keep vector at this position, 0 means padding
    """

    batch_size = len(data)
    features, labels = zip(*data)

    # Stack and pad features and masks (convert 2D to 3D tensors, i.e. add batch dimension)
    lengths = [X.shape[0] for X in features]  # original sequence length for each time series
    if max_len is None:
        max_len = max(lengths)

    X = torch.zeros(batch_size, max_len, features[0].shape[-1])  # (batch_size, padded_length, feat_dim)
    for i in range(batch_size):
        end = min(lengths[i], max_len)
        # X[i, :end, :] = features[i][:end, :]
        # keep the most recent time steps
        # previous implementation was truncating the last few time steps if goes beyond max_len
        X[i, :end, :] = features[i][-end:, :] 

    targets = torch.stack(labels, dim=0)  # (batch_size, num_labels)

    padding = padding_mask(torch.tensor(lengths, dtype=torch.int16),
                                 max_len=max_len)  # (batch_size, padded_length) boolean tensor, "1" means keep

    return X, targets, padding


def padding_mask(lengths, max_len=None):
    """
    Used to mask padded positions: creates a (batch_size, max_len) boolean mask from a tensor of sequence lengths,
    where 1 means keep element at this position (time step)
    """
    batch_size = lengths.numel()
    max_len = max_len or lengths.max_val()  # trick works because of overloading of 'or' operator for non-boolean types
    
    mask = (torch.arange(0, max_len, device=lengths.device)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1)))
    return mask

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
    
class OASIS_TemporalDataset(Dataset):
    def __init__(
        self, features, train, 
        seed=7, test_size=0.2
    ):
        self.train = train
        self.seed = seed
        self.selected_patients = self.split(features, test_size)
        
        id_day_map = {}
        id_day_feature_map = {}

        for patient_id, day, feature, label in zip(
            features['patient_id'], features['day'], 
            features['feature'], features['label']
        ):
            if patient_id not in self.selected_patients: continue
            
            if patient_id not in id_day_map:
                id_day_map[patient_id] = [day]
            else:    
                id_day_map[patient_id].append(day)
                
            id_day_feature_map[(patient_id, day)] = (feature, label)
            
        self.id_day_map = id_day_map
        self.id_day_feature_map = id_day_feature_map
        self.id_days = list(id_day_feature_map.keys())
        
    def split(self, features, test_size):
        patient_ids, labels = [], []
        for i in range(len(features['patient_id'])):
            patient_id = features['patient_id'][i]
        
            if patient_id in patient_ids: continue
            
            patient_ids.append(patient_id)
            # assuming same patient id and day can't be in both healthy and disease dir
            labels.append(features['label'][i])
        
        train_ids, test_ids = train_test_split(
            patient_ids, test_size=test_size, shuffle=True, 
            random_state=self.seed, stratify=labels
        )
        
        if self.train: return train_ids
        else: return test_ids
        
    def __len__(self):
        return len(self.id_days)
    
    def __getitem__(self, idx):
        id, current_day = self.id_days[idx]
        
        # sort the current and previous days in ascending order
        prev_days = sorted([day for day in self.id_day_map[id] if day <= current_day])
        
        # get feature for each previous day
        features = np.array([self.id_day_feature_map[(id, d)][0] for d in prev_days]    )
        
        # predict the label for the current day
        label = self.id_day_feature_map[(id, current_day)][1]
        
        return torch.tensor(features), torch.tensor(label)