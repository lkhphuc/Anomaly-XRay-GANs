import os
import cv2
import pandas as pd
from pathlib import Path

from torch.utils.data import DataLoader, Dataset 
from torchvision import utils, transforms, datasets

def customDf(path, studyClass=None, studyResult=None):
    '''
    Function to get custom csv based on class of study and type of study
    Args:
        - path(string): path to original csv
        - studyClass(list): class of study, list must contains one of the following: 
            "XR_ELBOW", 
            "XR_FINGER",
            "XR_FOREARM", 
            "XR_HAND", 
            "XR_HUMERUS", 
            "XR_SHOULDER", 
            "XR_WRIST"
            if None, take all 
        - studyResult(list): Result of study, list must contains one of the following:
            "positive", "negative"
            if None, take all
    '''
    df = pd.read_csv(path, header=None)
    
    if studyClass:
        cond = df[0].str.contains(studyClass)
        df = df[cond]
    if studyResult:
        cond = df[0].str.contains(studyResult)
        df = df[cond]
    return df


class MURA_dataset(Dataset):
    '''
    Dataset class for MURA dataset
    Args:
        - df: Dataframe with the first columns contains the path to the images
        - root_dir: string contains path of  root directory
        - transforms: Pytorch transform operations
    '''
    
    def __init__(self, df, root_dir, transforms=None):
        self.df = df 
        self.root_dir = root_dir
        self.transforms = transforms
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.df.iloc[idx, 0])
        img = cv2.imread(img_name)
        
        if self.transforms:
            img = self.transforms(img)

        if 'negative' in img_name: label = 0
        else: label = 1
        
        return img, label

