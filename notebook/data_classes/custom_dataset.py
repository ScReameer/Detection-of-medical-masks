import os
import xml.etree.ElementTree as ET
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from PIL import Image

class CustomImageDataset(Dataset):
    def __init__(
        self, 
        path_images,
        path_annotations,
        input_size=None,
        split=None, 
        random_state=42,
        test_size=0.3,
        transform=None,
        device='cpu'
        ):
        """Dataset for image detection

        Args:
            `data_path` (`str`): path to data
            `split` ([`None`, `str`], optional): Splits data to `train`, `valid` and `test`. Defaults to `None` (full dataset without split).
            `target_size` (`tuple`): network input size of images to resize bounding boxes
            `random_state` (`float`, optional): Random state for group splitting. Defaults to `42`
            `test_size` (`float`, optional): Size of test data, when `split` is not `None`. Defaults to `0.2`
            `transform` (`torchvision.transforms.v2.Compose`, optional): Augmentation and/or normalization. Defaults to `None`.
            `device` (`str`, optional): Move data to `cuda` or `cpu`. Defaults to `cpu`.
        
        """
        super().__init__()
        self.data_path_annotations = path_annotations
        self.data_path_images = path_images
        self.split = split
        self.transform = transform
        self.input_size = input_size
        self.device = device
        self.label_encoder = LabelEncoder()
        self.items = []
        # Parse xml annotations
        for xml_file in os.listdir(self.data_path_annotations):
            tree = ET.parse(os.path.join(self.data_path_annotations, xml_file))
            root = tree.getroot()
            size = tree.find('size')
            width = int(size.find('width').text)
            height = int(size.find('height').text)
            for obj in root.findall('object'):
                bbx = obj.find('bndbox')
                xmin = float(bbx.find('xmin').text)
                ymin = float(bbx.find('ymin').text)
                xmax = float(bbx.find('xmax').text)
                ymax = float(bbx.find('ymax').text)
                # Encode label
                label = obj.find('name').text
                result = (
                    root.find('filename').text,
                    label,
                    width,
                    height,
                    xmin,
                    ymin,
                    xmax,
                    ymax
                )
                self.items.append(result)
        self.full_df = pd.DataFrame(
            self.items, 
            columns=['name', 'label', 'width', 'height', 'xmin', 'ymin', 'xmax', 'ymax']
        )
        for col in ['xmin', 'xmax']:
            self.full_df[col] = self.full_df[col].clip(0, self.full_df['width'].values)
        for col in ['ymin', 'ymax']:
            self.full_df[col] = self.full_df[col].clip(0, self.full_df['height'].values)
        # Split data to train, valid and test samples
        if split:
            self.random_state = random_state
            self.test_size = test_size
            self.sss = StratifiedShuffleSplit(n_splits=1, test_size=self.test_size, random_state=self.random_state)
            # Get train indexes
            train_idxs, remainder_idxs = list(self.sss.split(
                [item[0] for item in self.items], 
                [item[1] for item in self.items]
            ))[0]
            # Get valid and test indexes (1:1)
            half_total_remainder_len = len(remainder_idxs) // 2
            valid_idxs, test_idxs = remainder_idxs[:half_total_remainder_len], remainder_idxs[half_total_remainder_len:]
            # Split data
            match split:
                case 'train':
                    self.split_df = self.full_df.iloc[train_idxs, :].copy()
                case 'val':
                    self.split_df = self.full_df.iloc[valid_idxs, :].copy()
                case 'test':
                    self.split_df = self.full_df.iloc[test_idxs, :].copy()
        # self.df['xmin_norm'] = self.df['xmin'] / self.df['width']
        # self.df['xmax_norm'] = self.df['xmax'] / self.df['width']
        # self.df['ymin_norm'] = self.df['ymin'] / self.df['height']
        # self.df['ymax_norm'] = self.df['ymax'] / self.df['height']
        self.label_encoder.fit(self.full_df['label'])
        self.unique_images = self.split_df['name'].unique()
        
    def __len__(self):
        return len(self.unique_images)
    
    def __getitem__(self, idx):
        # Get image name, convert to tensor
        image_name = self.unique_images[idx]
        img_rgb = np.array(Image.open(os.path.join(self.data_path_images, image_name)).convert('RGB'))
        # Get all rows with one unique image [idx]
        all_rows = self.full_df[self.full_df['name'] == image_name]
        # All boxes, belongs to unique image
        boxes = all_rows[['xmin', 'ymin', 'xmax', 'ymax']].values
        # All labels, belongs to every box
        labels = self.label_encoder.transform(all_rows['label'])
        if self.transform:
            img_transformed, boxes_transformed, labels_transformed = self.transform(image=img_rgb, bboxes=boxes, class_labels=labels).values()
        # Target dict with boxes and labels, belongs to one unique image
        target = {}
        target['boxes'] = torch.as_tensor(boxes_transformed, dtype=torch.float32, device=self.device)
        target['labels'] = torch.tensor(labels_transformed, device=self.device)
        img_transformed = torch.as_tensor(img_transformed, dtype=torch.float32, device=self.device).permute(2, 0, 1)
        return img_transformed, target