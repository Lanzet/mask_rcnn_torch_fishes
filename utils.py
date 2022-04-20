import os
import numpy as np
import torch
import torch.utils.data
import torchvision
from PIL import Image, ImageDraw
from pycocotools.coco import COCO
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import matplotlib.pyplot as plt


class myOwnDataset(torch.utils.data.Dataset):
    def __init__(self, root, annotation, transforms=None):
        self.root = root
        self.transforms = transforms
        self.coco = COCO(annotation)
        self.ids = list(sorted(self.coco.imgs.keys()))

    def __getitem__(self, index):
        # Own coco file
        coco = self.coco
        # Image ID
        img_id = self.ids[index]
        # List: get annotation id from coco
        ann_ids = coco.getAnnIds(imgIds=img_id)
        # Dictionary: target coco_annotation file for an image
        coco_annotation = coco.loadAnns(ann_ids)
        # path for input image
        path = coco.loadImgs(img_id)[0]["file_name"]
        # open the input image
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        # img = img_ori(:,:,:)
        # number of objects in the image
        num_objs = len(coco_annotation)

        # Bounding boxes for objects
        # In coco format, bbox = [xmin, ymin, width, height]
        
        mask = []
        # In pytorch, the input should be [xmin, ymin, xmax, ymax]
        boxes = []
        for i in range(num_objs):
            segm = coco_annotation[i]["segmentation"][0]
            img = Image.new('L', (img.size[0] , img.size[1]), 0)
            ImageDraw.Draw(img).polygon(segm, outline=1, fill=1)
            mask.append(np.array(img))
            xmin = coco_annotation[i]["bbox"][0]
            ymin = coco_annotation[i]["bbox"][1]
            xmax = xmin + coco_annotation[i]["bbox"][2]
            ymax = ymin + coco_annotation[i]["bbox"][3]
            boxes.append([xmin, ymin, xmax, ymax])
        
        # print(mask.shape)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # print(boxes.shape)
        # Labels (In my case, I only one class: target class or background)
        labels = torch.ones((num_objs,), dtype=torch.int64)
        # Tensorise img_id
        img_id = torch.tensor([img_id])
        # Size of bbox (Rectangular)
        areas = []
        for i in range(num_objs):
            areas.append(coco_annotation[i]["area"])
        areas = torch.as_tensor(areas, dtype=torch.float32)
        # Iscrowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        
        # Annotation is in dictionary format
        my_annotation = {}
        my_annotation["boxes"] = boxes
        my_annotation["labels"] = labels
        my_annotation["image_id"] = img_id
        my_annotation["area"] = areas
        my_annotation["iscrowd"] = iscrowd
        
        masks = []
        if self.transforms is not None:
            img = self.transforms(img)
            # mask = self.transforms(mask) # Lalo del futuro, lo siento por este error
            for mas_k in mask:
                mas_k = self.transforms(mas_k)
                masks.append(mas_k)
                
        # mask = torch.as_tensor(masks, dtype=torch.uint8).unsqueeze(dim=0)        
        my_annotation["masks"] = masks

        return img, my_annotation

    def __len__(self):
        return len(self.ids)


# In my case, just added ToTensor
def get_transform():
    custom_transforms = []
    custom_transforms.append(torchvision.transforms.Resize(64))
    custom_transforms.append(torchvision.transforms.ToTensor())
    custom_transforms.append(torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    return torchvision.transforms.Compose(custom_transforms)


# collate_fn needs for batch
def collate_fn(batch):
    return tuple(zip(*batch))


def get_model_instance_segmentation(num_classes, file_name=False):
    
    # load an instance segmentation model pre-trained pre-trained on COCO
    if file_name:
        model =  torch.load(file_name)
    else:
        model = torchvision.models.detection.mask_rcnn.maskrcnn_resnet50_fpn(pretrained=True)
        # model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
        # get number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model
