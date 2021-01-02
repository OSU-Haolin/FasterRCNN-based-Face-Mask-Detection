import torch
import torchvision
from PIL import Image
import cv2
from faster_rcnn_utils.engine import evaluate
from faster_rcnn_utils.AIZOODataset import AIZOODataset
from faster_rcnn_utils.transforms import get_transform
from faster_rcnn_utils import utils
from torchvision.transforms import functional as F
import os
import time
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = '0'




# test_path = './demo'
def py_cpu_nms(dets, thresh):
    """Pure Python NMS baseline."""
    x1 = dets[:, 0].astype(int)
    y1 = dets[:, 1].astype(int)
    x2 = dets[:, 2].astype(int)
    y2 = dets[:, 3].astype(int)
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


# train on the GPU or on the CPU, if a GPU is not available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
device = torch.device("cpu")
# 3 classes, background, face，face_mask
num_classes = 3
BATCH_SIZE = 1

# get the model using our helper function
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, progress=True, num_classes=num_classes, pretrained_backbone=True)
model.load_state_dict(torch.load('checkpoints_faster_rcnn/faster_rcnn_ckpt_10.pth'))
# move model to the right device
model.to(device)

# evaluate on the test dataset
# evaluate(model, data_loader_test, device=device)
# imgs=[]
img = Image.open("demo.jpg").convert("RGB")
img = img.resize((800,800))
# img = Image.open("AIZOO/val/1_Handshaking_Handshaking_1_158.jpg").convert("RGB")
img = F.to_tensor(img)
img = img.unsqueeze(0)
# img.to(device)
# imgs.append(img)
n_threads = torch.get_num_threads()
# FIXME remove this and make paste_masks_in_image run on the GPU
torch.set_num_threads(1)

model.eval()
metric_logger = utils.MetricLogger(delimiter="  ")
torch.cuda.synchronize()
model_time = time.time()
outputs = model(img)
outputs = [{k: v.to(device) for k, v in t.items()} for t in outputs]
boxes = outputs[0]['boxes'].cpu().detach().numpy().astype(int)
labels = outputs[0]['labels'].cpu().detach().numpy()
scores = outputs[0]['scores'].cpu().detach().numpy()
all = np.c_[boxes,scores]
keep = py_cpu_nms(all, 0.05)
model_time = time.time() - model_time
id2class = {0: 'No Mask', 1: 'Mask'}

##visual
img=cv2.imread("demo.jpg")
img = cv2.resize(img,(800,800))
for i in keep:
    xmin = int(all[i][0])
    ymin = int(all[i][1])
    xmax = int(all[i][2])
    ymax = int(all[i][3])

    if all[i][5] == 2:
        classid =1
        color = (255, 0, 0)

    else:
        classid = 0
        color = (0, 255, 0)
    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2)
    cv2.putText(img, "%s: %.2f" % (id2class[classid], all[i][4]), (xmin + 2, ymin - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color)
cv2.imwrite('Demo_result_fasterrcnn.jpg',img)
torch.set_num_threads(n_threads)
print(f"running time : {model_time}s")
