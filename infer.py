#!pip install faiss-gpu
import faiss
import torch
from torchvision import transforms, utils
import glob
import os
import numpy as np
from PIL import Image

faiss_index = faiss.IndexFlatL2(1000)  # build the index


PATH_TRAIN = ''
PATH_TEST = ''
MODEl_PATH = ''
#load model
model = torch.load(MODEl_PATH)


val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
# storing the image representations
im_indices = []
with torch.no_grad():
    for f in glob.glob(os.path.join(PATH_TRAIN, '*/*')):
        im = Image.open(f)
        im = im.resize((224, 224))
        im = torch.tensor([val_transforms(im).numpy()]).cuda()

        preds = model(im)
        preds = np.array([preds[0].cpu().numpy()])
        faiss_index.add(preds)  # add the representation to index
        im_indices.append(f)  # store the image name to find it later on

# Retrieval with a query image
with torch.no_grad():
    for f in os.listdir(PATH_TEST):
        # query/test image
        im = Image.open(os.path.join(PATH_TEST, f))
        im = im.resize((224, 224))
        im = torch.tensor([val_transforms(im).numpy()]).cuda()

        test_embed = model(im).cpu().numpy()
        _, I = faiss_index.search(test_embed, 5)
        print("Retrieved Image: {}".format(im_indices[I[0][0]]))