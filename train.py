import torch
from dataloader import TripletData
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision import models
import tqdm
from loss import TripletLoss


# Transforms
train_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

#specify the path
PATH_TRAIN = ''
PATH_VALID = ''

# Datasets and Dataloaders
train_data = TripletData(PATH_TRAIN, train_transforms)
val_data = TripletData(PATH_VALID, val_transforms)

train_loader = torch.utils.data.DataLoader(dataset = train_data, batch_size=32, shuffle=True, num_workers=4)
val_loader = torch.utils.data.DataLoader(dataset = val_data, batch_size=32, shuffle=False, num_workers=4)

device = 'cuda'

# Our base model
model = models.resnet18().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
triplet_loss = TripletLoss()

epochs = 100
# Training
for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0
    for data in tqdm(train_loader):
        optimizer.zero_grad()
        x1, x2, x3 = data
        e1 = model(x1.to(device))
        e2 = model(x2.to(device))
        e3 = model(x3.to(device))

        loss = triplet_loss(e1, e2, e3)
        epoch_loss += loss
        loss.backward()
        optimizer.step()
    print("Train Loss: {}".format(epoch_loss.item()))