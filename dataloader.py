import time

import numpy as np
import torch
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
#return MNIST dataloader
def mnist_dataloader(batch_size=256,train=True):

    dataloader=DataLoader(
    datasets.MNIST('./data/mnist',train=train,download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
                   ])),
    batch_size=batch_size,shuffle=True)

    return dataloader

def svhn_dataloader(batch_size=4,train=True):
    dataloader = DataLoader(
        datasets.SVHN('./data/SVHN', split=('train' if train else 'test'), download=True,
                       transform=transforms.Compose([
                           transforms.Resize((28,28)),
                           transforms.Grayscale(),
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                       ])),
        batch_size=batch_size, shuffle=False)

    return dataloader


def sample_data():
    dataset=datasets.MNIST('./data/mnist',train=True,download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
                   ]))
    n=len(dataset)

    X=torch.Tensor(n,1,28,28)
    Y=torch.LongTensor(n)

    inds=torch.randperm(len(dataset))
    for i,index in enumerate(inds):
        x,y=dataset[index]
        X[i]=x
        Y[i]=y
    return X,Y


def create_target_samples(n=1):
    dataset=datasets.SVHN('./data/SVHN', split='train', download=False,
                       transform=transforms.Compose([
                           transforms.Resize((28,28)),
                           transforms.Grayscale(),
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                       ]))
    X,Y=[],[]
    classes=10*[n]

    i=0
    while True:
        if len(X)==n*10:
            break
        x,y=dataset[i]
        if classes[y]>0:
            X.append(x)
            Y.append(y)
            classes[y]-=1
        i+=1

    assert (len(X)==n*10)
    return torch.stack(X,dim=0),torch.from_numpy(np.array(Y))
"""
G1: a pair of pic comes from same domain ,same class
G3: a pair of pic comes from same domain, different classes

G2: a pair of pic comes from different domain,same class
G4: a pair of pic comes from different domain, different classes
"""
def create_groups(X_s,Y_s,X_t,Y_t,seed=1):
    #change seed so every time wo get group data will different in source domain,but in target domain, data not change
    torch.manual_seed(1 + seed)
    torch.cuda.manual_seed(1 + seed)


    n=X_t.shape[0] #10*shot


    #shuffle order
    classes = torch.unique(Y_t)
    classes=classes[torch.randperm(len(classes))]


    class_num=classes.shape[0]
    shot=n//class_num



    def s_idxs(c):
        idx=torch.nonzero(Y_s.eq(int(c)))

        return idx[torch.randperm(len(idx))][:shot*2].squeeze()
    def t_idxs(c):
        return torch.nonzero(Y_t.eq(int(c)))[:shot].squeeze()

    source_idxs = list(map(s_idxs, classes))
    target_idxs = list(map(t_idxs, classes))


    source_matrix=torch.stack(source_idxs)

    target_matrix=torch.stack(target_idxs)


    G1, G2, G3, G4 = [], [] , [] , []
    Y1, Y2 , Y3 , Y4 = [], [] ,[] ,[]


    for i in range(10):
        for j in range(shot):
            G1.append((X_s[source_matrix[i][j*2]],X_s[source_matrix[i][j*2+1]]))
            Y1.append((Y_s[source_matrix[i][j*2]],Y_s[source_matrix[i][j*2+1]]))
            G2.append((X_s[source_matrix[i][j]],X_t[target_matrix[i][j]]))
            Y2.append((Y_s[source_matrix[i][j]],Y_t[target_matrix[i][j]]))
            G3.append((X_s[source_matrix[i%10][j]],X_s[source_matrix[(i+1)%10][j]]))
            Y3.append((Y_s[source_matrix[i % 10][j]], Y_s[source_matrix[(i + 1) % 10][j]]))
            G4.append((X_s[source_matrix[i%10][j]],X_t[target_matrix[(i+1)%10][j]]))
            Y4.append((Y_s[source_matrix[i % 10][j]], Y_t[target_matrix[(i + 1) % 10][j]]))



    groups=[G1,G2,G3,G4]
    groups_y=[Y1,Y2,Y3,Y4]


    #make sure we sampled enough samples
    for g in groups:
        assert(len(g)==n)
    return groups,groups_y




def sample_groups(X_s,Y_s,X_t,Y_t,seed=1):


    print("Sampling groups")
    return create_groups(X_s,Y_s,X_t,Y_t,seed=seed)


