import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset,RandomSampler
import time
import array
from pathlib import Path
import sys
import os
# import sobol_seq
from bte_train_soft import bte_train, bte_test
from mesh_gen import OneD_mesh, OneD_test_mesh

epochs = 8000
path = "./train/"
eta = 0.6 # eta = 2*pi*MFP/L = 2*pi*Kn
index = 4
v,tau = 1,1
L = 2*np.pi*v*tau/eta
Lt = 3*tau # length of time range

############################################
Nx = 60 
Nt = 60
Ns = 16 # number of quadrature points   

x,t,mu,w,xi,tb = OneD_mesh(Nx,Nt,Ns)
# mesh = sobol_seq.i4_sobol_generate(2, N)
# x = mesh[:,0].reshape(-1,1)
# t = mesh[:,1].reshape(-1,1)

############################################
learning_rate = 4e-3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

bte_train(x,t,xi,tb,mu,w,v,tau,Nx,Nt,Ns,L,Lt,learning_rate,epochs,path,device)

############################################
Nx = 81
Nt = 81
x,t,mu,w = OneD_test_mesh(Nx,Nt,Ns)

bte_test(x,t,mu,w,Nx,Nt,Ns,v,tau,L,Lt,index,path,device)


