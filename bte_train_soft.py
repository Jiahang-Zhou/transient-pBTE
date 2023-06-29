import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader,TensorDataset,RandomSampler
import time
from model import Net

def bte_train(x,t,xi,tb,mu,w,v,tau,Nx,Nt,Ns,L,Lt,learning_rate,epochs,path,device):
	net0 = Net(3, 5, 30, 1).to(device)
	net1 = Net(2, 5, 30, 1).to(device)

	optimizer0 = optim.Adam(net0.parameters(), lr=learning_rate, betas=(0.9,0.99), eps=1e-10)
	optimizer1 = optim.Adam(net1.parameters(), lr=learning_rate, betas=(0.9,0.99), eps=1e-10)

	def init_normal(m):
		if type(m) == nn.Linear:
			nn.init.kaiming_normal_(m.weight)

	net0.apply(init_normal)
	net1.apply(init_normal)
	net0.train()
	net1.train()

	############################################################################

	def criterion(x,t,mu,w,mub,mui,tb,xi):
		x.requires_grad = True
		t.requires_grad = True

		######### Interior points ##########
		n = net0(torch.cat((x,t,mu),1))*(v*tau/L) # used to formulate non-equilibrium part
		T = net1(torch.cat((x,t),1)) # equilibrium pseudo-temperature

		n_x = torch.autograd.grad(n,x,grad_outputs=torch.ones_like(x).to(device),create_graph=True)[0]
		n_t = torch.autograd.grad(n,t,grad_outputs=torch.ones_like(t).to(device),create_graph=True)[0]
		T_x = torch.autograd.grad(T,x,grad_outputs=torch.ones_like(x).to(device),create_graph=True)[0]
		T_t = torch.autograd.grad(T,t,grad_outputs=torch.ones_like(t).to(device),create_graph=True)[0]

		sum_n = torch.matmul(n.reshape(-1,Ns),w).reshape(-1,1)/(4*np.pi)
		n = n - sum_n.repeat(1,Ns).reshape(-1,1)
		sum_n_x = torch.matmul(n_x.reshape(-1,Ns),w).reshape(-1,1)/(4*np.pi)
		n_x = n_x - sum_n_x.repeat(1,Ns).reshape(-1,1)
		sum_n_t = torch.matmul(n_t.reshape(-1,Ns),w).reshape(-1,1)/(4*np.pi)
		n_t = n_t - sum_n_t.repeat(1,Ns).reshape(-1,1)

		dq_x = torch.matmul((mu*n_x).reshape(-1,Ns),w).reshape(-1,1)
		dq_t = torch.matmul(T_t.reshape(-1,Ns),w).reshape(-1,1)

		######### Periodic boundary ##########
		r_in = torch.cat((torch.ones_like(tb),tb,mub),1)
		nr = net0(r_in)*(v*tau/L)
		sum_nr = torch.matmul(nr.reshape(-1,Ns),w).reshape(-1,1)/(4*np.pi)
		nr = nr - sum_nr.repeat(1,Ns).reshape(-1,1)
		Tr = net1(torch.cat((torch.ones_like(tb),tb),1))

		l_in = torch.cat((torch.zeros_like(tb),tb,mub),1)
		nl = net0(l_in)*(v*tau/L)
		sum_nl = torch.matmul(nl.reshape(-1,Ns),w).reshape(-1,1)/(4*np.pi)
		nl = nl - sum_nl.repeat(1,Ns).reshape(-1,1)
		Tl = net1(torch.cat((torch.zeros_like(tb),tb),1))

		######### Initial condition ##########
		t0_in = torch.cat((xi,torch.zeros_like(xi)),1)
		Ti = net1(t0_in) # initial temperature

		n0_in = torch.cat((xi,torch.zeros_like(xi),mui),1)
		ni = net0(n0_in)*(v*tau/L) # initial e
		sum_ni = torch.matmul(ni.reshape(-1,Ns),w).reshape(-1,1)/(4*np.pi)
		ni = ni - sum_ni.repeat(1,Ns).reshape(-1,1) # initial eNeq

		######### Loss ##########
		loss_1 = ((n_t+T_t)/Lt + v*mu*(n_x+T_x)/L + n/tau)
		loss_2 = (dq_t/Lt + dq_x*v/L)/4
		loss_3 = (nr + Tr - nl - Tl)
		loss_4 = ni
		loss_5 = (Ti - torch.cos(xi*2*np.pi))

		##############
		# MSE LOSS
		loss_f = nn.MSELoss()

		loss1 = loss_f(loss_1,torch.zeros_like(loss_1))
		loss2 = loss_f(loss_2,torch.zeros_like(loss_2))
		loss3 = loss_f(loss_3,torch.zeros_like(loss_3))
		loss4 = loss_f(loss_4,torch.zeros_like(loss_4))
		loss5 = loss_f(loss_5,torch.zeros_like(loss_5))

		return loss1,loss2,loss3,loss4,loss5

	###################################################################

	# Main loop
	Loss_min = 100
	Loss_list = []
	tic = time.time()

	TC = tau/3*v**2/L*2*4*np.pi
	x = torch.FloatTensor(x).repeat(1,Ns).reshape(-1,1).to(device)
	t = torch.FloatTensor(t).repeat(1,Ns).reshape(-1,1).to(device)
	mub = torch.FloatTensor(mu).repeat(Nt,1).to(device)
	mui = torch.FloatTensor(mu).repeat(Nx,1).to(device)
	mu = torch.FloatTensor(mu).repeat(Nx*Nt,1).to(device)
	w = torch.FloatTensor(w).to(device)
	tb = torch.FloatTensor(tb).repeat(1,Ns).reshape(-1,1).to(device)
	xi = torch.FloatTensor(xi).repeat(1,Ns).reshape(-1,1).to(device)
	

	for epoch in range(epochs):
		net0.zero_grad()
		net1.zero_grad()
		loss1,loss2,loss3,loss4,loss5 = criterion(x,t,mu,w,mub,mui,tb,xi)
		loss = loss1 + loss2 + loss3 + loss4+ loss5
		loss.backward()
		optimizer0.step()
		optimizer1.step()
		Loss = loss.item()
		Loss_list.append([loss1.item(),loss2.item(),loss3.item(),loss4.item(),loss5.item(),loss.item()])
		if epoch%1 == 0:
			print('Train Epoch: {}  Loss: {:.6f}  {:.6f}  {:.6f}  {:.6f}  {:.6f}'.format(epoch,loss1.item(),loss2.item(),loss3.item(),loss4.item(),loss5.item()))
			# torch.save(net0.state_dict(),path+"train_epoch"+str(epoch)+".pt")
		if Loss < Loss_min:
			torch.save(net0.state_dict(),path+"model0.pt")
			torch.save(net1.state_dict(),path+"model1.pt")
			Loss_min = Loss

	toc = time.time()
	elapseTime = toc - tic
	print ("elapse time in parallel = ", elapseTime)
	np.savetxt(path+'Loss.txt',np.array(Loss_list), fmt='%.6f')

def bte_test(x,t,mu,w,Nx,Nt,Ns,v,tau,L,Lt,index,path,device):
	net0 = Net(3, 5, 30, 1).to(device)
	net1 = Net(2, 5, 30, 1).to(device)

	net0.load_state_dict(torch.load(path+"model0.pt",map_location=device))
	net0.eval()
	net1.load_state_dict(torch.load(path+"model1.pt",map_location=device))
	net1.eval()

	
	TC = tau/3*v**2/L*2*4*np.pi
	x1 = torch.FloatTensor(x).repeat(1,Ns).reshape(-1,1).to(device)
	t1 = torch.FloatTensor(t).repeat(1,Ns).reshape(-1,1).to(device)
	mu = torch.FloatTensor(mu).repeat(Nx*Nt,1).to(device)
	w = torch.FloatTensor(w).to(device)

	x1.requires_grad = True
	t1.requires_grad = True

	tic = time.time()

	n = net0(torch.cat((x1,t1,mu),1))*(v*tau/L)
	T = net1(torch.cat((x1,t1),1))

	toc = time.time()
	elapseTime = toc - tic
	print ("elapse time in parallel = ", elapseTime)

	n_x = torch.autograd.grad(n,x1,grad_outputs=torch.ones_like(x1).to(device),create_graph=True)[0]
	T_t = torch.autograd.grad(T,t1,grad_outputs=torch.ones_like(t1).to(device),create_graph=True)[0]

	sum_n = torch.matmul(n.reshape(-1,Ns),w).reshape(-1,1)/(4*np.pi)
	n = n - sum_n.repeat(1,Ns).reshape(-1,1)
	sum_n_x = torch.matmul(n_x.reshape(-1,Ns),w).reshape(-1,1)/(4*np.pi)
	n_x = n_x - sum_n_x.repeat(1,Ns).reshape(-1,1)

	q_x = torch.matmul(n_x.reshape(-1,Ns),w*mu[0:Ns].reshape(-1,1)).reshape(-1,1)
	q_t = torch.matmul(T_t.reshape(-1,Ns),w).reshape(-1,1)

	q = (q_t/Lt + q_x*v/L).cpu().data.numpy()/(4*np.pi)
	T = T.reshape(-1,Ns)[:,0].cpu().data.numpy()

	np.savez(str(int(index))+'gray_1d_s',x = x,t = t,T = T,q = q)
