import torch
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def OneD_mesh(Nx,Nt,Ns):
	xm = np.linspace(0,1,Nx).reshape(-1,1)
	# tm = np.linspace(0,1,Nt).reshape(-1,1)
	tm = np.linspace(0,1,Nt).reshape(-1,1)

	x,t = np.meshgrid(xm,tm)
	x = x.reshape(-1,1)
	t = t.reshape(-1,1)

	xi = np.linspace(0,1,Nx).reshape(-1,1) # x samples for initial condition
	tb = np.linspace(0,1,Nt+2)[1:Nt+1].reshape(-1,1) # time samples for boundary condition

	mu,w = np.polynomial.legendre.leggauss(Ns)
	mu = mu.reshape(-1,1)
	w = w.reshape(-1,1)*2*np.pi

	return x,t,mu,w,xi,tb

def OneD_test_mesh(Nx,Nt,Ns):
	xm = np.linspace(0,1,Nx).reshape(-1,1)
	tm = np.linspace(0,1,Nt).reshape(-1,1)

	x,t = np.meshgrid(xm,tm)
	x = x.reshape(-1,1)
	t = t.reshape(-1,1)

	mu,w = np.polynomial.legendre.leggauss(Ns)
	mu = mu.reshape(-1,1)
	w = w.reshape(-1,1)*2*np.pi

	return x,t,mu,w
