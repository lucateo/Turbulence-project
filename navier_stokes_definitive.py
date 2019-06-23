import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import time

# Defining global variables to make them work also inside functions
global scale
global L
global N
global alpha
global end_time
global dt
global k_x_half
global k_y_half
global k_squared
global nu

# plotting stuff
fig = plt.figure(2, figsize=(10,10))
ax1 = fig.add_subplot(321) #subplot(nrows, ncols, index)
ax2 = fig.add_subplot(322) #subplot(nrows, ncols, index)
ax3 = fig.add_subplot(323) #subplot(nrows, ncols, index)
ax4 = fig.add_subplot(324) #subplot(nrows, ncols, index)
ax5 = fig.add_subplot(325)


def plotting(omega, psi, ux, uy, epsilon, t):
	ax1.set_title('Omega')
	ax2.set_title('Psi')
	ax3.set_title('u_x')
	ax4.set_title('u_y')
	ax5.set_title('$\epsilon$')
	im1 = ax1.imshow(omega.T, interpolation="nearest", origin="lower", cmap="seismic" )
	cb1 = plt.colorbar(im1,ax=ax1)
	ax1.quiver(x[::10, ::10], y[::10, ::10], ux[::10, ::10].T, uy[::10, ::10].T, scale=0.5) # to plot arrows, the :: is to plot every tot (in this case every 10)
	im2 = ax2.imshow(psi.T, interpolation="nearest", origin="lower", cmap="seismic" )
	cb2 = plt.colorbar(im2,ax=ax2)
	ax2.quiver(x[::20,::20],y[::20,::20],ux[::20,::20].T,uy[::20,::20].T,scale=0.3)
	im3 = ax3.imshow(ux.T, interpolation="nearest", origin="lower", cmap="seismic" )
	cb3 = plt.colorbar(im3,ax=ax3)
	im4 = ax4.imshow(uy.T, interpolation="nearest", origin="lower", cmap="seismic" )
	cb4 = plt.colorbar(im4,ax=ax4)
	im5 = ax5.plot(t,epsilon,'.',color='b')
	plt.suptitle('time = %f ' %t) # overall title
	plt.pause(0.0001)
	ax1.clear()
	ax2.clear()
	ax3.clear()
	ax4.clear()
	cb1.remove()
	cb2.remove()
	cb3.remove()
	cb4.remove()


# Right hand side (using integrating factor so no -nu*k^2*omega_hat term
def rhs(omega_hat):
	u_x_hat = (1j*k_y_half*omega_hat)/k_squared
	u_y_hat = -1j*(k_x_half*omega_hat)/k_squared
	N_hat = -1j * (k_x_half * np.fft.rfft2(np.fft.irfft2(u_x_hat)*np.fft.irfft2(omega_hat))
		+k_y_half*np.fft.rfft2(np.fft.irfft2(u_y_hat)*np.fft.irfft2(omega_hat)))
	return N_hat
def enstrophy(omega): # omega is a 2D array
	omega_bl=np.array(omega)
	omega_bl[1:-1,1:-1]=0 # ???
	return np.sum(omega_bl**2)

# Initializing interactive plot
plt.ion()

# Parameters
scale = 1
L = 2*np.pi/scale # this will translate in the lenght of the box in the inverse fourier
N=2**8
end_time = 1000
dt = 0.1
nu = 0.0001

# Defining real space and Fourier space
xy=np.meshgrid(np.arange(0,N),np.arange(0,N))
x=xy[0]
y=xy[1]
kx = np.hstack([np.arange(0, N // 2), np.arange(-N // 2, 0)])
ky = np.arange(0, N // 2+1)
k = np.meshgrid(kx, ky)
k_x_half = k[0].T
k_y_half = k[1].T

time = 0
index = 0

# Initial conditions
omega = (np.random.rand(N,N)-0.5)*3
omega_hat = np.fft.rfft2(omega)

# defining integrating factor
k_matrix = -nu*(k_x_half**2 + k_y_half**2)
k_squared = k_x_half**2 + k_y_half**2
k_squared[0,0] = 1 # put this to avoid divisions by zero

# Make it go
while time < end_time:
	try: # to make close everything with ctrl+c
		# 2 order Runge Kutta
		if time!=0:
			dt=nu/(L/N*u_max)
		k_matrix = np.exp(k_matrix * dt/2)
		omega_hat_temp = k_matrix * (omega_hat + 0.5*dt *rhs(omega_hat))
		omega_hat = k_matrix*(omega_hat+dt*rhs(omega_hat_temp))
		# Plotting
		if(index%10 == 0):
			omega = np.fft.irfft2(omega_hat)
			epsilon = enstrophy(omega)
			psi = np.fft.irfft2(omega_hat/k_squared)
			u_x = np.fft.irfft2(1j*k_y_half*omega_hat/k_squared)
			u_y = np.fft.irfft2(-1j*k_x_half*omega_hat/k_squared)
			u_max = np.sqrt(np.max(u_x ** 2 + u_y ** 2))
			plotting(omega, psi, u_x, u_y, epsilon, time)
		# updating index and time
		index = index +1
		time=time+dt
	except KeyboardInterrupt:
		plt.close('all')
		break
