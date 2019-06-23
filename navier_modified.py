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
# variables for the modified Navier Stokes
global gamma
global gamma2

# plotting stuff
fig = plt.figure(2, figsize=(10,10))
ax1 = fig.add_subplot(221) #subplot(nrows, ncols, index)
ax2 = fig.add_subplot(222) #subplot(nrows, ncols, index)
ax3 = fig.add_subplot(223) #subplot(nrows, ncols, index)
ax4 = fig.add_subplot(224) #subplot(nrows, ncols, index)
ax1.set_title('Omega')
ax2.set_title('Psi')
ax3.set_title('u_x')
ax4.set_title('u_y')

def plotting(omega, psi, ux, uy, t):
	im1 = ax1.imshow(omega.T, interpolation="nearest", origin="lower", cmap="seismic" )
	cb1 = plt.colorbar(im1,ax=ax1)
	im2 = ax2.imshow(psi.T, interpolation="nearest", origin="lower", cmap="seismic" )
	cb2 = plt.colorbar(im2,ax=ax2)
	im3 = ax3.imshow(ux.T, interpolation="nearest", origin="lower", cmap="seismic" )
	cb3 = plt.colorbar(im3,ax=ax3)
	im4 = ax4.imshow(uy.T, interpolation="nearest", origin="lower", cmap="seismic" )
	cb4 = plt.colorbar(im4,ax=ax4)
	plt.suptitle('time = %f ' %t) # overall title
	plt.pause(0.0001)
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


# Initializing interactive plot
plt.ion()

# Parameters
scale = 1
L = 2*np.pi/scale # this will translate in the lenght of the box in the inverse fourier 
N=2**8
end_time = 1000
dt = 0.1
nu = 0.0001
gamma = 0.16
gamma2 = -0.02


# Initialize wave numbers
k_x_half = np.zeros((N, N//2 +1)) # it creates a matrix with N rows and N/2+1 columns
k_y_half = np.zeros((N, N//2 +1))

# Assigning wavenumber to 2D arrays
for i in range(N//2+1):
	for j in range(N//2+1): # j index is only for broadcasting (do operations directly with matrices)
		k_x_half[i,j] = i*scale
		k_y_half[i,j] = j*scale
# fill the negative wavenumbers, remember that they must go in the second part of the array
for i in range(N//2+1,N):
	for j in range(N//2+1):
		k_x_half[i,j] = (i - N)*scale
		k_y_half[i,j] = j*scale
		

time = 0
index = 0

# Initial conditions
omega = (np.random.rand(N,N)-0.5)*3
omega_hat = np.fft.rfft2(omega)

# defining integrating factor
k_squared = k_x_half**2 + k_y_half**2
k_matrix = -(k_squared)*(1 + gamma * k_squared + gamma*gamma2**2 *k_squared**2)
k_matrix = np.exp(k_matrix * dt/2) # to use in Runge-Kutta
k_squared[0,0] = 1 # put this to avoid divisions by zero

# Make it go
while time < end_time:
	try: # to make close everything with ctrl+c	
		# 2 order Runge Kutta
		omega_hat_temp = k_matrix * (omega_hat + 0.5*dt *rhs(omega_hat))
		omega_hat = k_matrix*(omega_hat+dt*rhs(omega_hat_temp))
		# Plotting
		if(index%10 == 0):
			# export_and_plot(np.fft.irfft2(omega_hat),time)
			omega = np.fft.irfft2(omega_hat)
			psi = np.fft.irfft2(omega_hat/k_squared)
			u_x = np.fft.irfft2(1j*k_y_half*omega_hat/k_squared)
			u_y = np.fft.irfft2(-1j*k_x_half*omega_hat/k_squared)
			plotting(omega, psi, u_x, u_y, time)
		# updating index and time	
		index = index +1
		time=time+dt
	except KeyboardInterrupt:
		plt.close('all')
		break
