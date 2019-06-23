import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import time
from matplotlib import animation

# Trying to do animations with figures (never used)

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
# PCV variables
global nu_0
global nu_1
global nu_2
global k_min
global k_max

# Right hand side (using integrating factor so no -nu*k^2*omega_hat term
def rhs(omega_hat):
	u_x_hat = (1j*k_y_half*omega_hat)/k_squared
	u_y_hat = -1j*(k_x_half*omega_hat)/k_squared
	N_hat = -1j * (k_x_half * np.fft.rfft2(np.fft.irfft2(u_x_hat)*np.fft.irfft2(omega_hat))
		+k_y_half*np.fft.rfft2(np.fft.irfft2(u_y_hat)*np.fft.irfft2(omega_hat)))
	return N_hat

# Parameters
scale = 1
L = 2*np.pi/scale # this will translate in the lenght of the box in the inverse fourier
N=2**8
end_time = 1000
nu_0 = 1e-4
nu_1 = -nu_0*5
nu_2 = nu_0*10
k_min = 33
k_max = 40

time = 0
index = 0

xy=np.meshgrid(np.arange(0,N),np.arange(0,N))
x=xy[0]
y=xy[1]
kx = np.hstack([np.arange(0, N // 2), np.arange(-N // 2, 0)])
ky = np.arange(0, N // 2+1)
k = np.meshgrid(kx, ky)
k_x_half = k[0].T # the maximum is k_x=127, then it spans the negative values
k_y_half = k[1].T
xx,yy = np.meshgrid(np.arange(0,N),np.arange(0,N)) # for initial conditions

# Defining PCV
def pcv(k):
	return np.piecewise(k,[k<k_min, k_min <= k and k<=k_max, k>k_max], [nu_0, nu_1, nu_2])
pcv = np.vectorize(pcv)
k_module = np.sqrt(k_x_half**2 + k_y_half**2)

pcv = pcv(k_module)
nu_eff=np.real(np.mean((np.fft.ifft(pcv))))

k_squared = k_x_half**2 + k_y_half**2
k_squared[k_squared==0] = 1 # put this to avoid divisions by zero

# Initial conditions
omega = np.array(np.sin(xx) + np.cos(yy), dtype=float) # cos-sin initial conditions 
# omega = (np.random.rand(N,N)-0.5) # random initial conditions
omega_hat = np.fft.rfft2(omega)
omega_hat[np.logical_or(abs(k_x_half)>=1/2*np.max(abs(k_x_half)),abs(k_y_half)>=1/2*np.max(abs(k_y_half)))]=0
u_x = np.fft.irfft2(1j*k_y_half*omega_hat/k_squared)
u_y = np.fft.irfft2(-1j*k_x_half*omega_hat/k_squared)
u_max = np.sqrt(np.max(u_x ** 2 + u_y ** 2))


# plotting stuff
fig = plt.figure(figsize=(10,5))
ax1 = fig.add_subplot(121) #subplot(nrows, ncols, index)
ax2 = fig.add_subplot(122) #subplot(nrows, ncols, index)
ax1.set_title('$\omega$')
ax2.set_title('$|u|$')
plt.suptitle(r'time = %(time)f ,  $\nu_0$ =  %(nu_0)f , $\nu_1$ =  %(nu_1)f , $\nu_2$ =  %(nu_2)f  ' %{'time' : time, 'nu_0' : nu_0, 'nu_1' : nu_1, 'nu_2' : nu_2 }) # overall title
im1 = ax1.imshow(omega.T, interpolation="nearest", origin="lower", cmap="seismic" )
cb1 = plt.colorbar(im1,ax=ax1, fraction=0.046, pad=0.04)
ims = []
# Make it go
while time < 5:
	# to make close everything with ctrl+c
	# 2 order Runge Kutta
	# Aliasing, puts to zero the high modes (is this right?)
	omega_hat[np.logical_or(abs(k_x_half)>=2/3*np.max(abs(k_x_half)),abs(k_y_half)>=2/3*np.max(abs(k_y_half)))]=0
	
	dt=(L/N/u_max)/2
	# defining integrating factor
	k_matrix = -pcv * (k_x_half ** 2 + k_y_half ** 2)
	k_matrix = np.exp(k_matrix * dt / 2)  # to use in Runge-Kutta
	omega_hat_temp = k_matrix * (omega_hat + 0.5*dt *rhs(omega_hat))
	omega_hat = k_matrix*(k_matrix*omega_hat+dt*rhs(omega_hat_temp))
	# dirty fix to correct the instability due to mismatch between negative imaginary values and positive ones
	omega_hat = np.fft.rfft2(np.fft.irfft2(omega_hat))
	# Plotting
	if(index%10 == 0): # in this way the time step is updated only every 10 iterations, so changing this can cause stability problems
		omega = np.fft.irfft2(omega_hat)
		u_x = np.fft.irfft2(1j*k_y_half*omega_hat/k_squared)
		u_y = np.fft.irfft2(-1j*k_x_half*omega_hat/k_squared)
		u_plot = np.sqrt(u_x**2 + u_y**2)
		u_max = np.sqrt(np.max(u_x ** 2 + u_y ** 2))
		#im = []		
		im1 = ax1.imshow(omega.T, interpolation="nearest", origin="lower", cmap="seismic" )
		im2 = ax2.imshow(u_plot.T, interpolation="nearest", origin="lower", cmap="seismic" )
		quiver1 = ax1.quiver(x[::10, ::10], y[::10, ::10], u_x[::10, ::10].T, u_y[::10, ::10].T)
		quiver2 = ax2.quiver(x[::10, ::10], y[::10, ::10], u_x[::10, ::10].T, u_y[::10, ::10].T)
		
		ims.append([im1, im2, quiver1, quiver2])
	index = index +1
	time=time+dt
	
anim = animation.ArtistAnimation(fig, ims, interval=200, blit=True)

anim.save('animation2.mp4', fps=10, extra_args=['-vcodec', 'libx264'])                              
plt.show()
