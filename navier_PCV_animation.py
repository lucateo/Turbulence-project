import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import time
from matplotlib import animation

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


# Parameters
scale = 1
L = 2*np.pi/scale # this will translate in the lenght of the box in the inverse fourier
N=2**8
end_time = 1000000000
nu_0 = 1e-4
nu_1 = -nu_0*1
nu_2 = nu_0*1
k_min = 10
k_max = 10

# Right hand side (using integrating factor so no -nu*k^2*omega_hat term
def rhs(omega_hat):
	u_x_hat = (1j*k_y_half*omega_hat)/k_squared
	u_y_hat = -1j*(k_x_half*omega_hat)/k_squared
	N_hat = -1j * (k_x_half * np.fft.rfft2(np.fft.irfft2(u_x_hat)*np.fft.irfft2(omega_hat))
		+k_y_half*np.fft.rfft2(np.fft.irfft2(u_y_hat)*np.fft.irfft2(omega_hat)))
	return N_hat

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
#omega = np.array(np.sin(xx) + np.cos(yy), dtype=float) # cos-sin initial conditions 
omega = (np.random.rand(N,N)-0.5) # random initial conditions
omega_hat = np.fft.rfft2(omega)
u_x = np.fft.irfft2(1j*k_y_half*omega_hat/k_squared)
u_y = np.fft.irfft2(-1j*k_x_half*omega_hat/k_squared)
u_max = np.sqrt(np.max(u_x ** 2 + u_y ** 2))
u_plot = np.sqrt(u_x**2 + u_y**2)

# plotting stuff
fig = plt.figure(figsize=(10,5))
ax1 = fig.add_subplot(121) #subplot(nrows, ncols, index)
ax2 = fig.add_subplot(122) #subplot(nrows, ncols, index)
ax1.set_title('$\omega$')
ax2.set_title('$|u|$')
#plt.suptitle(r'time = %(time)f ,  $\nu_0$ =  %(nu_0)f , $\nu_1$ =  %(nu_1)f , $\nu_2$ =  %(nu_2)f  ' %{'time' : time, 'nu_0' : nu_0, 'nu_1' : nu_1, 'nu_2' : nu_2 }) # overall title
plt.suptitle(r'time = %(time)f , $k_f $ = %(k_min)d, $\nu_0$ =  %(nu_0)f , $\nu_1$ =  %(nu_1)f , $\nu_2$ =  %(nu_2)f  ' %{'time' : time, 'k_min' : k_min, 'nu_0' : nu_0, 'nu_1' : nu_1, 'nu_2' : nu_2 }) # overall title
im1 = ax1.imshow(omega.T,interpolation="nearest", origin="lower", cmap="seismic", animated=True)
im2 = ax2.imshow(u_plot.T, interpolation="nearest", origin="lower", cmap="seismic", animated=True )
cb1 = plt.colorbar(im1,ax=ax1, fraction=0.046, pad=0.04) # the second is to fit the colorbar to the graph
cb2 = plt.colorbar(im2,ax=ax2, fraction=0.046, pad=0.04)

# Make it go
def update(*args):
	global omega_hat
	global u_max, time, index, cb1, cb2
	for i in range(3):
		omega_hat[np.logical_or(abs(k_x_half)>=2/3*np.max(abs(k_x_half)),abs(k_y_half)>=2/3*np.max(abs(k_y_half)))]=0
		dt=(L/N/u_max)/2
		# defining integrating factor
		k_matrix = -pcv * (k_x_half ** 2 + k_y_half ** 2)
		k_matrix = np.exp(k_matrix * dt / 2)  # to use in Runge-Kutta
		omega_hat_temp = k_matrix * (omega_hat + 0.5*dt *rhs(omega_hat))
		omega_hat = k_matrix*(k_matrix*omega_hat+dt*rhs(omega_hat_temp))
		# fixing
		for i in range(1,int(len(omega_hat)/2)+1):
			omega_hat[-i,0]=np.conj(omega_hat[i,0])
		u_x = np.fft.irfft2(1j * k_y_half * omega_hat / k_squared)
		u_y = np.fft.irfft2(-1j * k_x_half * omega_hat / k_squared)
		u_max = np.sqrt(np.max(u_x ** 2 + u_y ** 2))
		index = index +1
		time=time+dt
	# Plotting (in animation)
	omega = np.fft.irfft2(omega_hat)
	u_plot = np.sqrt(u_x**2 + u_y**2)
	ax1.clear()
	cb1.remove()
	im1 = ax1.imshow(omega.T,interpolation="nearest", origin="lower", cmap="seismic", animated=True)
	cb1 = plt.colorbar(im1,ax=ax1, fraction=0.046, pad=0.04)
	ax2.clear()
	cb2.remove()
	im2 = ax2.imshow(u_plot.T,interpolation="nearest", origin="lower", cmap="seismic", animated=True)
	cb2 = plt.colorbar(im2,ax=ax2, fraction=0.046, pad=0.04)
	# plt.suptitle(r'time = %(time)f ,  $\nu_0$ =  %(nu_0)f , $\nu_1$ =  %(nu_1)f , $\nu_2$ =  %(nu_2)f  ' %{'time' : time, 'nu_0' : nu_0, 'nu_1' : nu_1, 'nu_2' : nu_2 }) # overall title
	plt.suptitle(r'time = %(time)f , $k_f $ = %(k_min)d, $\nu_0$ =  %(nu_0)f , $\nu_1$ =  %(nu_1)f , $\nu_2$ =  %(nu_2)f  ' %{'time' : time, 'k_min' : k_min, 'nu_0' : nu_0, 'nu_1' : nu_1, 'nu_2' : nu_2 }) # overall title
	ax1.set_title('$\omega$')
	ax2.set_title('$|u|$')
	ax1.quiver(x[::10, ::10], y[::10, ::10], u_x[::10, ::10].T, u_y[::10, ::10].T)
	ax2.quiver(x[::10, ::10], y[::10, ::10], u_x[::10, ::10].T, u_y[::10, ::10].T)
	if(index%50 == 0): # to understand at which point it arrived and stop with cntr+c
		print(time)
		print(index)

# Set up formatting for the movie files (requires ffmpeg installed)
Writer = animation.writers['ffmpeg']
writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=1800)

# without blit option (that updates only stuff that changes, usually recommended) I don't have to return artist argument from update function	
anim = animation.FuncAnimation(fig, update, frames = 2000000, interval=20)
anim.save('pattern_kf_20_boh.mp4', writer=writer)
#anim.save('basic_animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])                              
#plt.show()
