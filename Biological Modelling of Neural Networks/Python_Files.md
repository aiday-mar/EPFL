**Week 1**

from numpy import * 
from pylab import * 

def LIF(t,I): 
   """ solve the linear integrate-and-fire using Euler's method Inputs: 
       t - equally spaced time bins e.g. arange(0,100,0.1) 
       I - current input to LIF, same shape as t 
   """ 
   # simple error check 
   if I.shape!=t.shape: 
      # here in the type error you have a phrase which describes the error
      raise TypeError, 'require I.shape == t.shape' 
      L = len(t) 
      dt = t[1]-t[0] 

   # allocate array for voltage 
   # meaning it is a matrix of zeroes with shape that of I
   v = zeros(I.shape) 

   # set initial condition 
   v[0] = 0.0 

   # simple but slow python loop 
   for i in range(1,L): 
      dvdt = -v[i-1] + I[i-1] 
      v[i] = v[i-1] + dvdt*dt  # FORWARD EULER 
   
      # reset voltage (spike) if v >= 1 (threshold) 
      if v[i] >= 1: 
         v[i] = 0.0 
   return v 

def LIF_Step(tstart=0, tend=100, I_tstart = 20, I_tend = 70, I_amp=1.005, dt=0.1): 

   """ run the LIF for a step current """ 
   # make time bins 
   t = arange(tstart,tend,dt) 

   # make current array for each time bin 
   I = zeros(t.shape) 
   
   # find array index of I start and stop times 
   index_start = searchsorted(t,I_tstart) 
   index_end = searchsorted(t,I_tend) 

   # assign amplitude to that range 
   I[index_start:index_end] = I_amp 

   # run the integrator 
   v = LIF(t,I) 

   # open new figure 
   figure() 

   # get help on subplot at the ipython prompt: 
   # In [x]: ? subplot 

   subplot(211) 
   plot(t,v,lw=2) 

   # red dashes
   plot([tstart,tend],[1,1],'r--',label='threshold',lw=2) 
   xlabel('t') 
   ylabel('v') 
   
   subplot(212)
   plot(t,I,lw=2) 
   xlabel('t') 
   ylabel('I') 

# you have default values in the parameters 
def LIF_Sinus(I_freq = 0.1, tstart=0,tend=100,dt=0.1, I_offset=0.5,I_amp=0.4): 

   """ run the LIF for a sinusoid current Inputs: 
   I_freq - frequency of current sinusoid in cycles per unit time """ 

   # make time bins 
   t = arange(tstart,tend,dt) 

   # make current array for each time bin 
   I = I_amp*sin(2.0*pi*I_freq*t)+I_offset 

   # run the integrator 
   v = LIF(t,I) 
 
   # open new figure 
   figure() 

   # get help on subplot at the ipython prompt: 
   # In [x]: ? subplot 
   plot(t,v,label=r'$v(t)$',lw=2) 
   xlabel('t') 
   ylabel('v') 

   # equilibrium (v*) 
   # set dvdt = 0 
   # since tau=1, v_reset=0.0, v*(t) = I(t) 
   plot(t,I,'r:',label=r'$v^\ast(t)$',lw=2) 
   plot([tstart,tend],[1,1],'r--',label='threshold',lw=2) 
   legend()
