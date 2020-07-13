**Week 1**

```
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
```

**Week 2**

```
from pylab import *
from numpy import *

def HH_Step(tstart=0, tend=100, Step_tstart = 20, Step_tend = 70, I_amp=7, dt=0.01):
    """
    DEFINITION
    RunS the Hodgkin Huxley model for a step current.
    
    INPUT
    tstart: starting time (ms)
    tend: ending time (ms)
    Step_tstart: time at which the step current begins (ms)
    Step_tend: time  at which the step current ends (ms)
    I_amp: magnitude of the current step (uA/cm^2)
    dt: integration timestep
    
    OUTPUT
    graph with three panels: 1) voltage trace, 2) gating variables m, n, and h, 3) current injected.
    
    ALGORITHM
    uses Forward-Euler numerical integration in HH_ForwardEuler
    
    -R.Naud 02.2009.
    """

    # make time bins, from the start time to the end time with the little step in time dt
    t = arange(tstart,tend,dt)  
    # make current array for each time bin, this is an array of just zeroes of the appropriate shape
    I = zeros(t.shape)

    # find array index of I start and stop times
    index_start = searchsorted(t,Step_tstart)
    index_end = searchsorted(t,Step_tend)
    # assign amplitude to that range, the amplitude here is then set to I_amp
    I[index_start:index_end] = I_amp

    # run the integrator, where you have the time vector here t and the array of zeroes which is here I
    [v, m, n, h] = HH_ForwardEuler(t,I)

    # open new figure and plot
    figure()
    # plot voltage time series
    subplot(311)
    plot(t,v,lw=2)
    xlabel('t (ms)')
    ylabel('v (mV)')
    # plot activation and inactivation variables
    subplot(312)
    plot(t,m,'k',lw=2)
    plot(t,h,'r',lw=2)
    plot(t,n,'b',lw=2)
    xlabel('t (ms)')
    ylabel('act./inact.')
    legend(('m','h','n'))
    # plot current
    subplot(313)
    plot(t,I,lw=2)
    axis((tstart, tend, 0, I_amp*1.1))
    xlabel('t (ms)')
    ylabel('I (pA)')

def HH_Ramp(tstart=0, tend=200, Ramp_tstart = 30, Ramp_tend = 170, FinalAmp=20, dt=0.01):
    """
    DEFINITION
    RunS the Hodgkin Huxley model for a step current.
    
    INPUT
    tstart: starting time (ms)
    tend: ending time (ms)
    Ramp_tstart: time at which the ramp current begins (ms)
    Ramp_tend: time  at which the ramp current ends (ms)
    FinalAmp: magnitude of the current at the end of the ramp (uA/cm^2)
    dt: integration timestep
    
    OUTPUT
    graph with three panels: 1) voltage trace, 2) gating variables m, n, and h, 3) current injected.
    
    ALGORITHM
    uses Forward-Euler numerical integration in HH_ForwardEuler
    
    -R.Naud 02.2009.
    """

    # make time bins
    t = arange(tstart,tend,dt)  
    # make current array for each time bin
    I = zeros(t.shape)

    # find array index of I start and stop times
    index_start = searchsorted(t,Ramp_tstart)
    index_end = searchsorted(t,Ramp_tend)
    # assign ramp in that range
    I[index_start:index_end] = arange(0,index_end-index_start,1.0)/(index_end-index_start)*FinalAmp   

    # run the integrator
    [v, m, n, h] = HH_ForwardEuler(t,I)

    # open new figure and plot
    figure()
    # plot voltage time series
    subplot(311)
    plot(t,v,lw=2)
    xlabel('t (ms)')
    ylabel('v (mV)')
    # plot activation and inactivation variables
    subplot(312)
    plot(t,m,'k',lw=2)
    plot(t,h,'r',lw=2)
    plot(t,n,'b',lw=2)
    xlabel('t (ms)')
    ylabel('act./inact.')
    legend(('m','h','n'))
    # plot current
    subplot(313)
    plot(t,I,lw=2)
    axis((tstart, tend, 0, FinalAmp*1.1))
    xlabel('t (ms)')
    ylabel('I (pA)')


def HH_Sinus(I_freq = 0.01, tstart=0,tend=600,dt=0.01, I_offset=0.5,I_amp=7):
    """
    DEFINITION
    Runs the Hodgkin Huxley model for a step current.
    
    INPUT
    tstart: starting time (ms)
    tend: ending time (ms)
    I_freq : frequency of stimulating current (kHz)
    I_offset: offset (uA/cm^2)
    I_amp: amplitude of the sine (uA/cm^2)
    dt: integration timestep
    
    OUTPUT
    graph superposing voltage trace, and current injected
    
    ALGORITHM
    uses Forward-Euler numerical integration in HH_ForwardEuler
    
    -R.Naud 02.2009.
    """
    # make time bins
    t = arange(tstart,tend,dt)
    # make current array for each time bin
    # we add here an additional constant to each coordinate
    I = I_amp*sin(2.0*pi*I_freq*t)+I_offset
    # run the integrator
    [v,m,h,n] = HH_ForwardEuler(t,I)
    # plot voltage and current 
    figure()
    plot(t,v,label=r'$v(t)$',lw=2)
    xlabel('t')
    ylabel('v')
    plot(t,I,'r:',label=r'$v^\ast(t)$',lw=2)
    legend(('v','I'))
    

def HH_ForwardEuler(t,I):
    """
    solve the original Hodgkin-Huxley model
    using Euler's method

    Inputs:
       t   - equally spaced time bins (ms)
             e.g. arange(0,100,0.1)
             
       I   - current input to LIF,
             same shape as t 
    OUPTUT
    [v,m, h, n]
    the timeseries of the voltage (v, in mV), sodium activation (m), sodium deactivation (h) 
    and potassium activation (n)        
    
    ALGORITHM
    uses the equations and parameters given in 'Spiking Neuron Models' p36.  These parameters are based
    on the voltage scale where the resting potential is zero.
    """
    
    # simple error check
    if I.shape!=t.shape:
        # here you essentially raise an exception and then you send out a string which says that a condition was not fulfilled 
        raise TypeError, 'require I.shape == t.shape'
    
    # setting Parameter Values
    # in mV, uF/cm2, mS/cm2
    Params = {'gNa': 120, 
          'C': 1, 
          'ENa': 115, 
          'EK':-12, 
          'EL': 10.6,
          'gK': 36, 
          'gL':0.3}
    
    
    L = len(t)
    # it's the difference between the first two coordinates of the vector t 
    dt = t[1]-t[0]

    # allocate array for voltage and gating variables
    v = zeros(I.shape) # this sets initial condition of v to be zero as required
    m = zeros(I.shape)
    h = zeros(I.shape)
    n = zeros(I.shape)
    
    # let initial value be m_inf(initial voltage), etc.
    m[0] = dmdt(0,v[0])/(dmdt(0,v[0])-dmdt(1,v[0]))
    n[0] = dndt(0,v[0])/(dndt(0,v[0])-dndt(1,v[0]))
    h[0] = dhdt(0,v[0])/(dhdt(0,v[0])-dhdt(1,v[0]))
    
    
    for i in range(1,len(t)):
        # forward euler step
        v[i] = v[i-1] + dudt(v[i-1],m[i-1], h[i-1], n[i-1], I[i-1], Params)*dt
        m[i] = m[i-1] + dmdt(m[i-1], v[i-1])*dt
        n[i] = n[i-1] + dndt(n[i-1], v[i-1])*dt
        h[i] = h[i-1] + dhdt(h[i-1], v[i-1])*dt
    
    return [v, m, n, h]



# define derivatives; factor 1e3 needed for t in ms
def dudt(u,m,h,n,I,Params):
    """
    DEF
    This is Kirchoff law for a neuron membrane with sodium and potassium ion channels.
    INPUTS
    (state variables)
    u: voltage (float)
    m: sodium activation (float)
    h: sodium inactivation (float)
    n: potassium activation (float)
    (parameters)
    I: current (float)
    Params: parameters for maximal conductance 
    and reversal potentials (dictionary with keys ENa, gNa, gK, EK, gL, EL)
    """
    dudt = 1/Params['C']*(-Params['gNa']*m**3*h*(u-Params['ENa']) 
                          - Params['gK']*n**4*(u-Params['EK'])
                          -Params['gL']*(u-Params['EL']) + I )
    return dudt

# m activation variable related function
def dmdt(m,u):
    """
    dm/dt in units of 1/ms
    """
    # as for original Hodgkin and Huxley model (see equations given in Gerstner 2002, p.36)
    alpha = (2.5-0.1*u)/(exp(2.5-0.1*u)-1)
    beta = 4*exp(-u/18)
    dmdt = (alpha*(1-m) - beta*m) 
    return dmdt
def dndt(n,u):
    # as for original Hodgkin and Huxley model (see equations given in Gerstner 2002, p.36)
    alpha = (0.1-0.01*u)/(exp(1-0.1*u)-1)
    beta = 0.125*exp(-u/80)
    dndt = (alpha*(1-n) - beta*n)
    return dndt
def dhdt(h,u):
    # as for original Hodgkin and Huxley model (see equations given in Gerstner 2002, p.36)
    alpha = 0.07*exp(-u/20) 
    beta = 1/(exp(3-0.1*u)+1)
    dhdt = (alpha*(1-h) - beta*h)
    return dhdt
```

**Week 3**

```
# in a pylab environment
# >> execfile('demo.py')
# here we are importing the class from another python file
import TypeX

# then we are calling the two methods below 
ion()
figure()

I = 0.0
dt = 0.5

# v nullcline
# must be like a lambda function where we specify the parameters v and I that will be needed in it and
# then we have the actual calculation
null_v = lambda v,I: v - (v**3)/3 + I
# this is a vector which has 0.1 as a space between points 
v = arange(-3.0,3.0,0.1)

# here we have the name of the parameter and the numerical value assigned to the parameter 
Params = {'a': 1.25, 'tau': 15.6}

# meaning here we are accessing this sort of dictionary and we use the 'a' string to access 
null_w = lambda v,I: Params['a']*(v+0.7)

subplot(211)
# where we also specify the specific parameters that will be used in the lambda function
# there is a parameter specified as well as a corresponding label
plot(v,null_w(v,I),'g:',lw=2,label='dw/dt=0')


axis([-2.5,2.5,-2.0,3.0])


# initial v,w
vw = (0.0,0.0)
vhist = [0.0]
whist = [0.0]
thist = [0.0]

# v,w at next time step
def iterate(vw,I):
    # to specify some global parameters you need to write the keyword in front 
    global Params,dt
    
    v = vw[0]
    w = vw[1]
    v_next = v + TypeX.dudt(v,w, I)*dt
    w_next = w + TypeX.dwdt(v, w, Params)*dt
    
    # you can return two variables at the same time 
    return (v_next,w_next)


l1, = plot(v,null_v(v,I),'r:',lw=2,label='dv/dt=0',scalex=False,scaley=False)
l2, = plot([vw[0]],[vw[1]],'ko',lw=2,scalex=False,scaley=False,label='v,w')

# the below must be like the format function in hava meaning you write %.2f and then instead of this
# you can write the actual value of variable I instead 
tex1 = figtext(0.5,0.8,'I=%.2f' % I,size=26)

legend()

l4, = plot(vhist,whist,'k-',lw=1,alpha=0.5,scalex=False,scaley=False)

xlabel('v')
ylabel('w')

subplot(212)

l3, = plot(thist,vhist,'b-',scalex=False,scaley=False,lw=2)

# the labels are given by strings 
xlabel('t [ms]')
ylabel('v')

axis([0.0,200.0,-2.0,2.0])

# meaning that the arange function returns a vector and you can make a list out of this vector 
# what does [0.0] mean though ? like an array with one zero entry 
for I in [0.0]*100+list(arange(0.0,0.5,0.01))+[0.5]*200:
    vw = iterate(vw,I)
    vhist.append(vw[0])
    whist.append(vw[1])
    thist.append(thist[-1]+dt)
    
    l2.set_data([vw[0]],[vw[1]])

    l4.set_data(vhist,whist)

    # recalc nullcline

    l1.set_ydata(null_v(v,I))

    tex1.set_text('I=%.2f' % I)

    l3.set_data(thist,vhist)

    draw()
```

Now we also have the following code : 

```
# where we have v_th is a boolean and now we have v_above_th is maybe a vector of booleans 
v_above_th = v>v_th
# Return the indices of the elements that are non-zero.
# Returns a tuple of arrays, one for each dimension of a, containing the indices of the non-zero elements in that dimension. 
# The values in a are always tested and returned in row-major, C-style order.
# select from index 1 to the end, where we have that the value is actually true
# select from the beginning to the second last where false 
#  In Python "and" is a logical operator and "&" is a bitwise operator (compare bit by bit)
numpy.nonzero((v_above_th[:-1]==False)&(v_above_th[1:]==True))
```

Suppose we have the following array :

```
x = np.array([[3, 0, 0], [0, 4, 0], [5, 6, 0]])
np.nonzero(x)
```
The above returns : `(array([0, 1, 2, 2]), array([0, 1, 0, 1]))`.

**Week 4**

```
from pylab import *

# where we have some parameters with default values 
def make_cloud(size=10000,ratio=1,angle=0):
    """
    DEFINITION
    builds an oriented elliptic (or circular) gaussian cloud of 2D points
    
    INPUT
    size: the number of points
    ratio: (std along the short axis) / (std along the long axis)
    angle: the rotation angle in degrees
    
    -G.Hennequin 02.2009.
    """
    
    # we have the value of the parameter ratio
    if ratio>1: ratio=1/ratio
    
    x = randn(size,1)
    y = ratio * randn(size,1)
    # you can concatenate the vector with this new scalar
    z = concatenate((x,y),1)
    radangle = (180 - angle) * pi / 180
    # which is a matrix with 4 elements 
    transfo = [[cos(radangle),sin(radangle)],[-sin(radangle),cos(radangle)]]
    # we take the dot product and then the transpose of the matrix 
    data = dot(transfo,z.T).T
    return data

def learn(cloud,eta=0.001):
    """
    DEFINITION
    run Oja's learning rule on a cloud of datapoints
    
    INPUT
    eta: learning rate
    cloud: the cloud of datapoints
    
    OUTPUT
    the time course of the weight vector 
       
    -G.Hennequin 02.2009.
    """
    # an array with two elements in it 
    w = array([1/sqrt(2),1/sqrt(2)])
    # we have the dimensions of the zeros matrix and the type is float 
    # the name of the matrix is wcourse
    wcourse = zeros((len(cloud),2),float)
    for i in range(0,len(cloud)):
        # we have all the elements are initialized the w 
        wcourse[i] = w
        y = dot(w,cloud[i]) # output
        w = w + eta*y*(cloud[i]-y*w) # learning rule        
    return wcourse

# for internal use
def circ_dist(n,i,j):
  if i == j: 
      return 0.
  else:
      if j < i:
          if (i - j) > (n/2): return (i-n-j)
          else: return (i-j)
      # in the below we return this image by the circ_dist method of the given parameters 
      else: return (-circ_dist(n,j,i))


def make_image(m,sigma):
    """
    DEFINITION
    builds an m-by-m matrix featuring a gaussian bump
    centered randomly
    NOTA: 
    - to make this matrix a vector (concatenating all rows),
    you can use m.flatten()
    - conversely, if you need to reshape a vector into an 
    m-by-m matrix, use v.reshape((m,m))
    
    INPUT
    m: the size of the image (m-by-m)
    sigma: the std of the gaussian bump

    -G.Hennequin 02.2009.
    """
    
    img = zeros((m,m),float)
    ci = int(m*rand())
    cj = int(m*rand())
    for i in range(0,m):
        di = circ_dist(m,ci,i)
        for j in range(0,m):
            dj = circ_dist(m,cj,j)
            img[i,j] = exp (-(di*di + dj*dj)/(2.0*sigma*sigma))
    # now we have each element is the inverse of the previous elements there, we
    # multiply by the scalar img
    return (1./norm(img))*img
```


