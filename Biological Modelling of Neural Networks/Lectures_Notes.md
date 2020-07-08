# Biological Modelling of Neural Networks

**Week 1**

The neurons are such that there are dendrites connected to a soma, which is connected to an axon, the main fiber. In the action potential it takes about 1ms for the potential to come back to normal. When you receive a signal, you have a certain threshold u_i and above that we call this the spike emission. The brain holds memory and learning in the synaptic connections. The passive membrane model displays the following Linear Integrated and Fire model :

<a href="https://www.codecogs.com/eqnedit.php?latex=\tau&space;\frac{du}{dt}&space;=&space;-(u&space;-&space;u_{rest})&space;&plus;&space;RI(t)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\tau&space;\frac{du}{dt}&space;=&space;-(u&space;-&space;u_{rest})&space;&plus;&space;RI(t)" title="\tau \frac{du}{dt} = -(u - u_{rest}) + RI(t)" /></a>

Solving the above equation we get :

<a href="https://www.codecogs.com/eqnedit.php?latex=u(t)&space;=&space;u_{rest}&space;&plus;&space;(u_0&space;-&space;u_{rest}).exp(\frac{-(t-t_0)}{\tau})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?u(t)&space;=&space;u_{rest}&space;&plus;&space;(u_0&space;-&space;u_{rest}).exp(\frac{-(t-t_0)}{\tau})" title="u(t) = u_{rest} + (u_0 - u_{rest}).exp(\frac{-(t-t_0)}{\tau})" /></a>

We also have the following non linear Integrate-and-Fire model :

<a href="https://www.codecogs.com/eqnedit.php?latex=\tau&space;\frac{du}{dt}&space;=&space;F(u)&space;&plus;&space;RI(t)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\tau&space;\frac{du}{dt}&space;=&space;F(u)&space;&plus;&space;RI(t)" title="\tau \frac{du}{dt} = F(u) + RI(t)" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=u(t)&space;=&space;\theta" target="_blank"><img src="https://latex.codecogs.com/gif.latex?u(t)&space;=&space;\theta" title="u(t) = \theta" /></a>

In this case we can for example have :

<a href="https://www.codecogs.com/eqnedit.php?latex=F(u)&space;=&space;c_2&space;(u&space;-&space;c_1)^2&space;&plus;&space;c_0" target="_blank"><img src="https://latex.codecogs.com/gif.latex?F(u)&space;=&space;c_2&space;(u&space;-&space;c_1)^2&space;&plus;&space;c_0" title="F(u) = c_2 (u - c_1)^2 + c_0" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=F(u)&space;=&space;-(u&space;-&space;u_{rest})&space;&plus;&space;c_0&space;(exp(u&space;-&space;\theta))" target="_blank"><img src="https://latex.codecogs.com/gif.latex?F(u)&space;=&space;-(u&space;-&space;u_{rest})&space;&plus;&space;c_0&space;(exp(u&space;-&space;\theta))" title="F(u) = -(u - u_{rest}) + c_0 (exp(u - \theta))" /></a>

In LIF we have a strict voltage threshold and in the non linear integrate and fire model there is a strict firing threshold. We have :

<a href="https://www.codecogs.com/eqnedit.php?latex=\frac{du}{dt}&space;-&space;\frac{1}{c}I(t)&space;=&space;F(u)\frac{1}{\tau}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{du}{dt}&space;-&space;\frac{1}{c}I(t)&space;=&space;F(u)\frac{1}{\tau}" title="\frac{du}{dt} - \frac{1}{c}I(t) = F(u)\frac{1}{\tau}" /></a>

**Week 2**

The neuronal cell is such that the membrane contains ion channels, ion pumps. We have that the ion density n is such that :

<a href="https://www.codecogs.com/eqnedit.php?latex=n&space;\propto&space;e^{-E/kt}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?n&space;\propto&space;e^{-E/kt}" title="n \propto e^{-E/kt}" /></a>

There must be a concentration difference for the ion pumps to exist. This also means a voltage difference which can be quantified as :

<a href="https://www.codecogs.com/eqnedit.php?latex=\Delta&space;(u)&space;=&space;u_1&space;-&space;u_2&space;=&space;\frac{KT}{q}&space;ln(n(u_1)/n(u_2))" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\Delta&space;(u)&space;=&space;u_1&space;-&space;u_2&space;=&space;\frac{KT}{q}&space;ln(n(u_1)/n(u_2))" title="\Delta (u) = u_1 - u_2 = \frac{KT}{q} ln(n(u_1)/n(u_2))" /></a>

We have the following equations too :

<a href="https://www.codecogs.com/eqnedit.php?latex=C&space;\frac{du}{dt}&space;=&space;-g_{Na}&space;m^3&space;h(u&space;-&space;E_{Na})&space;-&space;g_k&space;n^4&space;(u&space;-&space;E_k)&space;-&space;g_l(u-E_l)&space;&plus;&space;I(t)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?C&space;\frac{du}{dt}&space;=&space;-g_{Na}&space;m^3&space;h(u&space;-&space;E_{Na})&space;-&space;g_k&space;n^4&space;(u&space;-&space;E_k)&space;-&space;g_l(u-E_l)&space;&plus;&space;I(t)" title="C \frac{du}{dt} = -g_{Na} m^3 h(u - E_{Na}) - g_k n^4 (u - E_k) - g_l(u-E_l) + I(t)" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=\frac{dm}{dt}&space;=&space;-&space;\frac{m&space;-&space;m_0(u)}{\tau_m(u)}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{dm}{dt}&space;=&space;-&space;\frac{m&space;-&space;m_0(u)}{\tau_m(u)}" title="\frac{dm}{dt} = - \frac{m - m_0(u)}{\tau_m(u)}" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=\frac{dh}{dt}&space;=&space;-&space;\frac{h&space;-&space;h_0(u)}{\tau_h(u)}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{dh}{dt}&space;=&space;-&space;\frac{h&space;-&space;h_0(u)}{\tau_h(u)}" title="\frac{dh}{dt} = - \frac{h - h_0(u)}{\tau_h(u)}" /></a>

Model of synaptic input and the conductance change :

<a href="https://www.codecogs.com/eqnedit.php?latex=\tau&space;\frac{du}{dt}&space;=&space;-(u&space;-&space;u_{rest})&space;-&space;\sum_{k,&space;f}&space;g_l&space;(t&space;-&space;t_k^f)[u&space;-&space;E_l]&space;-&space;\sum_{k',&space;f'}&space;g_i&space;(t-t_{k'}^{f'})[u&space;-&space;E_i]" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\tau&space;\frac{du}{dt}&space;=&space;-(u&space;-&space;u_{rest})&space;-&space;\sum_{k,&space;f}&space;g_l&space;(t&space;-&space;t_k^f)[u&space;-&space;E_l]&space;-&space;\sum_{k',&space;f'}&space;g_i&space;(t-t_{k'}^{f'})[u&space;-&space;E_i]" title="\tau \frac{du}{dt} = -(u - u_{rest}) - \sum_{k, f} g_l (t - t_k^f)[u - E_l] - \sum_{k', f'} g_i (t-t_{k'}^{f'})[u - E_i]" /></a>

The membrane potential is as follows :

<a href="https://www.codecogs.com/eqnedit.php?latex=\tau&space;\frac{du}{dt}&space;=&space;-(u&space;-&space;u_{rest})&space;-&space;g_l&space;(t&space;-&space;t_0)(u&space;-&space;E_l)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\tau&space;\frac{du}{dt}&space;=&space;-(u&space;-&space;u_{rest})&space;-&space;g_l&space;(t&space;-&space;t_0)(u&space;-&space;E_l)" title="\tau \frac{du}{dt} = -(u - u_{rest}) - g_l (t - t_0)(u - E_l)" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=g_l&space;(t&space;-&space;t_0)&space;=&space;g_0&space;\frac{t}{\tau}&space;e^{-t/&space;\tau}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?g_l&space;(t&space;-&space;t_0)&space;=&space;g_0&space;\frac{t}{\tau}&space;e^{-t/&space;\tau}" title="g_l (t - t_0) = g_0 \frac{t}{\tau} e^{-t/ \tau}" /></a>