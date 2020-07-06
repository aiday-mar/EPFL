# Biological Modelling of Neural Networks

*Week 1*

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

