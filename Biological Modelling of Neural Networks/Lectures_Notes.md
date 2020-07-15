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

**Week 3**

We can reduce the Hodgkin-Huxley method as follows :

<a href="https://www.codecogs.com/eqnedit.php?latex=\tau&space;\frac{du}{dt}&space;=&space;au&space;-&space;w&space;&plus;&space;I_0" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\tau&space;\frac{du}{dt}&space;=&space;au&space;-&space;w&space;&plus;&space;I_0" title="\tau \frac{du}{dt} = au - w + I_0" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=\tau_w&space;\frac{du}{dt}&space;=&space;cu&space;-&space;w" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\tau_w&space;\frac{du}{dt}&space;=&space;cu&space;-&space;w" title="\tau_w \frac{du}{dt} = cu - w" /></a>

In the type 1 model we have that :

<a href="https://www.codecogs.com/eqnedit.php?latex=\frac{dw}{dt}&space;=&space;-&space;\frac{w&space;-&space;0.5[1&space;&plus;&space;tanh(\frac{u&space;-\theta}{d})]}{\tau_w(u)}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{dw}{dt}&space;=&space;-&space;\frac{w&space;-&space;0.5[1&space;&plus;&space;tanh(\frac{u&space;-\theta}{d})]}{\tau_w(u)}" title="\frac{dw}{dt} = - \frac{w - 0.5[1 + tanh(\frac{u -\theta}{d})]}{\tau_w(u)}" /></a>

**Week 4**

We will now study Hebbian learning. The rule is that when an axon of cell j repeatedly or persistently takes part in firing cell i, then j’s efficiency as one of the cells firing i is increased. Slow induction of changes is called homeostatis. We study short-term plasticity/fast synaptic dynamics. Hebbian learning is also unsupervised learning. We have :

<a href="https://www.codecogs.com/eqnedit.php?latex=\Delta&space;w_{ij}&space;\propto&space;F(pre,&space;post)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\Delta&space;w_{ij}&space;\propto&space;F(pre,&space;post)" title="\Delta w_{ij} \propto F(pre, post)" /></a>

Reinforcement learning is hebbian learning with a reward. We have :

<a href="https://www.codecogs.com/eqnedit.php?latex=\Delta&space;w_{ij}&space;\propto&space;F(pre,&space;post,&space;SUCCESS)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\Delta&space;w_{ij}&space;\propto&space;F(pre,&space;post,&space;SUCCESS)" title="\Delta w_{ij} \propto F(pre, post, SUCCESS)" /></a>

In the Hebbian learning the rate model says that a high rate implies many spikes per second. We have the following rate-based learning rate :

<a href="https://www.codecogs.com/eqnedit.php?latex=\frac{dw_{ij}}{dt}&space;=&space;F(w_{ij},&space;v_j^{pre},&space;v_i^{post})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{dw_{ij}}{dt}&space;=&space;F(w_{ij},&space;v_j^{pre},&space;v_i^{post})" title="\frac{dw_{ij}}{dt} = F(w_{ij}, v_j^{pre}, v_i^{post})" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=\frac{dw_{ij}}{dt}&space;=&space;a_0&space;&plus;&space;a_1^{pre}v_j^{pre}&space;&plus;&space;a_1^{post}v_i^{post}&space;&plus;&space;a_2^{corr}&space;v_j^{pre}&space;v_i^{post}&space;&plus;..." target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{dw_{ij}}{dt}&space;=&space;a_0&space;&plus;&space;a_1^{pre}v_j^{pre}&space;&plus;&space;a_1^{post}v_i^{post}&space;&plus;&space;a_2^{corr}&space;v_j^{pre}&space;v_i^{post}&space;&plus;..." title="\frac{dw_{ij}}{dt} = a_0 + a_1^{pre}v_j^{pre} + a_1^{post}v_i^{post} + a_2^{corr} v_j^{pre} v_i^{post} +..." /></a>

The presynaptically gated rule :

<a href="https://www.codecogs.com/eqnedit.php?latex=\frac{dw_{ij}}{dt}&space;=&space;a_2^{corr}(v_i^{post}&space;-&space;\theta)&space;v_j^{pre}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{dw_{ij}}{dt}&space;=&space;a_2^{corr}(v_i^{post}&space;-&space;\theta)&space;v_j^{pre}" title="\frac{dw_{ij}}{dt} = a_2^{corr}(v_i^{post} - \theta) v_j^{pre}" /></a>

The BCM rule is :

<a href="https://www.codecogs.com/eqnedit.php?latex=\frac{dw_{ij}}{dt}&space;=&space;a_2^{corr}&space;\phi&space;(v_i^{post}&space;-&space;\theta)&space;v_j^{pre}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{dw_{ij}}{dt}&space;=&space;a_2^{corr}&space;\phi&space;(v_i^{post}&space;-&space;\theta)&space;v_j^{pre}" title="\frac{dw_{ij}}{dt} = a_2^{corr} \phi (v_i^{post} - \theta) v_j^{pre}" /></a>

And we have :

<a href="https://www.codecogs.com/eqnedit.php?latex=\theta&space;=&space;f(v_i^{post})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\theta&space;=&space;f(v_i^{post})" title="\theta = f(v_i^{post})" /></a>

Hebbian learning leads to specialized Neurons (developmental learning). There is also the receptive field development. 

**Week 5**

Pattern recognition is also the classification by similarity. We consider the following detour :

<a href="https://www.codecogs.com/eqnedit.php?latex=S_i(t&plus;1)&space;=&space;sgn[\sum_j&space;w_{ij}S_j(t)]" target="_blank"><img src="https://latex.codecogs.com/gif.latex?S_i(t&plus;1)&space;=&space;sgn[\sum_j&space;w_{ij}S_j(t)]" title="S_i(t+1) = sgn[\sum_j w_{ij}S_j(t)]" /></a>

Where :

<a href="https://www.codecogs.com/eqnedit.php?latex=w_{ij}&space;=&space;1/N&space;\sum_{\mu}&space;p_i^{\mu}&space;p_j^{\mu}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?w_{ij}&space;=&space;1/N&space;\sum_{\mu}&space;p_i^{\mu}&space;p_j^{\mu}" title="w_{ij} = 1/N \sum_{\mu} p_i^{\mu} p_j^{\mu}" /></a>

**Week 6**

Consider the following memory retrieval used in the Hopfield model, it also has overlaps. 

<a href="https://www.codecogs.com/eqnedit.php?latex=S_i(t&plus;1)&space;=&space;sgn[\sum_{\mu}&space;p_i^{\mu}&space;m^{\mu}(t)]" target="_blank"><img src="https://latex.codecogs.com/gif.latex?S_i(t&plus;1)&space;=&space;sgn[\sum_{\mu}&space;p_i^{\mu}&space;m^{\mu}(t)]" title="S_i(t+1) = sgn[\sum_{\mu} p_i^{\mu} m^{\mu}(t)]" /></a>

In the attractor networks the dynamics moves the network state to a fixed point. In the Hopfield model, for a small number of patterns, states with overlap 1 are fixed points. The stochastic hopfield model says that neurons may be noisy, and we check what this means for the attractor dynamics. We have the following equation : 

<a href="https://www.codecogs.com/eqnedit.php?latex=Pr\{S_i(t&plus;1)&space;=&space;&plus;1&space;|&space;h_i\}&space;=&space;g[h_i]&space;=&space;g[\sum_j&space;w_{ij}&space;S_j(t)]&space;=&space;g[\sum_{\mu}&space;p_i^{\mu}&space;m^{\mu}(t)]" target="_blank"><img src="https://latex.codecogs.com/gif.latex?Pr\{S_i(t&plus;1)&space;=&space;&plus;1&space;|&space;h_i\}&space;=&space;g[h_i]&space;=&space;g[\sum_j&space;w_{ij}&space;S_j(t)]&space;=&space;g[\sum_{\mu}&space;p_i^{\mu}&space;m^{\mu}(t)]" title="Pr\{S_i(t+1) = +1 | h_i\} = g[h_i] = g[\sum_j w_{ij} S_j(t)] = g[\sum_{\mu} p_i^{\mu} m^{\mu}(t)]" /></a>

The stochastic hopfield mode is the attractor model. Consider the energy picture, we have :

<a href="https://www.codecogs.com/eqnedit.php?latex=E&space;=&space;-\frac{1}{2}\sum_{i,j}&space;w_{ij}&space;S_i&space;S_j" target="_blank"><img src="https://latex.codecogs.com/gif.latex?E&space;=&space;-\frac{1}{2}\sum_{i,j}&space;w_{ij}&space;S_i&space;S_j" title="E = -\frac{1}{2}\sum_{i,j} w_{ij} S_i S_j" /></a>

Here is the attractor model with low activity patterns :

<a href="https://www.codecogs.com/eqnedit.php?latex=w_{i,j}&space;=&space;c&space;\sum_{\mu}&space;(\epsilon_i^{\mu}&space;-&space;b)(\epsilon_j^{\mu}&space;-&space;a)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?w_{i,j}&space;=&space;c&space;\sum_{\mu}&space;(\epsilon_i^{\mu}&space;-&space;b)(\epsilon_j^{\mu}&space;-&space;a)" title="w_{i,j} = c \sum_{\mu} (\epsilon_i^{\mu} - b)(\epsilon_j^{\mu} - a)" /></a>

Introduce the overlap :

<a href="https://www.codecogs.com/eqnedit.php?latex=m^{\mu}(t)&space;=&space;c&space;\sum_{j}&space;(\epsilon_j^{\mu}&space;-&space;a)S_j(t)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?m^{\mu}(t)&space;=&space;c&space;\sum_{j}&space;(\epsilon_j^{\mu}&space;-&space;a)S_j(t)" title="m^{\mu}(t) = c \sum_{j} (\epsilon_j^{\mu} - a)S_j(t)" /></a>

**Week 6 old**

Reward-based action learning says that :

<a href="https://www.codecogs.com/eqnedit.php?latex=Q(s,a)&space;=&space;\sum_{s'}&space;P^a_{s&space;\rightarrow&space;s'}R^a_{s&space;\rightarrow&space;s'}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?Q(s,a)&space;=&space;\sum_{s'}&space;P^a_{s&space;\rightarrow&space;s'}R^a_{s&space;\rightarrow&space;s'}" title="Q(s,a) = \sum_{s'} P^a_{s \rightarrow s'}R^a_{s \rightarrow s'}" /></a>

The iterative update says that :

<a href="https://www.codecogs.com/eqnedit.php?latex=\Delta&space;Q(s,a)&space;=&space;\eta&space;[r&space;-&space;Q(s,a)]" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\Delta&space;Q(s,a)&space;=&space;\eta&space;[r&space;-&space;Q(s,a)]" title="\Delta Q(s,a) = \eta [r - Q(s,a)]" /></a>

Hippocampal place cells is the fuzzy discretisation of continuous space. The external stimuli comes in to the spatial representation and that goes to the action learning. 

...

**Week 7**

The following is the population activity :

<a href="https://www.codecogs.com/eqnedit.php?latex=A(t)&space;=&space;\frac{n(t;&space;t&space;&plus;&space;\Delta&space;t)}{N&space;\Delta&space;t}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?A(t)&space;=&space;\frac{n(t;&space;t&space;&plus;&space;\Delta&space;t)}{N&space;\Delta&space;t}" title="A(t) = \frac{n(t; t + \Delta t)}{N \Delta t}" /></a>

Neighboring cells in the visual cortex have a similar preferred orientation, this is called the cortical orientation map. A population is a group of neurons with similar neuronal properties, similar input, similar receptive field, similar connectivity. There are different connectivity schemes : full connectivity or all-to-all or random connectivity with the number K of fixed inputs.

A homogeneous network is such that all neurons are ‘the same’, all synapses are ‘the same’, each neuron receives input from k neurons in the network, each neuron receives the same (mean) external input. Mean-field connectivity, or full-connectivity.

<a href="https://www.codecogs.com/eqnedit.php?latex=I(t)&space;=&space;I^{ext}(t)&space;&plus;&space;I^{net}(t)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?I(t)&space;=&space;I^{ext}(t)&space;&plus;&space;I^{net}(t)" title="I(t) = I^{ext}(t) + I^{net}(t)" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=I^{net}(t)&space;=&space;\sum_{j}&space;\sum_{f}&space;w_{ij}&space;\alpha&space;(t-t_j^f)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?I^{net}(t)&space;=&space;\sum_{j}&space;\sum_{f}&space;w_{ij}&space;\alpha&space;(t-t_j^f)" title="I^{net}(t) = \sum_{j} \sum_{f} w_{ij} \alpha (t-t_j^f)" /></a>

We also have the following equality :

<a href="https://www.codecogs.com/eqnedit.php?latex=I_i(t)&space;=&space;J_0&space;\int&space;\alpha(s)&space;A(t-s)ds&space;&plus;&space;I^{ext}(t)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?I_i(t)&space;=&space;J_0&space;\int&space;\alpha(s)&space;A(t-s)ds&space;&plus;&space;I^{ext}(t)" title="I_i(t) = J_0 \int \alpha(s) A(t-s)ds + I^{ext}(t)" /></a>

Suppose all the variables are constant in time, we have :

<a href="https://www.codecogs.com/eqnedit.php?latex=I_0&space;=&space;J_0&space;A_0&space;\int&space;\alpha(s)&space;ds&space;&plus;&space;I_0^{ext}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?I_0&space;=&space;J_0&space;A_0&space;\int&space;\alpha(s)&space;ds&space;&plus;&space;I_0^{ext}" title="I_0 = J_0 A_0 \int \alpha(s) ds + I_0^{ext}" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=v&space;=&space;g(I_0)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?v&space;=&space;g(I_0)" title="v = g(I_0)" /></a>

Which is the frequency v. 

Gain-function g is frequency-current relation. Function g can be calculated analytically or measured in single-neuron simulations/single-neuron experiments. Consider the following equations :

<a href="https://www.codecogs.com/eqnedit.php?latex=\tau&space;\frac{du_i}{dt}&space;=&space;-u&space;&plus;&space;I_i" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\tau&space;\frac{du_i}{dt}&space;=&space;-u&space;&plus;&space;I_i" title="\tau \frac{du_i}{dt} = -u + I_i" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=I_i&space;=&space;\sum_{k,f}&space;w_{ik}&space;\alpha(t-t_k^f)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?I_i&space;=&space;\sum_{k,f}&space;w_{ik}&space;\alpha(t-t_k^f)" title="I_i = \sum_{k,f} w_{ik} \alpha(t-t_k^f)" /></a>

Connectivity schemes are random, for a fixed p, but are balanced. For this we have the following equation :

<a href="https://www.codecogs.com/eqnedit.php?latex=I_i&space;=&space;\sum_{k,f}&space;w_{ik}&space;\alpha^{exc}&space;(t-t_k^f)&space;-&space;\sum_{k,f}&space;w_{ik}&space;\alpha^{inh}&space;(t-t_k^f)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?I_i&space;=&space;\sum_{k,f}&space;w_{ik}&space;\alpha^{exc}&space;(t-t_k^f)&space;-&space;\sum_{k,f}&space;w_{ik}&space;\alpha^{inh}&space;(t-t_k^f)" title="I_i = \sum_{k,f} w_{ik} \alpha^{exc} (t-t_k^f) - \sum_{k,f} w_{ik} \alpha^{inh} (t-t_k^f)" /></a>

**Week 8**

Consider the following spike reponse model :

<a href="https://www.codecogs.com/eqnedit.php?latex=u(t)&space;=&space;\int&space;\eta(s)&space;S(t-s)&space;ds&space;&plus;&space;\int_0^{\infty}&space;\kappa(s)I(t-s)ds&space;&plus;&space;u_{rest}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?u(t)&space;=&space;\int&space;\eta(s)&space;S(t-s)&space;ds&space;&plus;&space;\int_0^{\infty}&space;\kappa(s)I(t-s)ds&space;&plus;&space;u_{rest}" title="u(t) = \int \eta(s) S(t-s) ds + \int_0^{\infty} \kappa(s)I(t-s)ds + u_{rest}" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=\rho(t)&space;=&space;f(u(t)&space;-&space;\theta(t))" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\rho(t)&space;=&space;f(u(t)&space;-&space;\theta(t))" title="\rho(t) = f(u(t) - \theta(t))" /></a>

Membrane potential caused by the input :

<a href="https://www.codecogs.com/eqnedit.php?latex=\tau&space;\frac{dh(t)}{dt}&space;=&space;-h(t)&space;&plus;&space;RI(t)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\tau&space;\frac{dh(t)}{dt}&space;=&space;-h(t)&space;&plus;&space;RI(t)" title="\tau \frac{dh(t)}{dt} = -h(t) + RI(t)" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=\tau&space;\frac{dh(t)}{dt}&space;=&space;-h(t)&space;&plus;&space;RI^{ext}(t)&plus;\gamma&space;F(h(t))" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\tau&space;\frac{dh(t)}{dt}&space;=&space;-h(t)&space;&plus;&space;RI^{ext}(t)&plus;\gamma&space;F(h(t))" title="\tau \frac{dh(t)}{dt} = -h(t) + RI^{ext}(t)+\gamma F(h(t))" /></a>

In the continuum model we have the Field equation. 

<a href="https://www.codecogs.com/eqnedit.php?latex=\tau&space;\frac{dh(x,t)}{dt}&space;=&space;-h(x,t)&space;&plus;&space;RI^{ext}(x,t)&plus;d\int&space;w(x-x')F(h(x',t))dx'" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\tau&space;\frac{dh(x,t)}{dt}&space;=&space;-h(x,t)&space;&plus;&space;RI^{ext}(x,t)&plus;d\int&space;w(x-x')F(h(x',t))dx'" title="\tau \frac{dh(x,t)}{dt} = -h(x,t) + RI^{ext}(x,t)+d\int w(x-x')F(h(x',t))dx'" /></a>
