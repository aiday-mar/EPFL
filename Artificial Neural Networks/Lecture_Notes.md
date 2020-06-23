# Artifical Neural Networks Lectures Notes

**Lecture 1**

We try to model machine learning the way the brain tries to learn. The brain has a visual and a motor cortex and a frontal cortex. The visual cortex is at the back of the head, the comparison is done at the fron of the head. Movements of the arms are controlled by the motor cortex above the ears. In one cubic milimeter there are 10'000 neurons and 3 km of wire. Signals are transmitted through axons. The neurons connect with synapses, or contact points on the dendritic tree. The responses add up when several spikes come to the neuron. If the summed response reaches a threshold value then this neuron sends out a spike to yet other neurons. All learning is a change of weights. 

The activity of the output is :

<a href="https://www.codecogs.com/eqnedit.php?latex=x_i&space;=&space;g(\sum_k&space;w_{ik}&space;x_k)" target="_blank"><img src="https://latex.codecogs.com/svg.latex?x_i&space;=&space;g(\sum_k&space;w_{ik}&space;x_k)" title="x_i = g(\sum_k w_{ik} x_k)" /></a>

Where here above the x_i is the output in the neuron i and the w_{ik} are the weights from node k to node i, and x_k are the input weights of node k. Here g is the threshold function. Consider the following hebbian learning rule:

When an axon of cell j repeatedly or persistently takes part in firing cell i, then j's efficiency as one of the cels firing i is increased. The coactivation of the neurons is important in memorizing. Memory is located in the connections, it is largely distributed, memory is not separated from the processing. In artificial neural networks all input starts on the bottom layer of neurons and then moves to upper layers of neurons through a feed forward network. We can also model the role of reward using reinforcement learning. In the game of Go, the reward is sent at the end of the game, positive if the game was won and negative if lost. 

Artificial neural networks are use for classification, for action learning (where reinforcement learning comes into play) and for sequences (of music, speech). Sequence learning requires recurrent connections (feedback connections). 

Start with the classification which is supposed to output a boolean, yes or no. When you classify images, you send an image to a vector and a vector to a classification. You assign either +1 or 0 to inputs. When you consider classification as a geometric problem then you are meant to find a separating surface in the high dimensional input space. Here you use a discriminant function where d(x) = 0 on the surface, d(x) > 0 on all positive examples x, d(x) < 0 for all counter examples x. For the supervised learning we need an original data set of points {(x_i,t_i), 1 <= i <= P} where the t_i = 1 if the output is correct and 0 otherwise. We have here a set of pairs of inputs and outputs. Use the errors meaning when the predicted output is not the actual output to optimize the classifier. Below is an example of a simple perceptron :

<a href="https://www.codecogs.com/eqnedit.php?latex=y^{\mu}&space;=&space;0.5[1&space;&plus;&space;sgn(\sum_k&space;w_k&space;x_k&space;-&space;\theta)]" target="_blank"><img src="https://latex.codecogs.com/svg.latex?y^{\mu}&space;=&space;0.5[1&space;&plus;&space;sgn(\sum_k&space;w_k&space;x_k&space;-&space;\theta)]" title="y^{\mu} = 0.5[1 + sgn(\sum_k w_k x_k - \theta)]" /></a>

Where here we denote by a' the value :

<a href="https://www.codecogs.com/eqnedit.php?latex=\sum_k&space;w_k&space;x_k" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\sum_k&space;w_k&space;x_k" title="\sum_k w_k x_k" /></a>

And then we have :

<a href="https://www.codecogs.com/eqnedit.php?latex=y^{\mu}&space;=&space;g(a')" target="_blank"><img src="https://latex.codecogs.com/svg.latex?y^{\mu}&space;=&space;g(a')" title="y^{\mu} = g(a')" /></a>

And then we have :

<a href="https://www.codecogs.com/eqnedit.php?latex=g(a')&space;=&space;\left\{\begin{matrix}&space;1&space;&&space;a'&space;>&space;\theta&space;\\&space;0.5&space;&&space;a'&space;=&space;\theta&space;\\&space;0&space;&&space;a'&space;<&space;\theta&space;\end{matrix}\right." target="_blank"><img src="https://latex.codecogs.com/svg.latex?g(a')&space;=&space;\left\{\begin{matrix}&space;1&space;&&space;a'&space;>&space;\theta&space;\\&space;0.5&space;&&space;a'&space;=&space;\theta&space;\\&space;0&space;&&space;a'&space;<&space;\theta&space;\end{matrix}\right." title="g(a') = \left\{\begin{matrix} 1 & a' > \theta \\ 0.5 & a' = \theta \\ 0 & a' < \theta \end{matrix}\right." /></a>

In this case we have the discriminant function is obtained when equating the inside of the sgn function with zero meaning when d(x) = 0. We can also make the d(x) into a whole expression with N+1 terms x_k by writing x_{N+1} = -1 and by writing w_{N+1} = \theta. In this dimensional space which is bigger by one then we have the hyperplane passes through the origin. In gradient descent the quadratic error is :

<a href="https://www.codecogs.com/eqnedit.php?latex=E(w)&space;=&space;\frac{1}{2}&space;\sum_{\mu&space;=&space;1}^p&space;[t^{\mu}&space;-&space;y^{\mu}]" target="_blank"><img src="https://latex.codecogs.com/svg.latex?E(w)&space;=&space;\frac{1}{2}&space;\sum_{\mu&space;=&space;1}^p&space;[t^{\mu}&space;-&space;y^{\mu}]" title="E(w) = \frac{1}{2} \sum_{\mu = 1}^p [t^{\mu} - y^{\mu}]" /></a>

**Lecture 2**

You can use supervised learning in order to try to predict the next element of a sequence. In the brain the reward is modulated by dopamine, it is a neuromodulator. Activity in the brain are changed if three factors come together : the activity of the sending neuron j, some form of activity of the receiving neuron i, and the success signal. Animals learn through conditioning, consider the example of the Morris water maze ? In the same way the artificial neural network AlphaZero discovered different strategies by playing against itself. In deep reinforcement learning we change the connections, and the aim is to choose the next action to win, and the aim for the current value unit is to predict the value of the current position.

Let's start with the formalization of reinforcement learning. We have three notions as follows : states, actions, rewards. In standard RL we have discretized space and actions. We introduce the following notations :

old state : s

new state : s'

current state : s_t

discrete actions : a 

mean rewards for the transitions : 
<a href="https://www.codecogs.com/eqnedit.php?latex=R^a_{s&space;\rightarrow&space;s'}=&space;E(r&space;|&space;s,&space;a,&space;s')" target="_blank"><img src="https://latex.codecogs.com/svg.latex?R^a_{s&space;\rightarrow&space;s'}=&space;E(r&space;|&space;s,&space;a,&space;s')" title="R^a_{s \rightarrow s'}= E(r | s, a, s')" /></a>

current reward : r_t

Most transitions have zero reward. An episode finishes when the target is reached. Over time the episodes get shorter and shorter indicating that the reinforcement learning is ahcieving the results quicker. One notion in RL is the Q-value. Q(s,a) has two indices : you start in state s and take action a. This Q-value is the mean expected reward that you will get if you take action a starting from state s. In other words we have :

<a href="https://www.codecogs.com/eqnedit.php?latex=Q(s,&space;a)&space;=&space;\sum_{s'}&space;P^a_{s&space;\rightarrow&space;s'}&space;E(r&space;|&space;s,&space;a,&space;s')" target="_blank"><img src="https://latex.codecogs.com/svg.latex?Q(s,&space;a)&space;=&space;\sum_{s'}&space;P^a_{s&space;\rightarrow&space;s'}&space;E(r&space;|&space;s,&space;a,&space;s')" title="Q(s, a) = \sum_{s'} P^a_{s \rightarrow s'} E(r | s, a, s')" /></a>

