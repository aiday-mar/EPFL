# Artifical Neural Networks Lectures Notes

**Lecture 1**

We try to model machine learning the way the brain tries to learn. The brain has a visual and a motor cortex and a frontal cortex. The visual cortex is at the back of the head, the comparison is done at the fron of the head. Movements of the arms are controlled by the motor cortex above the ears. In one cubic milimeter there are 10'000 neurons and 3 km of wire. Signals are transmitted through axons. The neurons connect with synapses, or contact points on the dendritic tree. The responses add up when several spikes come to the neuron. If the summed response reaches a threshold value then this neuron sends out a spike to yet other neurons. All learning is a change of weights. 

The activity of the output is :

<a href="https://www.codecogs.com/eqnedit.php?latex=x_i&space;=&space;g(\sum_k&space;w_{ik}&space;x_k)" target="_blank"><img src="https://latex.codecogs.com/svg.latex?x_i&space;=&space;g(\sum_k&space;w_{ik}&space;x_k)" title="x_i = g(\sum_k w_{ik} x_k)" /></a>

Where here above the x_i is the output in the neuron i and the w_{ik} are the weights from node k to node i, and x_k are the input weights of node k. Here g is the threshold function. Consider the following hebbian learning rule:

When an axon of cell j repeatedly or persistently takes part in firing cell i, then j's efficiency as one of the cels firing i is increased. The coactivation of the neurons is important in memorizing. Memory is located in the connections, it is largely distributed, memory is not separated from the processing. In artificial neural networks all input starts on the bottom layer of neurons and then moves to upper layers of neurons through a feed forward network. We can also model the role of reward using reinforcement learning. In the game of Go, the reward is sent at the end of the game, positive if the game was won and negative if lost. 

Artificial neural networks are use for classification, for action learning (where reinforcement learning comes into play) and for sequences (of music, speech). Sequence learning requires recurrent connections (feedback connections). 

Start with the classification which is supposed to output a boolean, yes or no. When you classify images, you send an image to a vector and a vector to a classification. You assign either +1 or 0 to inputs. When you consider classification as a geometric problem then you are meant to find a separating surface in the high dimensional input space. Here you use a discriminant function where d(x) = 0 on the surface, d(x) > 0 on all positive examples x, d(x) < 0 for all counter examples x. For the supervised learning we need an original data set of points {(x_i,t_i), 1 <= i <= P} where the t_i = 1 if the output is correct and 0 otherwise. We have here a set of pairs of inputs and outputs. Use the errors meaning when the predicted output is not the actual output to optimize the classifier. 
