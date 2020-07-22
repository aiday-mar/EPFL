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

The optimal policy is the policy a such that :

<a href="https://www.codecogs.com/eqnedit.php?latex=a*&space;=&space;argmax_a[Q(s,&space;a)]" target="_blank"><img src="https://latex.codecogs.com/svg.latex?a*&space;=&space;argmax_a[Q(s,&space;a)]" title="a* = argmax_a[Q(s, a)]" /></a>

In the example, we found a rather specific scheme for how to reduce the learning rate over time. But many other schemes also work in practice. For example you keep h constant for a block of time, and then you decrease it for the next block. This learning rate is used in : 

<a href="https://www.codecogs.com/eqnedit.php?latex=\triangle&space;Q(s,a)&space;=&space;\eta&space;[r_t&space;-&space;Q(s,a)]" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\triangle&space;Q(s,a)&space;=&space;\eta&space;[r_t&space;-&space;Q(s,a)]" title="\triangle Q(s,a) = \eta [r_t - Q(s,a)]" /></a>

Generally you want to choose the action with the maximal Q(s,a), but the problem is that the correct value of Q is unknown. Which means we are in the situation of an exploration - exploitation dilemna where we explore so as to estimate reward probabilities and we exploit by taking the actions which looks optimal and maximizes the reward. A softer version of this method allows you to occasionally choose an action which looks suboptimal but which allows you to further explore the Q-values of other options. The epsilon greey and the softmax algorithms are examples following this idea. The softmax strategy says :

take action a' with the probability 

<a href="https://www.codecogs.com/eqnedit.php?latex=P(a')&space;=&space;\frac{exp[\beta&space;Q(a')]}{\sum_a&space;exp[\beta&space;Q(a)]}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?P(a')&space;=&space;\frac{exp[\beta&space;Q(a')]}{\sum_a&space;exp[\beta&space;Q(a)]}" title="P(a') = \frac{exp[\beta Q(a')]}{\sum_a exp[\beta Q(a)]}" /></a>

Consider the following example of exploration and exploitation, consider the following epsilon-greedy algorithm.

```
bandit(A) :

Initialize for a = 1 to k :
  Q(a) = 0
  N(a) = 0

Repeat forever : 
  A = [ argmax_a Q(a)     with probability 1 - e
        a random action   with probability e ]

R = bandit(A)
N(A) = N(A) + 1
Q(A) = Q(A) + 1/N(A)[R-Q(A)]
```

Here N(a) is a counter of how many times the agent has taken action a. Here the learning rate is 1/N(A). The total expected discounter reward is :

<a href="https://www.codecogs.com/eqnedit.php?latex=Q(s,a)&space;=&space;E[r_t&space;&plus;&space;\gamma&space;r_{t&plus;1}&space;&plus;&space;\gamma^2&space;r_{t&plus;2}&space;&plus;&space;...&space;|&space;s,&space;a]" target="_blank"><img src="https://latex.codecogs.com/svg.latex?Q(s,a)&space;=&space;E[r_t&space;&plus;&space;\gamma&space;r_{t&plus;1}&space;&plus;&space;\gamma^2&space;r_{t&plus;2}&space;&plus;&space;...&space;|&space;s,&space;a]" title="Q(s,a) = E[r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + ... | s, a]" /></a>

We require gamma to be less than one for recurrent networks. The bellman equation relates the Q-value for state s and action a with the Q-values of the neighboring states. We have : 

<a href="https://www.codecogs.com/eqnedit.php?latex=Q(s,a)&space;=&space;\sum_{s'}&space;P_{s&space;\rightarrow&space;s'}^a&space;[R^a_{s&space;\rightarrow&space;s'}&plus;&space;\gamma&space;\cdot&space;\sum_{a'}&space;\pi(s',a')Q(s',a')]" target="_blank"><img src="https://latex.codecogs.com/svg.latex?Q(s,a)&space;=&space;\sum_{s'}&space;P_{s&space;\rightarrow&space;s'}^a&space;[R^a_{s&space;\rightarrow&space;s'}&plus;&space;\gamma&space;\cdot&space;\sum_{a'}&space;\pi(s',a')Q(s',a')]" title="Q(s,a) = \sum_{s'} P_{s \rightarrow s'}^a [R^a_{s \rightarrow s'}+ \gamma \cdot \sum_{a'} \pi(s',a')Q(s',a')]" /></a>

Which is also equal to :

<a href="https://www.codecogs.com/eqnedit.php?latex=Q(s,a)&space;=&space;\sum_{s'}&space;P_{s&space;\rightarrow&space;s'}^a&space;[R^a_{s&space;\rightarrow&space;s'}&plus;&space;\gamma&space;\cdot&space;max_{a'}Q(s',a')]" target="_blank"><img src="https://latex.codecogs.com/svg.latex?Q(s,a)&space;=&space;\sum_{s'}&space;P_{s&space;\rightarrow&space;s'}^a&space;[R^a_{s&space;\rightarrow&space;s'}&plus;&space;\gamma&space;\cdot&space;max_{a'}Q(s',a')]" title="Q(s,a) = \sum_{s'} P_{s \rightarrow s'}^a [R^a_{s \rightarrow s'}+ \gamma \cdot max_{a'}Q(s',a')]" /></a>

For the one step horizon scenario we can calculate the Q-values iteratively. We increase the Q-value by a small amount if the reward observed at time t is larger than our current estimate of Q, and conversely. In the multi step horizon we find :

<a href="https://www.codecogs.com/eqnedit.php?latex=\triangle&space;\hat{Q}(s,a)&space;=&space;\eta&space;[r_t&space;&plus;&space;\gamma&space;\hat{Q}(s',a')&space;-&space;\hat{Q}(s,a)]" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\triangle&space;\hat{Q}(s,a)&space;=&space;\eta&space;[r_t&space;&plus;&space;\gamma&space;\hat{Q}(s',a')&space;-&space;\hat{Q}(s,a)]" title="\triangle \hat{Q}(s,a) = \eta [r_t + \gamma \hat{Q}(s',a') - \hat{Q}(s,a)]" /></a>

The above is called the SARSA update rule. The SARSA algorithm is as follows :

```
being in state s, choose action a
ovserve reward r and next state s'
chose action a' in state s'
update with the SARSA update rule
set s=s' and a=a'
start again
```

Consider the following theorem : suppose the SARSA algorithm has been applied for a very long time with the correct updates. If all the Q-values have converged in expectation meaning the expectation of the updates of Q gives zero then the set of Q-values solves the Bellmann equation. 

**Lecture 3**

The Q-value is the expectation of the accumulated reward (discounted with a factor gamma smaller than one). In the balancing probabilities pi(s, A) denotes the probability to choose action A in state S. We have :

<a href="https://www.codecogs.com/eqnedit.php?latex=1&space;=&space;\sum_{a'}&space;\pi(s,&space;a')" target="_blank"><img src="https://latex.codecogs.com/gif.latex?1&space;=&space;\sum_{a'}&space;\pi(s,&space;a')" title="1 = \sum_{a'} \pi(s, a')" /></a>

The expected SARSA algorithm is :

```
Initialize Q(s, a), for all s in S, all a in A(s), and Q(terminate-state, .) = 0
Repeat :
  Initialize S
  Choose A from S using a policy derived from Q
  Repeat :
    Take action A, observe R, S'
    Choose A' from S' using the policy derived from Q
```

<a href="https://www.codecogs.com/eqnedit.php?latex=Q(s,&space;A)&space;=&space;Q(s,&space;A)&space;&plus;&space;\alpha&space;[R&space;&plus;&space;\sum_{\alpha}&space;\pi(a&space;|&space;S_{t&plus;1})&space;\cdot&space;Q(S_{t&plus;1},&space;a)&space;-&space;Q(S_t,&space;A_t)]" target="_blank"><img src="https://latex.codecogs.com/gif.latex?Q(s,&space;A)&space;=&space;Q(s,&space;A)&space;&plus;&space;\alpha&space;[R&space;&plus;&space;\sum_{\alpha}&space;\pi(a&space;|&space;S_{t&plus;1})&space;\cdot&space;Q(S_{t&plus;1},&space;a)&space;-&space;Q(S_t,&space;A_t)]" title="Q(s, A) = Q(s, A) + \alpha [R + \sum_{\alpha} \pi(a | S_{t+1}) \cdot Q(S_{t+1}, a) - Q(S_t, A_t)]" /></a>

```
    S <- S'
    A <- A'
Until S is terminal 
```

Here we are essentially averaging over all possible next actions with a weight given by policy \pi. Sometimes you can instead use the Q-learning where you perform the average with the best policy, you use the greedy algorithm. Hence instead of the above image in the algorithm, you have the following line. 

<a href="https://www.codecogs.com/eqnedit.php?latex=Q(s,&space;A)&space;=&space;Q(s,&space;A)&space;&plus;&space;\alpha&space;[R&space;&plus;&space;\gamma&space;\cdot&space;max_{\alpha}&space;Q(S',&space;a)&space;-&space;Q(s,A)]" target="_blank"><img src="https://latex.codecogs.com/gif.latex?Q(s,&space;A)&space;=&space;Q(s,&space;A)&space;&plus;&space;\alpha&space;[R&space;&plus;&space;\gamma&space;\cdot&space;max_{\alpha}&space;Q(S',&space;a)&space;-&space;Q(s,A)]" title="Q(s, A) = Q(s, A) + \alpha [R + \gamma \cdot max_{\alpha} Q(S', a) - Q(s,A)]" /></a>

Here you use the greedy policy during the update. You turn off the current policy during the update. IT is a temporal difference algorithm because neighboring states are visited one after the other. 

Now define the state value V(s) of a state s which is the total discounted expected reward the agent gets when starting from state s. Meaning :

<a href="https://www.codecogs.com/eqnedit.php?latex=V(s)&space;=&space;\sum_a&space;\pi(s,A)&space;\cdot&space;Q(s,&space;A)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?V(s)&space;=&space;\sum_a&space;\pi(s,A)&space;\cdot&space;Q(s,&space;A)" title="V(s) = \sum_a \pi(s,A) \cdot Q(s, A)" /></a>

It's basically the averaging over the Q-values where we take into account the actions that can be taken starting from state s.

<a href="https://www.codecogs.com/eqnedit.php?latex=V(s)&space;=&space;\sum_a&space;\pi(s,A)&space;\sum_{s'}&space;P_{s&space;\rightarrow&space;s'}^a&space;[R^a_{s&space;\rightarrow&space;s'}&space;&plus;&space;\gamma&space;V(s')]" target="_blank"><img src="https://latex.codecogs.com/gif.latex?V(s)&space;=&space;\sum_a&space;\pi(s,A)&space;\sum_{s'}&space;P_{s&space;\rightarrow&space;s'}^a&space;[R^a_{s&space;\rightarrow&space;s'}&space;&plus;&space;\gamma&space;V(s')]" title="V(s) = \sum_a \pi(s,A) \sum_{s'} P_{s \rightarrow s'}^a [R^a_{s \rightarrow s'} + \gamma V(s')]" /></a>

We have the following temporal distance learning :

```
Input : the policy \pi to be evaluated
InitiaiLize V(s) arbitrarily
Repeat :
  Initialize s
  Repeat :
  A <- action gieVen by \pi for s
  Take action A, observe R, S'
```
<a href="https://www.codecogs.com/eqnedit.php?latex=V(s)&space;=&space;V(s)&space;&plus;&space;\alpha[R&space;&plus;&space;\gamma&space;V(s')&space;-&space;V(s)]" target="_blank"><img src="https://latex.codecogs.com/gif.latex?V(s)&space;=&space;V(s)&space;&plus;&space;\alpha[R&space;&plus;&space;\gamma&space;V(s')&space;-&space;V(s)]" title="V(s) = V(s) + \alpha[R + \gamma V(s') - V(s)]" /></a>

```
  S <- S'
Until S is terminal 
```

We have then :

<a href="https://www.codecogs.com/eqnedit.php?latex=\Delta&space;V(s)&space;=&space;\eta[r_t&space;&plus;&space;\gamma&space;V(s')&space;-&space;V(s)]" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\Delta&space;V(s)&space;=&space;\eta[r_t&space;&plus;&space;\gamma&space;V(s')&space;-&space;V(s)]" title="\Delta V(s) = \eta[r_t + \gamma V(s') - V(s)]" /></a>

For some reason then you can calculate the return as :

<a href="https://www.codecogs.com/eqnedit.php?latex=Return(s)&space;=&space;r_t&space;&plus;&space;\gamma&space;r_{t&plus;1}&space;&plus;&space;\gamma^2&space;r_{t&plus;2}&space;&plus;&space;\gamma^3&space;r_{t&plus;3}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?Return(s)&space;=&space;r_t&space;&plus;&space;\gamma&space;r_{t&plus;1}&space;&plus;&space;\gamma^2&space;r_{t&plus;2}&space;&plus;&space;\gamma^3&space;r_{t&plus;3}" title="Return(s) = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \gamma^3 r_{t+3}" /></a>

This is the return for a single episode. 

Consider the following Monte-Carlo estimation of V-values.

```
Initialize :
  \pi <- policy to eb evaluated
  V <- an arbitrary state-value function
  Return(s) <- an empty list, for all s in S
Repeat forever :
  Generate an episode using \pi
  For each state s appearing in the episode:
    G <- the return that follows the first occurence of s
    Append G to Return(s)
    V(s) <- average(Returns(s))
 ```
 
 Now we have the monte carlo estimation of Q-values :
 
 ```
 Initialize for all s in S, for all a in A(s)
  Q(s, a) <- arbitrary
  \pi(s) <- arbitrary
  Return(s,a) <- empty list
Repeat forever :
  Choose s_0 in S, and A_o in A(s_0) such that all the pairs have probability > 0
  Generate an episode starting from s_0, A_0 following \pi
  For each pair {s,a} appearing in the episode :
    G <- the return that follows the first occurence of {s, a}
    Append G to Returns(s,a)
    Q(s,a) <- average(Returns(s,a))
  For each s in the episode :
    \pi(s) <- argmax_{\alpha} Q(s,a)
 ```
 
 Combine epsilon-greedy policy with Monte-Carlo Q-estimates :
 
 ```
 Initialize for all s in S, a in A(s) :
    Q(s,a) <- arbitrary
    Returns(s,a) <- empty list
    \pi(a | s) <- an arbitrary epsilon-soft policy
 
 Repeat forever :
    (a) Generate an episode using \pi
    (b) For each pair s,a appearing in the episode : 
        G <- the return follows the first occurence of s,a 
        Append G to Returns(s,a)
        Q(s,a) <- average(Returns(s,a))
    (c) For each s in the episode :
        A* <- argmax_a Q(s,a)
        For all a in A(s) :
 ```
 
<a href="https://www.codecogs.com/eqnedit.php?latex=\pi(a&space;|&space;s)&space;\leftarrow&space;\left\{\begin{matrix}&space;1&space;-&space;\epsilon&space;&plus;&space;\epsilon/&space;|A(s)|,&space;&&space;a&space;=&space;A*\\&space;\epsilon/&space;|A(s)|,&space;&&space;a&space;\neq&space;A*&space;\end{matrix}\right." target="_blank"><img src="https://latex.codecogs.com/gif.latex?\pi(a&space;|&space;s)&space;\leftarrow&space;\left\{\begin{matrix}&space;1&space;-&space;\epsilon&space;&plus;&space;\epsilon/&space;|A(s)|,&space;&&space;a&space;=&space;A*\\&space;\epsilon/&space;|A(s)|,&space;&&space;a&space;\neq&space;A*&space;\end{matrix}\right." title="\pi(a | s) \leftarrow \left\{\begin{matrix} 1 - \epsilon + \epsilon/ |A(s)|, & a = A*\\ \epsilon/ |A(s)|, & a \neq A* \end{matrix}\right." /></a>

You can decide to update the eligibility for all state-action pairs. The eligibility increases if the pair actually happens, otherwise it becomes smaller by a factor of lambda. 

<a href="https://www.codecogs.com/eqnedit.php?latex=e(s,a)&space;\leftarrow&space;\lambda&space;e(s,a)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?e(s,a)&space;\leftarrow&space;\lambda&space;e(s,a)" title="e(s,a) \leftarrow \lambda e(s,a)" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=e(s,a)&space;\leftarrow&space;e(s,a)&space;&plus;&space;1" target="_blank"><img src="https://latex.codecogs.com/gif.latex?e(s,a)&space;\leftarrow&space;e(s,a)&space;&plus;&space;1" title="e(s,a) \leftarrow e(s,a) + 1" /></a>

The second operation is executed when action a is chosen in state s. Eligibility traces make the flow of information from the target back into the graph more rapid.

Another solution is the n-step SARSA. Here below we have the equation of the 2-step SARSA :

<a href="https://www.codecogs.com/eqnedit.php?latex=\Delta&space;Q(s_t,&space;a_t)&space;=&space;\eta&space;[r_t&space;&plus;&space;\gamma&space;r_{t&plus;1}&space;&plus;&space;\gamma&space;\gamma&space;Q(s_{t&plus;2},&space;a_{t&plus;2})&space;-&space;Q(s_t,&space;a_t)]" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\Delta&space;Q(s_t,&space;a_t)&space;=&space;\eta&space;[r_t&space;&plus;&space;\gamma&space;r_{t&plus;1}&space;&plus;&space;\gamma&space;\gamma&space;Q(s_{t&plus;2},&space;a_{t&plus;2})&space;-&space;Q(s_t,&space;a_t)]" title="\Delta Q(s_t, a_t) = \eta [r_t + \gamma r_{t+1} + \gamma \gamma Q(s_{t+2}, a_{t+2}) - Q(s_t, a_t)]" /></a>

Let's define the following n-step SARSA algorithm :

```
Initialize Q(s,a) arbitrarily, for all s in S, a in A
Initialize \pi to be \epsilon-greedy with respect to Q, or to a fixed given policy
Parameters : step size \alpha in (0,1], small \epsilon > 0, a positive integer n
All store and access operations (for S_t, A_t and R_t) can take their index mod n

Repeat (for each episode) :
  Initialize and store S_0 != terminal
  Select and store an action A_0 ~ \pi(. | S_0)
  T <- \infinity
  For t = 0,1,2... :
    If t < T, then :
      Take action A_t
      Observe and store the next reward as R_{t+1} and the next state as S_{t+1}
      If S_{t+1} is terminal, then :
        T <- t + 1 
      else :
        select and store and action A_{t+1} ~ \pi(. | S_{t+1})
    \tau <- t - n + 1
    If \tau >= 0 :
      (1) ...
      (2) ...
      (3) ...
      If \pi is being learned then ensure that \pi(. | S_{\tau}) is epsilon-greedy with respect to Q
 Until \tau = T - 1
```

(1) 
<a href="https://www.codecogs.com/eqnedit.php?latex=G&space;\leftarrow&space;\sum_{i&space;=&space;\tau&space;&plus;&space;1}^{min(\tau&space;&plus;&space;n,&space;T)}&space;\gamma^{i&space;-&space;\tau&space;-&space;1}&space;R_i" target="_blank"><img src="https://latex.codecogs.com/gif.latex?G&space;\leftarrow&space;\sum_{i&space;=&space;\tau&space;&plus;&space;1}^{min(\tau&space;&plus;&space;n,&space;T)}&space;\gamma^{i&space;-&space;\tau&space;-&space;1}&space;R_i" title="G \leftarrow \sum_{i = \tau + 1}^{min(\tau + n, T)} \gamma^{i - \tau - 1} R_i" /></a>

(2)
<a href="https://www.codecogs.com/eqnedit.php?latex=\textrm{If&space;}&space;\tau&space;&plus;&space;n&space;<&space;T,&space;\textrm{&space;then&space;}&space;G&space;\leftarrow&space;G&space;&plus;&space;\gamma^n&space;Q(S_{\tau&space;&plus;&space;n},&space;A_{\tau&space;&plus;&space;n})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\textrm{If&space;}&space;\tau&space;&plus;&space;n&space;<&space;T,&space;\textrm{&space;then&space;}&space;G&space;\leftarrow&space;G&space;&plus;&space;\gamma^n&space;Q(S_{\tau&space;&plus;&space;n},&space;A_{\tau&space;&plus;&space;n})" title="\textrm{If } \tau + n < T, \textrm{ then } G \leftarrow G + \gamma^n Q(S_{\tau + n}, A_{\tau + n})" /></a>

(3)
<a href="https://www.codecogs.com/eqnedit.php?latex=Q(S_{\tau},&space;A_{\tau})&space;\leftarrow&space;Q(S_{\tau},&space;A_{\tau})&space;&plus;&space;\alpha&space;[G&space;-&space;Q(S_{\tau},&space;A_{\tau})]" target="_blank"><img src="https://latex.codecogs.com/gif.latex?Q(S_{\tau},&space;A_{\tau})&space;\leftarrow&space;Q(S_{\tau},&space;A_{\tau})&space;&plus;&space;\alpha&space;[G&space;-&space;Q(S_{\tau},&space;A_{\tau})]" title="Q(S_{\tau}, A_{\tau}) \leftarrow Q(S_{\tau}, A_{\tau}) + \alpha [G - Q(S_{\tau}, A_{\tau})]" /></a>

The mapping from input states to actions; or from the input states to value functions can be represented by a model with parameters, typically a neural network with adjustable weights. Today we have seen a large variety of TD algorithms. All of these can be understood as iterative solutions of the Bellman equation. The Bellman equation can be formulated with V-values or with Q-values. Bellman equations normally formulate a self-consistency condition over  one step (nearest neighbors), but can be extended to n steps. Monte Carlo methods do not exploit the ‘bootstrapping’ aspect of the Bellman equation since they do not rely on a self-consistency condition. An n-step SARSA is somewhere intermediate between normal SARSA and MonteCarlo.

Discretization of continuous spaces poses several problems. The first problem is that a rescaling becomes necessary after a change of discretization scheme. This problem is solved by eligibility traces as well as by the n-step TD methods The second problem is that a tabular scheme brakes down for fine discretizations. It is solved by a neural network where we learn the weights. Such a neural network enables generalization by forcing a ‘smooth’ V-value or Q-value. 

**Week 4**

Neural network parameterizes Q-values as a function of continuous state s. One output for one action a. Learn weights by playing against itself. We have that : neural network parameterizes V-values as a function of state s, it has one single output, it learns the weights by playing against itself, it minimizes the TD-error of a V-function and uses eligibility traces.

The following is a consistency conition charaterized by an error function :

<a href="https://www.codecogs.com/eqnedit.php?latex=E(w)&space;=&space;\frac{1}{2}[r_t&space;&plus;&space;\gamma&space;V(s'&space;|&space;w)&space;-&space;V(s&space;|w)]^2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?E(w)&space;=&space;\frac{1}{2}[r_t&space;&plus;&space;\gamma&space;V(s'&space;|&space;w)&space;-&space;V(s&space;|w)]^2" title="E(w) = \frac{1}{2}[r_t + \gamma V(s' | w) - V(s |w)]^2" /></a>

TD learning where Q-values are V-values are described by a smooth function, is also called ‘function approximation in TD learning’. The family of functions can be defined by the parameters of a  Neural Network or by the parameters of a linear superposition of basis functions. In all TD learning methods, we have that V-values or Q-values are the central quantities. The actions are taken with the softmax, greedy or the epsilon-greedy policy derived from the Q-values and the V-values.

We have different policy gradient methods : the 1 step-horizon method, where the stimulus is the input vector. We have a single sigmoidal neuron with transfer function g and weight vector w. Define the mean reward as :

<a href="https://www.codecogs.com/eqnedit.php?latex=<R>&space;=&space;\sum_x&space;\sum_{y=&space;\{0,1\}}\pi(y&space;|x)p(x)R(y,x)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?<R>&space;=&space;\sum_x&space;\sum_{y=&space;\{0,1\}}\pi(y&space;|x)p(x)R(y,x)" title="<R> = \sum_x \sum_{y= \{0,1\}}\pi(y |x)p(x)R(y,x)" /></a>

We have the following choice of actions :

<a href="https://www.codecogs.com/eqnedit.php?latex=\pi(a_1&space;|&space;x,&space;w)&space;=&space;prob(y&space;=&space;1&space;|x,w)&space;=&space;g(\sum_k^n&space;w_k&space;x_k)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\pi(a_1&space;|&space;x,&space;w)&space;=&space;prob(y&space;=&space;1&space;|x,w)&space;=&space;g(\sum_k^n&space;w_k&space;x_k)" title="\pi(a_1 | x, w) = prob(y = 1 |x,w) = g(\sum_k^n w_k x_k)" /></a>

We have the following update parameters used to maximize the rewards. If y = 1, then we have :

<a href="https://www.codecogs.com/eqnedit.php?latex=\Delta&space;w_j&space;=&space;\eta&space;\frac{g'}{g}&space;R(1,x)x_j" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\Delta&space;w_j&space;=&space;\eta&space;\frac{g'}{g}&space;R(1,x)x_j" title="\Delta w_j = \eta \frac{g'}{g} R(1,x)x_j" /></a>

Next we also have the following when y = 0 :

<a href="https://www.codecogs.com/eqnedit.php?latex=\Delta&space;w_j&space;=&space;\eta&space;\frac{-g'}{1-g}&space;R(0,x)x_j" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\Delta&space;w_j&space;=&space;\eta&space;\frac{-g'}{1-g}&space;R(0,x)x_j" title="\Delta w_j = \eta \frac{-g'}{1-g} R(0,x)x_j" /></a>

Consider then the following log likelihood trick : 

<a href="https://www.codecogs.com/eqnedit.php?latex=\triangledown_{\theta}&space;J=&space;\int&space;p(H)&space;\triangledown_{\theta}&space;\log&space;p(H)&space;R(H)dH" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\triangledown_{\theta}&space;J=&space;\int&space;p(H)&space;\triangledown_{\theta}&space;\log&space;p(H)&space;R(H)dH" title="\triangledown_{\theta} J= \int p(H) \triangledown_{\theta} \log p(H) R(H)dH" /></a>

Where here you want to optimize the function J. You do optimization by gradient descent. Now we can have the Monte-Carlo approximation of this expectation by taking N trials :

<a href="https://www.codecogs.com/eqnedit.php?latex=\triangledown_{\theta}&space;J=&space;E_H[\triangledown_{\theta}&space;\log&space;p(H)&space;R(H)]&space;\approx&space;\frac{1}{N}&space;\sum_{n=1}^N&space;\triangledown_{\theta}&space;\log&space;p(H^n)&space;R(H^n)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\triangledown_{\theta}&space;J=&space;E_H[\triangledown_{\theta}&space;\log&space;p(H)&space;R(H)]&space;\approx&space;\frac{1}{N}&space;\sum_{n=1}^N&space;\triangledown_{\theta}&space;\log&space;p(H^n)&space;R(H^n)" title="\triangledown_{\theta} J= E_H[\triangledown_{\theta} \log p(H) R(H)] \approx \frac{1}{N} \sum_{n=1}^N \triangledown_{\theta} \log p(H^n) R(H^n)" /></a>

We can now summarize the above \Delta w_j in one equation where <y> is the expectation of the output given input vector x. We have :
  
<a href="https://www.codecogs.com/eqnedit.php?latex=\Delta&space;w_j&space;=&space;\eta&space;\frac{g'}{g(1-g)}&space;R(y,x)&space;[y&space;-&space;g(\sum_k^N&space;w_k&space;x_k)]&space;x_j" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\Delta&space;w_j&space;=&space;\eta&space;\frac{g'}{g(1-g)}&space;R(y,x)&space;[y&space;-&space;g(\sum_k^N&space;w_k&space;x_k)]&space;x_j" title="\Delta w_j = \eta \frac{g'}{g(1-g)} R(y,x) [y - g(\sum_k^N w_k x_k)] x_j" /></a>

We can interpret this from a biological point of view. The learning rule depends on three factors : the reward given by R(y,x), the state of the postsynaptic neuron [ y - <y>] and the presynaptic activity x_j. Consider the policy gradient method over multiple time steps. A calculation yiels several terms of the form : 
  
<a href="https://www.codecogs.com/eqnedit.php?latex=\Delta&space;\theta_j&space;\propto&space;R_{s_t&space;\rightarrow&space;s_{end}&space;}^{a_t}&space;\cdot&space;\frac{d&space;\ln[\pi&space;(a_t&space;|&space;s_t,&space;\theta)]}{d&space;\theta_j}&space;&plus;&space;\gamma&space;\cdot&space;R^{a_{t&plus;1}}_{s_{t&plus;1}&space;\rightarrow&space;s_{end}}&space;\cdot&space;\frac{d&space;\ln&space;[\pi(a_{t&plus;1}&space;|&space;s_{t&plus;1},&space;\theta)]}{d&space;\theta_j}&space;&plus;&space;..." target="_blank"><img src="https://latex.codecogs.com/gif.latex?\Delta&space;\theta_j&space;\propto&space;R_{s_t&space;\rightarrow&space;s_{end}&space;}^{a_t}&space;\cdot&space;\frac{d&space;\ln[\pi&space;(a_t&space;|&space;s_t,&space;\theta)]}{d&space;\theta_j}&space;&plus;&space;\gamma&space;\cdot&space;R^{a_{t&plus;1}}_{s_{t&plus;1}&space;\rightarrow&space;s_{end}}&space;\cdot&space;\frac{d&space;\ln&space;[\pi(a_{t&plus;1}&space;|&space;s_{t&plus;1},&space;\theta)]}{d&space;\theta_j}&space;&plus;&space;..." title="\Delta \theta_j \propto R_{s_t \rightarrow s_{end} }^{a_t} \cdot \frac{d \ln[\pi (a_t | s_t, \theta)]}{d \theta_j} + \gamma \cdot R^{a_{t+1}}_{s_{t+1} \rightarrow s_{end}} \cdot \frac{d \ln [\pi(a_{t+1} | s_{t+1}, \theta)]}{d \theta_j} + ..." /></a>

Consider the following Monte-Carlo Policy-Gradient Control method :

```
Input : a differentiable policy parametrization \pi(a | s, \theta)
Algorithm parameter : step size \alpha > 0
Initialize policy parameter \theta \in R^d

Loop forever (for each episode) :
  Generate an episode S_0, A_0, R_1,...,S_{T-1}, A_{T-1}, R_T, following \pi(.|.,\theta)
  Loop for each step of the episode t = 0,1,...,T-1 :
```
<a href="https://www.codecogs.com/eqnedit.php?latex=G&space;\leftarrow&space;\sum_{k=&space;t&space;&plus;&space;1}^T&space;\gamma^{k-t-1}R_k" target="_blank"><img src="https://latex.codecogs.com/gif.latex?G&space;\leftarrow&space;\sum_{k=&space;t&space;&plus;&space;1}^T&space;\gamma^{k-t-1}R_k" title="G \leftarrow \sum_{k= t + 1}^T \gamma^{k-t-1}R_k" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=\theta&space;\leftarrow&space;\theta&space;&plus;&space;\alpha&space;\gamma^t&space;G&space;\triangledown&space;\ln&space;\pi(A_t&space;|&space;S_t,&space;\theta)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\theta&space;\leftarrow&space;\theta&space;&plus;&space;\alpha&space;\gamma^t&space;G&space;\triangledown&space;\ln&space;\pi(A_t&space;|&space;S_t,&space;\theta)" title="\theta \leftarrow \theta + \alpha \gamma^t G \triangledown \ln \pi(A_t | S_t, \theta)" /></a>

Where G is the total accumulated reward during the episode starting at S_t. The bias could have different choices. One attractive choice is to take the bias equal to the expectation (or empirical mean). The logic is that if you take an action that gives more accumulated discounted reward than your empirical mean in the past, then this action was good and should be reinforced. If you take an action that gives less accumulated discounted reward than your empirical mean in the past, then this action was not good and should be weakened.

In the following we reinforce the algorithm with a baseline, for estimating \pi_{\theta} \approx \pi_* _

```
Input : a differentiable policy parametrization pi(a|s, $\theta$)
Input : a differentiable state-value function parametrization \hat(v)(s, w)
Algorithm parameters : step sizes \alpha^{\theta} > 0, \alpha^w > 0
Initialize the policy parameter &theta \in R^d and state value weights w \in R^d

Loop forever (for each episode) : 
    Generate an episode S_0, A_0, R_1,...., S_{T-1}, A_{T-1}, R_T, following \pi(.|., \theta)
    Loop for each step of the episode t = 0,1,..., T-1 : 
```
<a href="https://www.codecogs.com/eqnedit.php?latex=G&space;\leftarrow&space;\sum_{k&space;=&space;t&space;&plus;&space;1}^T&space;\gamma^{k-t-1}&space;R_k" target="_blank"><img src="https://latex.codecogs.com/gif.latex?G&space;\leftarrow&space;\sum_{k&space;=&space;t&space;&plus;&space;1}^T&space;\gamma^{k-t-1}&space;R_k" title="G \leftarrow \sum_{k = t + 1}^T \gamma^{k-t-1} R_k" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=\delta&space;\leftarrow&space;G&space;-\hat{v}(S_t,&space;w)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\delta&space;\leftarrow&space;G&space;-\hat{v}(S_t,&space;w)" title="\delta \leftarrow G -\hat{v}(S_t, w)" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=w&space;\leftarrow&space;w&space;&plus;&space;\alpha^{w}&space;\gamma^t&space;\delta&space;\triangledown&space;\hat{v}(S_t,&space;w)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?w&space;\leftarrow&space;w&space;&plus;&space;\alpha^{w}&space;\gamma^t&space;\delta&space;\triangledown&space;\hat{v}(S_t,&space;w)" title="w \leftarrow w + \alpha^{w} \gamma^t \delta \triangledown \hat{v}(S_t, w)" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=\theta&space;\leftarrow&space;\theta&space;&plus;&space;\alpha^{\theta}&space;\gamma^t&space;\delta&space;\triangledown&space;\ln&space;\pi(A_t&space;|&space;S_t,&space;\theta)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\theta&space;\leftarrow&space;\theta&space;&plus;&space;\alpha^{\theta}&space;\gamma^t&space;\delta&space;\triangledown&space;\ln&space;\pi(A_t&space;|&space;S_t,&space;\theta)" title="\theta \leftarrow \theta + \alpha^{\theta} \gamma^t \delta \triangledown \ln \pi(A_t | S_t, \theta)" /></a>

**Week 5**

We need some data for the supervised learning. We have P data points as follows :

<a href="https://www.codecogs.com/eqnedit.php?latex=\{&space;(x^{\mu},&space;t^{\mu}),&space;1&space;\leq&space;\mu&space;\leq&space;P\}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\{&space;(x^{\mu},&space;t^{\mu}),&space;1&space;\leq&space;\mu&space;\leq&space;P\}" title="\{ (x^{\mu}, t^{\mu}), 1 \leq \mu \leq P\}" /></a>

Where here we have t^{\mu} can be either equal to one or zero. The task of classification should be that of finding a separating surface in the high dimensional input space which separates points having positive value from those having a value of zero. We have the following equations must be satisfied :

<a href="https://www.codecogs.com/eqnedit.php?latex=\hat{y}&space;=&space;0.5[1&space;&plus;&space;sgn(\sum_k&space;w_k&space;x_k&space;-&space;\theta)]" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\hat{y}&space;=&space;0.5[1&space;&plus;&space;sgn(\sum_k&space;w_k&space;x_k&space;-&space;\theta)]" title="\hat{y} = 0.5[1 + sgn(\sum_k w_k x_k - \theta)]" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=d(x)&space;=&space;\sum_k&space;w_k&space;x_k&space;-&space;\theta" target="_blank"><img src="https://latex.codecogs.com/gif.latex?d(x)&space;=&space;\sum_k&space;w_k&space;x_k&space;-&space;\theta" title="d(x) = \sum_k w_k x_k - \theta" /></a>

But then you can choose w_{N+1} to be \theta and x_{N+1} to be -1 and then we have :

<a href="https://www.codecogs.com/eqnedit.php?latex=d(x)&space;=&space;\sum_{k=1}^{N&plus;1}&space;w_k&space;x_k&space;=&space;0" target="_blank"><img src="https://latex.codecogs.com/gif.latex?d(x)&space;=&space;\sum_{k=1}^{N&plus;1}&space;w_k&space;x_k&space;=&space;0" title="d(x) = \sum_{k=1}^{N+1} w_k x_k = 0" /></a>

After this there is a review of the gradient descent method. We directly start with the Backprop and the multilayer networks. The XOR problem is not linearly separable. In this case you may want to use several layers of neurons to classify the problem. A multilayer perceptron (or multilayer network) has one or serveral hidden layers between input layer and output layer. Below we have a summary of the gradient chain rule :

Step 1 : identify intermediate variables

<a href="https://www.codecogs.com/eqnedit.php?latex=a_k^{(n)}&space;=&space;\sum_l&space;w_{kl}^{(n)}&space;x_l^{(n-1)}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?a_k^{(n)}&space;=&space;\sum_l&space;w_{kl}^{(n)}&space;x_l^{(n-1)}" title="a_k^{(n)} = \sum_l w_{kl}^{(n)} x_l^{(n-1)}" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=\delta_k^{(n)}&space;=&space;\frac{\delta&space;E}{\delta&space;a_k^{(n)}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\delta_k^{(n)}&space;=&space;\frac{\delta&space;E}{\delta&space;a_k^{(n)}}" title="\delta_k^{(n)} = \frac{\delta E}{\delta a_k^{(n)}}" /></a>

Step 2 : we write weight updates 

<a href="https://www.codecogs.com/eqnedit.php?latex=\Delta&space;w_{kl}^{n}&space;=&space;-&space;\gamma&space;\delta_k^{n}&space;x_l^{n-1}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\Delta&space;w_{kl}^{n}&space;=&space;-&space;\gamma&space;\delta_k^{n}&space;x_l^{n-1}" title="\Delta w_{kl}^{n} = - \gamma \delta_k^{n} x_l^{n-1}" /></a>

Step 3 : Analyze the dependency graph

<a href="https://www.codecogs.com/eqnedit.php?latex=\delta_k^{n-1}&space;=&space;\sum_j&space;\delta_j^{n}w_{jk}^n&space;g'" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\delta_k^{n-1}&space;=&space;\sum_j&space;\delta_j^{n}w_{jk}^n&space;g'" title="\delta_k^{n-1} = \sum_j \delta_j^{n}w_{jk}^n g'" /></a>

The following is the defintion of the BackProp algorithm :

0. Initialization of weights
1. Choose pattern x^{\mu}, input is x^{(0)}_k = x^{\mu}_k
2. Knowing the activity in layer n, we calculate the activity in layer n+1 and store the result. We have : x^{(n-1)}_k -> x^{(n)}_j.

<a href="https://www.codecogs.com/eqnedit.php?latex=x_j^{(n)}&space;=&space;g^{(n)}(a^{(n)}_j)&space;=&space;g^{(n)}(\sum&space;w_{jk}^{(n)}x_k^{(n-1)})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?x_j^{(n)}&space;=&space;g^{(n)}(a^{(n)}_j)&space;=&space;g^{(n)}(\sum&space;w_{jk}^{(n)}x_k^{(n-1)})" title="x_j^{(n)} = g^{(n)}(a^{(n)}_j) = g^{(n)}(\sum w_{jk}^{(n)}x_k^{(n-1)})" /></a>

The output is :

<a href="https://www.codecogs.com/eqnedit.php?latex=\hat{y}_i^{\mu}&space;=&space;x_i^{\eta_{max}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\hat{y}_i^{\mu}&space;=&space;x_i^{\eta_{max}}" title="\hat{y}_i^{\mu} = x_i^{\eta_{max}}" /></a>

3. Then we compute the errors in the output as follows

<a href="https://www.codecogs.com/eqnedit.php?latex=\delta_i^{\eta_{max}}&space;=&space;g'(a_i^{(\eta_{max})})[\hat{y}_i^{\mu}&space;-&space;t_i^{\mu}]" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\delta_i^{\eta_{max}}&space;=&space;g'(a_i^{(\eta_{max})})[\hat{y}_i^{\mu}&space;-&space;t_i^{\mu}]" title="\delta_i^{\eta_{max}} = g'(a_i^{(\eta_{max})})[\hat{y}_i^{\mu} - t_i^{\mu}]" /></a>

4. There is a backward propagation of the errors, we have \delta_i^{(n)} -> \delta_j^{(n-1)}

<a href="https://www.codecogs.com/eqnedit.php?latex=\delta_j^{(n-1)}&space;=&space;g'^{(n-1)}(a^{(n-1)})\sum_i&space;w_{ij}\delta_i^{(n)}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\delta_j^{(n-1)}&space;=&space;g'^{(n-1)}(a^{(n-1)})\sum_i&space;w_{ij}\delta_i^{(n)}" title="\delta_j^{(n-1)} = g'^{(n-1)}(a^{(n-1)})\sum_i w_{ij}\delta_i^{(n)}" /></a>

5. Then we update the weights for all (i,j) and all layers (n) :

<a href="https://www.codecogs.com/eqnedit.php?latex=\Delta&space;w_{ij}^{(n)}&space;=&space;-&space;\gamma&space;\delta_i^{(n)}&space;x_j^{(n-1)}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\Delta&space;w_{ij}^{(n)}&space;=&space;-&space;\gamma&space;\delta_i^{(n)}&space;x_j^{(n-1)}" title="\Delta w_{ij}^{(n)} = - \gamma \delta_i^{(n)} x_j^{(n-1)}" /></a>

6. Return to step 1


We consider instead now the direct numerical differentiation. We calculate the following :

<a href="https://www.codecogs.com/eqnedit.php?latex=E(w)&space;=&space;\frac{1}{2}&space;\sum_{\mu&space;=&space;1}^P&space;\sum_i&space;[t_i^{\mu}&space;-&space;\hat{y}_i^{\mu}]^2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?E(w)&space;=&space;\frac{1}{2}&space;\sum_{\mu&space;=&space;1}^P&space;\sum_i&space;[t_i^{\mu}&space;-&space;\hat{y}_i^{\mu}]^2" title="E(w) = \frac{1}{2} \sum_{\mu = 1}^P \sum_i [t_i^{\mu} - \hat{y}_i^{\mu}]^2" /></a>

1. Calculate \hat{y}_i^{\mu} for one pattern, where each weight is touched once
2. For each change of the weights, we evaluate E twice as follows :

<a href="https://www.codecogs.com/eqnedit.php?latex=\Delta&space;w_{jk}^{(1)}&space;=&space;-&space;\gamma&space;\frac{dE(w_{jk}^{(1)}&space;&plus;&space;\epsilon)&space;-&space;dE(w_{jk}^{(1)}&space;-&space;\epsilon)}{2&space;\epsilon}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\Delta&space;w_{jk}^{(1)}&space;=&space;-&space;\gamma&space;\frac{dE(w_{jk}^{(1)}&space;&plus;&space;\epsilon)&space;-&space;dE(w_{jk}^{(1)}&space;-&space;\epsilon)}{2&space;\epsilon}" title="\Delta w_{jk}^{(1)} = - \gamma \frac{dE(w_{jk}^{(1)} + \epsilon) - dE(w_{jk}^{(1)} - \epsilon)}{2 \epsilon}" /></a>

3. For the n weights, the order is then n-square

Therefore multilayer perceptrons are more powerful than simple perceptrons and can be trained using backprop, a gradient descent algorithm. These can implement flexible separating surfaces. The aim of a neural network is that in the end it can make correct predictions on new patterns. 

In the case of classification/approximations of functions, we have that flexibility is bad for noisy data, there is a danger of overfitting. The control of flexibility requires a split of the data in two or three subgroups. We start with a split in two groups : the training base and the validation base. Hence we have the following split of the data.

Training base used to optimize the parameters :

<a href="https://www.codecogs.com/eqnedit.php?latex=\{&space;(x^{\mu},&space;t^{\mu}),&space;1&space;\leq&space;\mu&space;\leq&space;P_1&space;\}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\{&space;(x^{\mu},&space;t^{\mu}),&space;1&space;\leq&space;\mu&space;\leq&space;P_1&space;\}" title="\{ (x^{\mu}, t^{\mu}), 1 \leq \mu \leq P_1 \}" /></a>

Validation base, used to mimic future data : 

<a href="https://www.codecogs.com/eqnedit.php?latex=\{&space;(x^{\mu},&space;t^{\mu}),&space;P_1&space;&plus;&space;1&space;\leq&space;\mu&space;\leq&space;P&space;\}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\{&space;(x^{\mu},&space;t^{\mu}),&space;P_1&space;&plus;&space;1&space;\leq&space;\mu&space;\leq&space;P&space;\}" title="\{ (x^{\mu}, t^{\mu}), P_1 + 1 \leq \mu \leq P \}" /></a>

The data above is used to test the performance (but not to change the weights). An error on the validation base that is much larger than the error on the training base is a signature of overfitting. More generally, the correct flexibility of the network is the one where the error on the validation set is minimal. We give the following algorithm which is used to control the flexibility with artificial neural networks :

```
Change the flexibility, several times
Choose the number of hidden neurons and the number of layers
  Split thedata base into a training base and a validation base 
    Optimize the parameters (several times) :
    Initialize the weights
      Iterate until convergence
      Gradient descent on the training error
    Report the training error and the validation error
  Report the meaning of the trining and the validation error and the standard deviation
Plot the mean training and the validation error
Pick the optimal number of layers and the hidden neurons 
```

Flexibility is a measure of the number of free parameters. Changing the flexibility also means changing the network structure or the number of hidden neurons. Regularization controls the flexibility without changing the explicit number of free parameters. For this we minimize the following training set on the modified error function :

<a href="https://www.codecogs.com/eqnedit.php?latex=\tilde{E}(w)&space;=&space;\frac{1}{2}&space;\sum_{\mu&space;=&space;1}^{P_1}&space;[t^{\mu}&space;-&space;\hat{y}^{\mu}]^2&space;&plus;&space;\lambda&space;\sum_k&space;(w_k)^2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\tilde{E}(w)&space;=&space;\frac{1}{2}&space;\sum_{\mu&space;=&space;1}^{P_1}&space;[t^{\mu}&space;-&space;\hat{y}^{\mu}]^2&space;&plus;&space;\lambda&space;\sum_k&space;(w_k)^2" title="\tilde{E}(w) = \frac{1}{2} \sum_{\mu = 1}^{P_1} [t^{\mu} - \hat{y}^{\mu}]^2 + \lambda \sum_k (w_k)^2" /></a>

We have the following algorithm used to the control the flexibility by the regularizer :

```
Change the flexibility several times
Choose \lambda
  Split the data base into the training set and the validation set
    Optimize the parameters several times :
    Initialize the weights
      Iterate until convergence
      Gradient descent on modified
      Training error \tilde{E}(w)
    Report training error E and test error E^{val} on the validation set
  Report mean training and test error and standard deviation
Plot the mean training and the test error 
Pick the weights for results with the optimal \lambda and the lowest validation error 
```

**Week 6**

The softmax function is given by :

<a href="https://www.codecogs.com/eqnedit.php?latex=\hat{y}_k&space;=&space;\frac{exp(a_k)}{\sum_j&space;exp(a_j)}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\hat{y}_k&space;=&space;\frac{exp(a_k)}{\sum_j&space;exp(a_j)}" title="\hat{y}_k = \frac{exp(a_k)}{\sum_j exp(a_j)}" /></a>

We have the folloing exponential linear unit function :

<a href="https://www.codecogs.com/eqnedit.php?latex=f(a)&space;=&space;\left\{\begin{matrix}&space;a&space;&&space;a&space;>0\\&space;exp(a)&space;-&space;1&space;&&space;a&space;<&space;0&space;\end{matrix}\right." target="_blank"><img src="https://latex.codecogs.com/gif.latex?f(a)&space;=&space;\left\{\begin{matrix}&space;a&space;&&space;a&space;>0\\&space;exp(a)&space;-&space;1&space;&&space;a&space;<&space;0&space;\end{matrix}\right." title="f(a) = \left\{\begin{matrix} a & a >0\\ exp(a) - 1 & a < 0 \end{matrix}\right." /></a>

We study here bagging which is a traditional and generic method of regularization used in machine learning. The first idea of bagging is to repeat a simple model where each variant is optimized for a different subset of the data. Where here we have :

<a href="https://www.codecogs.com/eqnedit.php?latex=\hat{y}_K&space;=&space;0.5[1&space;&plus;&space;tanh(\sum_k&space;w_k&space;x_k&space;-&space;\theta)]" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\hat{y}_K&space;=&space;0.5[1&space;&plus;&space;tanh(\sum_k&space;w_k&space;x_k&space;-&space;\theta)]" title="\hat{y}_K = 0.5[1 + tanh(\sum_k w_k x_k - \theta)]" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=\hat{y}_{bag}&space;=&space;\frac{1}{K}\sum_{k=1}^K&space;\hat{y}_k" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\hat{y}_{bag}&space;=&space;\frac{1}{K}\sum_{k=1}^K&space;\hat{y}_k" title="\hat{y}_{bag} = \frac{1}{K}\sum_{k=1}^K \hat{y}_k" /></a>

For the classification task y_{bag} is compared with a threshold to assign the class. A bagged output is always better than the output from a single model. It has a smaller quadratic error than a typical individual model, and if all K indiivdual models are uncorrelated, the gain in performane scales as 1/K. 

We have the following bagging theorem. Suppose that we have :

<a href="https://www.codecogs.com/eqnedit.php?latex=\delta_k^{\mu}&space;=&space;t^{\mu}&space;-&space;\hat{y}_k^{\mu}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\delta_k^{\mu}&space;=&space;t^{\mu}&space;-&space;\hat{y}_k^{\mu}" title="\delta_k^{\mu} = t^{\mu} - \hat{y}_k^{\mu}" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=\delta_{bag}^{\mu}&space;=&space;t^{\mu}&space;-&space;\hat{y}_{bag}^{\mu}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\delta_{bag}^{\mu}&space;=&space;t^{\mu}&space;-&space;\hat{y}_{bag}^{\mu}" title="\delta_{bag}^{\mu} = t^{\mu} - \hat{y}_{bag}^{\mu}" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=<\frac{1}{P}&space;\sum_{\mu&space;=&space;1}^P&space;[\delta_k^{\mu}]^2>&space;\geq&space;<\frac{1}{P}&space;\sum_{\mu&space;=&space;1}^P&space;[\delta_{bag}^{\mu}]^2>" target="_blank"><img src="https://latex.codecogs.com/gif.latex?<\frac{1}{P}&space;\sum_{\mu&space;=&space;1}^P&space;[\delta_k^{\mu}]^2>&space;\geq&space;<\frac{1}{P}&space;\sum_{\mu&space;=&space;1}^P&space;[\delta_{bag}^{\mu}]^2>" title="<\frac{1}{P} \sum_{\mu = 1}^P [\delta_k^{\mu}]^2> \geq <\frac{1}{P} \sum_{\mu = 1}^P [\delta_{bag}^{\mu}]^2>" /></a>

Dropout is a regularization method that has been specifically developed for neural networks. It is closely related to bagging. Dropout can be interpreted in two different ways. Either it is an approximation of the bagging, or a tool to enforce representation sharing in the hidden neurons. The difference to standard bagging is that models are not independent, they share weights, the data base is not fixed for each dropout configuration, the output is not a sum over the model outputs. 

Data augmentation is an effective regularization method and is low cost. In the case of images for example you can rotate, flip an image and add it too to the database. You can also pixel noise, elastic deformations or shift the color scheme. Data set augmentations avoid overfitting. We have the following initialization :

(1)

<a href="https://www.codecogs.com/eqnedit.php?latex=<x_j>&space;=&space;\frac{1}{P}\sum_{\mu&space;=&space;1}^P&space;x_j^{\mu}&space;=&space;0" target="_blank"><img src="https://latex.codecogs.com/gif.latex?<x_j>&space;=&space;\frac{1}{P}\sum_{\mu&space;=&space;1}^P&space;x_j^{\mu}&space;=&space;0" title="<x_j> = \frac{1}{P}\sum_{\mu = 1}^P x_j^{\mu} = 0" /></a>

(2)

<a href="https://www.codecogs.com/eqnedit.php?latex=<(x_j)^2>&space;=&space;\frac{1}{P}\sum_{\mu&space;=&space;1}^P&space;(x_j^{\mu})(x_j^{\mu})&space;=&space;1" target="_blank"><img src="https://latex.codecogs.com/gif.latex?<(x_j)^2>&space;=&space;\frac{1}{P}\sum_{\mu&space;=&space;1}^P&space;(x_j^{\mu})(x_j^{\mu})&space;=&space;1" title="<(x_j)^2> = \frac{1}{P}\sum_{\mu = 1}^P (x_j^{\mu})(x_j^{\mu}) = 1" /></a>

(3)

<a href="https://www.codecogs.com/eqnedit.php?latex=<w_{ij}^{(n)}>&space;=&space;0" target="_blank"><img src="https://latex.codecogs.com/gif.latex?<w_{ij}^{(n)}>&space;=&space;0" title="<w_{ij}^{(n)}> = 0" /></a>

Suppose that we work with the sigmoidal unit (black). If all the patterns cause activations in the range [-\epsilon, \epsilon] then all the patterns fall in the linear regims of the gain function g. 


Suppose that we work with the ReLu function. If all the patterns cause activations in the range [\epsilon, \alpha] then all the patterns fall in the linear regims of the gain function g. 

To exploit non linearities in the neurons of the multilayer network you need to male sure that the initial choice of the weights is such that each unit has a range of activation values that touch the non linear regime. During the training the weights remain in a regime such that each unit has a range of activation values that touche the non linear regime. 

In the backward pass, the vanishing gradient algorithm, we have the following equality :

<a href="https://www.codecogs.com/eqnedit.php?latex=\delta_i^{(n-1)}&space;=&space;\sum_j&space;w_{ji}^{(n)}g'^{(n-1)}(a_i^{(n-1)})\delta_j^{(n)}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\delta_i^{(n-1)}&space;=&space;\sum_j&space;w_{ji}^{(n)}g'^{(n-1)}(a_i^{(n-1)})\delta_j^{(n)}" title="\delta_i^{(n-1)} = \sum_j w_{ji}^{(n)}g'^{(n-1)}(a_i^{(n-1)})\delta_j^{(n)}" /></a>

We now want to include the weight update. The update formula of the BackProp Algorithm is :

<a href="https://www.codecogs.com/eqnedit.php?latex=\Delta&space;w_{i,j}^{(n-1)}&space;=&space;\delta_i^{(n-1)}x_j^{(n-2)}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\Delta&space;w_{i,j}^{(n-1)}&space;=&space;\delta_i^{(n-1)}x_j^{(n-2)}" title="\Delta w_{i,j}^{(n-1)} = \delta_i^{(n-1)}x_j^{(n-2)}" /></a>

The shifted exponential linear unit : 

<a href="https://www.codecogs.com/eqnedit.php?latex=g(a)=&space;\left\{\begin{matrix}&space;\beta&space;a&space;&&space;a&space;>&space;0\\&space;\gamma&space;[exp(a)-1]&space;&&space;a&space;<0&space;\end{matrix}\right." target="_blank"><img src="https://latex.codecogs.com/gif.latex?g(a)=&space;\left\{\begin{matrix}&space;\beta&space;a&space;&&space;a&space;>&space;0\\&space;\gamma&space;[exp(a)-1]&space;&&space;a&space;<0&space;\end{matrix}\right." title="g(a)= \left\{\begin{matrix} \beta a & a > 0\\ \gamma [exp(a)-1] & a <0 \end{matrix}\right." /></a>

After the update of the activation coefficients :

<a href="https://www.codecogs.com/eqnedit.php?latex=a_i^{(n)}&space;=&space;\sum_j&space;[w_{ij}^{(n)}&space;&plus;&space;\Delta&space;w_{ij}^{(n)}]x_j^{(n-1)}&space;-&space;\theta" target="_blank"><img src="https://latex.codecogs.com/gif.latex?a_i^{(n)}&space;=&space;\sum_j&space;[w_{ij}^{(n)}&space;&plus;&space;\Delta&space;w_{ij}^{(n)}]x_j^{(n-1)}&space;-&space;\theta" title="a_i^{(n)} = \sum_j [w_{ij}^{(n)} + \Delta w_{ij}^{(n)}]x_j^{(n-1)} - \theta" /></a>

We normalize the input on each line :

<a href="https://www.codecogs.com/eqnedit.php?latex=\hat{x}_j^k&space;=&space;\frac{x_j^k&space;-&space;E[x_j^k]}{\sqrt{Var[x_j^k]}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\hat{x}_j^k&space;=&space;\frac{x_j^k&space;-&space;E[x_j^k]}{\sqrt{Var[x_j^k]}}" title="\hat{x}_j^k = \frac{x_j^k - E[x_j^k]}{\sqrt{Var[x_j^k]}}" /></a>

**Week 7**

In two and more dimensions it is possible that the curvature is positive in one direction and negative in the other, this is a saddle point. In a network with m hidden layers with n neurons each there are :

<a href="https://www.codecogs.com/eqnedit.php?latex=n!^m" target="_blank"><img src="https://latex.codecogs.com/gif.latex?n!^m" title="n!^m" /></a>

Equivalent solutions. There are many permutation symmetries in the weights space. At the saddle point the gradient descent is slow. There are many more saddle points than minima. We now have a point is stable if all the eignevalues are positive for the hessian matrix evaluated at that point. Suppose we have N-1 eigenvalues that are positive and one is negative then we have a first-order saddle. Suppose N-2 eigenvalues are negative and 2 are positive, this means in N-2 dimensions the surface does up and in 2 dimensions it goes down, in which case we have a second-order saddle.

Two arguments to explain that there are many more saddle points than minima is that there is a statistical argument and a geometric one. Permutation minima are connected by saddle points. We slowly decrease the distance between two weight vectors, we let the other weight vetors equilibrate to minimum-loss configuration. Once two weight vectors have merged, exchange the labels and relax back on the same path. 

For first-order permutation points, we have to place n vector indices onto n-1 locations that define the configuration with n-1 neurons in the hidden layer that we found by our shifting-of-weight-vector construction. The number of saddle points increases rapidly with the number of neurons in the hidden layer. A layer with n neurons generates at least a factor of :

<a href="https://www.codecogs.com/eqnedit.php?latex=\frac{1}{2^K}\begin{pmatrix}&space;n&space;-&space;K\\&space;K&space;\end{pmatrix}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{1}{2^K}\begin{pmatrix}&space;n&space;-&space;K\\&space;K&space;\end{pmatrix}" title="\frac{1}{2^K}\begin{pmatrix} n - K\\ K \end{pmatrix}" /></a>

In a network with m hidden layers and n neurons per hidden layer, we have found one global minimum. There are at least n!^m minima with the same loss and at least :

<a href="https://www.codecogs.com/eqnedit.php?latex=\frac{m}{2^K}\begin{pmatrix}&space;n&space;-&space;K\\&space;K&space;\end{pmatrix}&space;n!^m" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{m}{2^K}\begin{pmatrix}&space;n&space;-&space;K\\&space;K&space;\end{pmatrix}&space;n!^m" title="\frac{m}{2^K}\begin{pmatrix} n - K\\ K \end{pmatrix} n!^m" /></a>

We have the following standard gradient descent equality :

<a href="https://www.codecogs.com/eqnedit.php?latex=\Delta&space;w_{i,j}^{(n)}(1)&space;=&space;-\gamma&space;\frac{dE(w(1))}{dw_{i,j}^{(n)}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\Delta&space;w_{i,j}^{(n)}(1)&space;=&space;-\gamma&space;\frac{dE(w(1))}{dw_{i,j}^{(n)}}" title="\Delta w_{i,j}^{(n)}(1) = -\gamma \frac{dE(w(1))}{dw_{i,j}^{(n)}}" /></a>


<a href="https://www.codecogs.com/eqnedit.php?latex=\Delta&space;w_{i,j}^{(n)}(m)&space;=&space;-\gamma&space;\frac{dE(w(m))}{dw_{i,j}^{(n)}}&space;&plus;&space;\alpha&space;\Delta&space;w_{i,j}^{(n)}(m-1)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\Delta&space;w_{i,j}^{(n)}(m)&space;=&space;-\gamma&space;\frac{dE(w(m))}{dw_{i,j}^{(n)}}&space;&plus;&space;\alpha&space;\Delta&space;w_{i,j}^{(n)}(m-1)" title="\Delta w_{i,j}^{(n)}(m) = -\gamma \frac{dE(w(m))}{dw_{i,j}^{(n)}} + \alpha \Delta w_{i,j}^{(n)}(m-1)" /></a>

A momentum term keeps information about the previous direction. It suppresses therefore these oscillation while giving rise to a speed-up in the directions where the gradient does not change. The following here is the Nesterov momentum :

We have below the running mean :

<a href="https://www.codecogs.com/eqnedit.php?latex=v_{i,j}^{(n)}(m)&space;=&space;\frac{dE(w(m))}{dw_{i,j}^{(n)}}&space;&plus;&space;\rho_1&space;v_{i,j}^{(n)}(m-1)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?v_{i,j}^{(n)}(m)&space;=&space;\frac{dE(w(m))}{dw_{i,j}^{(n)}}&space;&plus;&space;\rho_1&space;v_{i,j}^{(n)}(m-1)" title="v_{i,j}^{(n)}(m) = \frac{dE(w(m))}{dw_{i,j}^{(n)}} + \rho_1 v_{i,j}^{(n)}(m-1)" /></a>

Then we also have the running second momentum as follows :

<a href="https://www.codecogs.com/eqnedit.php?latex=r_{i,j}^{(n)}(m)&space;=&space;(1-\rho_2)\frac{dE(w(m))}{dw_{i,j}^{(n)}}\frac{dE(w(m))}{dw_{i,j}^{(n)}}&space;&plus;&space;\rho_2&space;r_{i,j}^{(n)}(m-1)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?r_{i,j}^{(n)}(m)&space;=&space;(1-\rho_2)\frac{dE(w(m))}{dw_{i,j}^{(n)}}\frac{dE(w(m))}{dw_{i,j}^{(n)}}&space;&plus;&space;\rho_2&space;r_{i,j}^{(n)}(m-1)" title="r_{i,j}^{(n)}(m) = (1-\rho_2)\frac{dE(w(m))}{dw_{i,j}^{(n)}}\frac{dE(w(m))}{dw_{i,j}^{(n)}} + \rho_2 r_{i,j}^{(n)}(m-1)" /></a>

Consider the following RMSProp algorithm :

```
Require : global learning rate \epsilon, decay rate \rho
Require : initial parameter \theta
Require : Small constant \delta, usually 10^{-6}, used to stabilite division by small numbers
  Intiialize accumulation variables r = 0
  while stopping criterion not met do
    sample a minibatch of m examples from the training set {x^{(1)}, ..., x^{(m)}} with corresponding targets y^{(i)}
    compute gradient : 
```
<a href="https://www.codecogs.com/eqnedit.php?latex=g&space;\leftarrow&space;\frac{1}{m}&space;\triangledown_{\theta}&space;\sum_i&space;L(f(x^{(i)};&space;\theta),&space;y^{(i)})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?g&space;\leftarrow&space;\frac{1}{m}&space;\triangledown_{\theta}&space;\sum_i&space;L(f(x^{(i)};&space;\theta),&space;y^{(i)})" title="g \leftarrow \frac{1}{m} \triangledown_{\theta} \sum_i L(f(x^{(i)}; \theta), y^{(i)})" /></a>
```
    accumulate squared gradient :
```
<a href="https://www.codecogs.com/eqnedit.php?latex=r&space;\leftarrow&space;\rho&space;r&space;&plus;&space;(1-\rho)g\odot&space;g" target="_blank"><img src="https://latex.codecogs.com/gif.latex?r&space;\leftarrow&space;\rho&space;r&space;&plus;&space;(1-\rho)g\odot&space;g" title="r \leftarrow \rho r + (1-\rho)g\odot g" /></a>
```
    compute the parameter update :
```
<a href="https://www.codecogs.com/eqnedit.php?latex=\Delta&space;\theta&space;=&space;-\frac{\epsilon}{\sqrt{\delta&space;&plus;&space;r}}&space;\odot&space;g" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\Delta&space;\theta&space;=&space;-\frac{\epsilon}{\sqrt{\delta&space;&plus;&space;r}}&space;\odot&space;g" title="\Delta \theta = -\frac{\epsilon}{\sqrt{\delta + r}} \odot g" /></a>
```
    apply the update :
```
<a href="https://www.codecogs.com/eqnedit.php?latex=\theta&space;\leftarrow&space;\theta&space;&plus;&space;\Delta&space;\theta" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\theta&space;\leftarrow&space;\theta&space;&plus;&space;\Delta&space;\theta" title="\theta \leftarrow \theta + \Delta \theta" /></a>
```
end while
```

Next we have the Adam algorithm :

```
Require : step size \epsilon
Require : Exponential decay rates for moment estimates, \rho_1 and \rho_2 in [0,1)
Require : Small constant \elta used for numerical stabilization
Require : Initial parameters \theta
  Initialize 1st and 2nd moment variables s=0, r=0
  Initialize time step t=0
  while stopping criterion not met do
    sample a minibatch of m examples from the training set {x^{(1)},...,x^{(m)}} with corresponding targets y^{(i)}
    compute gradient :
```
<a href="https://www.codecogs.com/eqnedit.php?latex=g&space;\leftarrow&space;\frac{1}{m}&space;\triangledown_{\theta}&space;\sum_i&space;L(f(x^{(i)};&space;\theta),&space;y^{(i)})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?g&space;\leftarrow&space;\frac{1}{m}&space;\triangledown_{\theta}&space;\sum_i&space;L(f(x^{(i)};&space;\theta),&space;y^{(i)})" title="g \leftarrow \frac{1}{m} \triangledown_{\theta} \sum_i L(f(x^{(i)}; \theta), y^{(i)})" /></a>
```
    t = t+1
    update biased first moment estimate :
```
<a href="https://www.codecogs.com/eqnedit.php?latex=s&space;\leftarrow&space;\rho_1&space;s&space;&plus;&space;(1-\rho_1)g" target="_blank"><img src="https://latex.codecogs.com/gif.latex?s&space;\leftarrow&space;\rho_1&space;s&space;&plus;&space;(1-\rho_1)g" title="s \leftarrow \rho_1 s + (1-\rho_1)g" /></a>
```
    update biased second moment estimate :
```
<a href="https://www.codecogs.com/eqnedit.php?latex=r&space;\leftarrow&space;\rho_2&space;r&space;&plus;&space;(1-\rho_2)g&space;\odot&space;g" target="_blank"><img src="https://latex.codecogs.com/gif.latex?r&space;\leftarrow&space;\rho_2&space;r&space;&plus;&space;(1-\rho_2)g&space;\odot&space;g" title="r \leftarrow \rho_2 r + (1-\rho_2)g \odot g" /></a>
```
    correct bias in the first moment :
```
<a href="https://www.codecogs.com/eqnedit.php?latex=\hat{s}&space;\leftarrow&space;\frac{s}{1-\rho_1^t}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\hat{s}&space;\leftarrow&space;\frac{s}{1-\rho_1^t}" title="\hat{s} \leftarrow \frac{s}{1-\rho_1^t}" /></a>
```
    correct bias in the second moment :
```
<a href="https://www.codecogs.com/eqnedit.php?latex=\hat{r}&space;\leftarrow&space;\frac{r}{1-\rho_2^t}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\hat{r}&space;\leftarrow&space;\frac{r}{1-\rho_2^t}" title="\hat{r} \leftarrow \frac{r}{1-\rho_2^t}" /></a>
```
    compute the update 
```
<a href="https://www.codecogs.com/eqnedit.php?latex=\Delta&space;\theta&space;=&space;-\epsilon&space;\frac{\hat{s}}{\sqrt{\hat{r}&space;&plus;&space;\delta}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\Delta&space;\theta&space;=&space;-\epsilon&space;\frac{\hat{s}}{\sqrt{\hat{r}&space;&plus;&space;\delta}}" title="\Delta \theta = -\epsilon \frac{\hat{s}}{\sqrt{\hat{r} + \delta}}" /></a>
```
    Apply the update
```
<a href="https://www.codecogs.com/eqnedit.php?latex=\theta&space;\leftarrow&space;\theta&space;&plus;&space;\Delta&space;\theta" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\theta&space;\leftarrow&space;\theta&space;&plus;&space;\Delta&space;\theta" title="\theta \leftarrow \theta + \Delta \theta" /></a>
```
end while
```

The no free lunch theorem states that any two optimization algorithms are equivalent when their performance is averaged across all possible problems. We have the following distributed multi-region representation, where the number of regions cut out by n hyperplanes, in an d-dimensional input space :

<a href="https://www.codecogs.com/eqnedit.php?latex=\sum_{j=0}^d&space;\bigl(\begin{smallmatrix}&space;n\\&space;j&space;\end{smallmatrix}\bigr)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\sum_{j=0}^d&space;\bigl(\begin{smallmatrix}&space;n\\&space;j&space;\end{smallmatrix}\bigr)" title="\sum_{j=0}^d \bigl(\begin{smallmatrix} n\\ j \end{smallmatrix}\bigr)" /></a>

We assign arbitrary class labels {+1,0} to each region.

**Week 8**
