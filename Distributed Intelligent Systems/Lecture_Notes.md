# Distributed Intelligent Systems

**Week 1**

There are limited simple rules at the local level but at the global level the global emerging structure accomplishes some function. Here is the definition of swarm intelligence : decentralized control, lcack of synchronicity, simple and identical members, self organization. Swarm robotics is the study of how a large number of relatively simple physically embodied agents can be designed such that a desired collective behavior emerges from the local interactions among the agents and between the agents and the environment.

In self organization strutcures appear at the global level as a result of lower level interactions. The characteristics of such a state in a natural system is that spatio-temporal structures are created, you can scale this system, in such a system there are several parameters. This system has randomness, positive and negative feedback. Stigmergy defines a class of mechanisms exploited by social insects to coordinate and control their activity via indirect interactions. There is quantitative and qualitative stygmergy.

Collecting can be done in a group (tandem, recruitment strategies). Mediated by thropallis, antennal contact. The ant follows a specific sequence of moves when disocvering a new food source :

picking up food -> laying a chemical trail -> stimulating nest mates -> deposition of food -> following the trail

Stochastic individual behavior combined with the amplification of information can lead to collective decisions. Some experiments show that the longer the travelled path, then the smaller is the number of ants on the trail. Also we see then that the higher is the pheromone concentration, and the more reliably can the ants follow the trail.

When you do the suspended bridge experiment you find that if p_A denotes the probability for an ant to pick branch A, and p_B to pick branch B then we have :

<a href="https://www.codecogs.com/eqnedit.php?latex=p_A&space;=&space;\frac{(k&plus;A_i)^n}{(k&plus;A_i)^n&space;&plus;&space;(k&space;&plus;&space;B_i)^n}&space;=&space;1&space;-&space;p_B" target="_blank"><img src="https://latex.codecogs.com/gif.latex?p_A&space;=&space;\frac{(k&plus;A_i)^n}{(k&plus;A_i)^n&space;&plus;&space;(k&space;&plus;&space;B_i)^n}&space;=&space;1&space;-&space;p_B" title="p_A = \frac{(k+A_i)^n}{(k+A_i)^n + (k + B_i)^n} = 1 - p_B" /></a>

Where : 

A_i : is the number of ants having chosen branch A
B_i : is the number of ants having chosen branch B
n : degree of nonlinearity
k : degree of attraction of an unmarked branch

We then have :

<a href="https://www.codecogs.com/eqnedit.php?latex=A_{i&plus;1}&space;=&space;\left\{\begin{matrix}&space;A_{i&plus;1}&space;&&space;if&space;\delta&space;\leq&space;p_A\\&space;A_i&space;&&space;if&space;\delta&space;>&space;p_A&space;\end{matrix}\right." target="_blank"><img src="https://latex.codecogs.com/gif.latex?A_{i&plus;1}&space;=&space;\left\{\begin{matrix}&space;A_{i&plus;1}&space;&&space;if&space;\delta&space;\leq&space;p_A\\&space;A_i&space;&&space;if&space;\delta&space;>&space;p_A&space;\end{matrix}\right." title="A_{i+1} = \left\{\begin{matrix} A_{i+1} & if \delta \leq p_A\\ A_i & if \delta > p_A \end{matrix}\right." /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=B_{i&plus;1}&space;=&space;\left\{\begin{matrix}&space;B_{i&plus;1}&space;&&space;if&space;\delta&space;\leq&space;p_B\\&space;B_i&space;&&space;if&space;\delta&space;>&space;p_B&space;\end{matrix}\right." target="_blank"><img src="https://latex.codecogs.com/gif.latex?B_{i&plus;1}&space;=&space;\left\{\begin{matrix}&space;B_{i&plus;1}&space;&&space;if&space;\delta&space;\leq&space;p_B\\&space;B_i&space;&&space;if&space;\delta&space;>&space;p_B&space;\end{matrix}\right." title="B_{i+1} = \left\{\begin{matrix} B_{i+1} & if \delta \leq p_B\\ B_i & if \delta > p_B \end{matrix}\right." /></a>

A_i + B_i = i, and \delta is a uniform random variables on [0,1].

When n is high we have high exploitation. When k is high we have high exploration. 

For most social insects the fundamental ecological unit is the colony. A collection of nests or sub-colonies forms what is called a super-colony. Ant colony optimization is the field of study centered around applying ideas from natural systems to digital artificial systems. ACO algorithms are examples of exploitation of swarm intelligence principles as a particular form of distributed intelligence. The travelling salesman problem tries to find the shortest path which allows the salesman to visit once and only once each city in the graph. Ants build solutions probabilistically without updating pheromone trails. Ants deterministically backward retrace the forward path to update the pheromones. Ants deposit a quantity of pheromone function of the quality of the solution that they generate. 

In the TSP problem we have :

<a href="https://www.codecogs.com/eqnedit.php?latex=\eta_{ij}&space;=&space;1/d_{ij}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\eta_{ij}&space;=&space;1/d_{ij}" title="\eta_{ij} = 1/d_{ij}" /></a>

which denotes the visibility, this information is static. We have the following AS for the TSP algorithm.

```
Loop
  place one ant on each node
  for k = 1 to m
    for step = 1 to n
      choose the next node to move by applying a probabilistic state transition rule
    end for
  end for
until end condition
```

During a tour T, an ant k at node i decided to move towards j with the following probability :

<a href="https://www.codecogs.com/eqnedit.php?latex=p^k_{ij}(t)&space;=&space;0" target="_blank"><img src="https://latex.codecogs.com/gif.latex?p^k_{ij}(t)&space;=&space;0" title="p^k_{ij}(t) = 0" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=p^k_{ij}(t)&space;=&space;\frac{[\tau_{ij}(t)]^{\alpha}&space;[\eta_{ij}]^{\beta}}{\sum_{j&space;\in&space;J_i^k}&space;[\tau_{ij}(t)]^{\alpha}[\eta_{ij}]^{\beta}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?p^k_{ij}(t)&space;=&space;\frac{[\tau_{ij}(t)]^{\alpha}&space;[\eta_{ij}]^{\beta}}{\sum_{j&space;\in&space;J_i^k}&space;[\tau_{ij}(t)]^{\alpha}[\eta_{ij}]^{\beta}}" title="p^k_{ij}(t) = \frac{[\tau_{ij}(t)]^{\alpha} [\eta_{ij}]^{\beta}}{\sum_{j \in J_i^k} [\tau_{ij}(t)]^{\alpha}[\eta_{ij}]^{\beta}}" /></a>

Here we have that alpha is the parameter controlling the includence of the virtual pheromone, while beta is the parameter controlling the influence of the local visibility. 

At the end of each tour T, each ant K deposits a quantity of virtual pheromones \Delta \tau_{ij}^k = 0, when (i,j) has not been used during the tour T. And the following quantity when the link (i,j) instead has been used during the tour T. 

<a href="https://www.codecogs.com/eqnedit.php?latex=\Delta&space;\tau_{ij}^k&space;=&space;\frac{Q}{L^k(t)}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\Delta&space;\tau_{ij}^k&space;=&space;\frac{Q}{L^k(t)}" title="\Delta \tau_{ij}^k = \frac{Q}{L^k(t)}" /></a>

Where here we have that : L_k^t is the length of the tour T done by ant k at iteration t, and where Q is the parameter adjusted by heuristic and not sensitive. We have the following AS for TSP algorithm. We have :

<a href="https://www.codecogs.com/eqnedit.php?latex=\tau_{ij}(t&plus;1)&space;\leftarrow&space;(1-\rho)\tau_{ij}(t)&plus;\sum_{k=1}^m&space;\Delta&space;\tau_{ij}(t)^k" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\tau_{ij}(t&plus;1)&space;\leftarrow&space;(1-\rho)\tau_{ij}(t)&plus;\sum_{k=1}^m&space;\Delta&space;\tau_{ij}(t)^k" title="\tau_{ij}(t+1) \leftarrow (1-\rho)\tau_{ij}(t)+\sum_{k=1}^m \Delta \tau_{ij}(t)^k" /></a>

When we update with elitism : 

<a href="https://www.codecogs.com/eqnedit.php?latex=\tau_{ij}(t&plus;1)&space;\leftarrow&space;(1-\rho)\tau_{ij}(t)&plus;\sum_{k=1}^m&space;\Delta&space;\tau_{ij}(t)^k&space;&plus;&space;e&space;\Delta&space;\tau_{ij}^e&space;(t)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\tau_{ij}(t&plus;1)&space;\leftarrow&space;(1-\rho)\tau_{ij}(t)&plus;\sum_{k=1}^m&space;\Delta&space;\tau_{ij}(t)^k&space;&plus;&space;e&space;\Delta&space;\tau_{ij}^e&space;(t)" title="\tau_{ij}(t+1) \leftarrow (1-\rho)\tau_{ij}(t)+\sum_{k=1}^m \Delta \tau_{ij}(t)^k + e \Delta \tau_{ij}^e (t)" /></a>

Next we also have the following :

<a href="https://www.codecogs.com/eqnedit.php?latex=\Delta&space;\tau_{ij}^e&space;(t)&space;=&space;\left\{\begin{matrix}&space;\frac{Q}{L^&plus;}&space;&&space;\textrm{if&space;(i,j)&space;belongs&space;to&space;the&space;best&space;tour&space;T&plus;}&space;\\&space;0&space;&&space;\textrm{otherwise}&space;\end{matrix}\right." target="_blank"><img src="https://latex.codecogs.com/gif.latex?\Delta&space;\tau_{ij}^e&space;(t)&space;=&space;\left\{\begin{matrix}&space;\frac{Q}{L^&plus;}&space;&&space;\textrm{if&space;(i,j)&space;belongs&space;to&space;the&space;best&space;tour&space;T&plus;}&space;\\&space;0&space;&&space;\textrm{otherwise}&space;\end{matrix}\right." title="\Delta \tau_{ij}^e (t) = \left\{\begin{matrix} \frac{Q}{L^+} & \textrm{if (i,j) belongs to the best tour T+} \\ 0 & \textrm{otherwise} \end{matrix}\right." /></a>

**Week 2**

The approximate solution of NP-hard combinatorial optimization problems in the coupling of a constructive heuristic (generate solutions from scratch by iteratively adding solution components), local search (start from some initial solution and repeatedly tries to improve by local changes). There are two extensions of the AS algorithm - there is the ant colony system algorithm, and the ACS-3-opt. We have the following ant-colony system algorithm : 

```
Loop \* t=0; t:=t+1 \* 
  Place one ant on each node \*there are n nodes \* 
  Fork := 1 to m \* each ant builds a solution, in this case m=n\* 
    For step := 1 to n \* each ant adds a node to its path \* 
      Choose the next city to move by applying a probabilistic solution construction rule 
    End-for 
  End-for 
Update pheromone trails UntilEnd_condition \* e.g., t=tmax \*
```

An ant k on city i chooses city k to move according to the following rule :

<a href="https://www.codecogs.com/eqnedit.php?latex=j&space;=&space;\left\{\begin{matrix}&space;arg&space;max_{u&space;\in&space;J_i^k}&space;\{&space;[\tau_{iu}(t)][\eta_{iu}]^{\beta}\}&space;&&space;\textrm{if&space;q&space;is&space;less&space;than&space;or&space;equal&space;to&space;q0}&space;\\&space;J&space;&&space;\textrm{if&space;q&space;is&space;greater&space;that&space;q0}&space;\end{matrix}\right." target="_blank"><img src="https://latex.codecogs.com/gif.latex?j&space;=&space;\left\{\begin{matrix}&space;arg&space;max_{u&space;\in&space;J_i^k}&space;\{&space;[\tau_{iu}(t)][\eta_{iu}]^{\beta}\}&space;&&space;\textrm{if&space;q&space;is&space;less&space;than&space;or&space;equal&space;to&space;q0}&space;\\&space;J&space;&&space;\textrm{if&space;q&space;is&space;greater&space;that&space;q0}&space;\end{matrix}\right." title="j = \left\{\begin{matrix} arg max_{u \in J_i^k} \{ [\tau_{iu}(t)][\eta_{iu}]^{\beta}\} & \textrm{if q is less than or equal to q0} \\ J & \textrm{if q is greater that q0} \end{matrix}\right." /></a>

Where J is in J_i^k being a city that is randomly selected according to the following rule :

<a href="https://www.codecogs.com/eqnedit.php?latex=p_{iJ}^k(t)&space;=&space;\frac{[\tau_{iJ}(t)][\eta_{iJ}]^{\beta}}{\sum_{l&space;\in&space;J_i^k}[\tau_{il}(t)][\eta_{il}]^{\beta}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?p_{iJ}^k(t)&space;=&space;\frac{[\tau_{iJ}(t)][\eta_{iJ}]^{\beta}}{\sum_{l&space;\in&space;J_i^k}[\tau_{il}(t)][\eta_{il}]^{\beta}}" title="p_{iJ}^k(t) = \frac{[\tau_{iJ}(t)][\eta_{iJ}]^{\beta}}{\sum_{l \in J_i^k}[\tau_{il}(t)][\eta_{il}]^{\beta}}" /></a>

AS : all ants can update the pheromone trails in the same way 
EAS : all ants update pheromone trails, extra amount for the best tour
ACS :  the global update is performed exclusively by theant that generated the best tourfrom the beginning of the trial; it updates only the edges of the best tour T^+ of length L^+ since the beginning of the trial

The update rule for (i,j) edges belonging to T^+ where :

<a href="https://www.codecogs.com/eqnedit.php?latex=\tau_{ij}(t&plus;1)&space;\leftarrow&space;(1-\rho)&space;\tau_{ij}(t)&space;&plus;&space;\rho&space;\Delta&space;\tau_{ij}(t)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\tau_{ij}(t&plus;1)&space;\leftarrow&space;(1-\rho)&space;\tau_{ij}(t)&space;&plus;&space;\rho&space;\Delta&space;\tau_{ij}(t)" title="\tau_{ij}(t+1) \leftarrow (1-\rho) \tau_{ij}(t) + \rho \Delta \tau_{ij}(t)" /></a>

where

<a href="https://www.codecogs.com/eqnedit.php?latex=\Delta&space;\tau_{ij}(t)&space;=&space;\frac{1}{L^&plus;}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\Delta&space;\tau_{ij}(t)&space;=&space;\frac{1}{L^&plus;}" title="\Delta \tau_{ij}(t) = \frac{1}{L^+}" /></a>

A candidate list is a list of cities of length cl where cl is the algorithmic parameter, to be visited from a given city. The cities are ranked according to th inverse of their distance. An ant retricts the choice of the next city to those in the candidate list. 

We have the next ACS algorithm with local search :

```
Loop \* t=0; t:=t+1 \* 
  Place one ant on each node \*there are n nodes \* 
  Fork := 1 to m \* each ant builds a solution, in this case m=n\* 
    For step := 1 to n \* each ant adds a node to its path \* 
      Choose the next city to move by applying a probabilistic solution construction rule 
    End-for 
  Apply local search
  End-for 
  Update pheromone trails
Update pheromone trails UntilEnd_condition \* e.g., t=tmax \*
```
Now we consider what k-opt Heuristic could be :

Take a give tour and delete up to k mutually disjoint edges. Each fragment endpoint can be connected to 2k-2 other possibilities. Reassemble the remainin fragments into a tour, leaving no disjoint subtours. Do the following systematically : generate the set of all candidate solutions possible by exchanging in all possible ways up to k edges. Local search is complementary to ant pheromone mechanisms. This lacks in good starting solutions on which it can perform combinatorial optimization.

Ant-based control algorithm ABC. Node i has maximal capacity C_i and spare capacity S_i (capacity available for new connections). Once a call is set-up between destinationdand source s, each node in the route is decreased in its spare capacity by one connection (multiple connections if the node is used by multiple routes). The routing table was such that for node i we have :

<a href="https://www.codecogs.com/eqnedit.php?latex=R_i&space;=&space;[r^i_{n,d}(t)]_{k_i,&space;N-1}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?R_i&space;=&space;[r^i_{n,d}(t)]_{k_i,&space;N-1}" title="R_i = [r^i_{n,d}(t)]_{k_i, N-1}" /></a>

Here we have that r_{n,d}^i(t) for ants is the probability that an ant with destination d will be routed from i to neighbor n, and similarly for calls we have a deterministic path. We have the sum of those over n we get one. Each visited node's routing table is updated according to :

<a href="https://www.codecogs.com/eqnedit.php?latex=r^{i}_{i-1,s}(t&plus;1)&space;=&space;\frac{r^{i}_{i-1,s}(t)&space;&plus;&space;\delta&space;r}{1&space;&plus;&space;\delta&space;r}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?r^{i}_{i-1,s}(t&plus;1)&space;=&space;\frac{r^{i}_{i-1,s}(t)&space;&plus;&space;\delta&space;r}{1&space;&plus;&space;\delta&space;r}" title="r^{i}_{i-1,s}(t+1) = \frac{r^{i}_{i-1,s}(t) + \delta r}{1 + \delta r}" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=r^{i}_{n,s}(t&plus;1)&space;=&space;\frac{r^{i}_{n,s}(t)}{1&space;&plus;&space;\delta&space;r}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?r^{i}_{n,s}(t&plus;1)&space;=&space;\frac{r^{i}_{n,s}(t)}{1&space;&plus;&space;\delta&space;r}" title="r^{i}_{n,s}(t+1) = \frac{r^{i}_{n,s}(t)}{1 + \delta r}" /></a>

Where here \delta r is the reinforcement parameter, and i-1 is the neighbor node the ant came from before joining i. We have the following form for \delta r where a and b are parameters and T is the absolute time spent in the network : 

<a href="https://www.codecogs.com/eqnedit.php?latex=\delta&space;r&space;=&space;\frac{a}{T}&space;&plus;&space;b" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\delta&space;r&space;=&space;\frac{a}{T}&space;&plus;&space;b" title="\delta r = \frac{a}{T} + b" /></a>

A delay is imposed on an ant reaching a give node i, where c and d are parameters and S_i is the spare capacity :

<a href="https://www.codecogs.com/eqnedit.php?latex=D_i&space;=&space;c&space;e^{-d&space;S_i}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?D_i&space;=&space;c&space;e^{-d&space;S_i}" title="D_i = c e^{-d S_i}" /></a>

We have the following AntNet algorithm : ants are launched from each node and they build their paths wih a probability function of artificial pheromone values and heuristic values, ants memorize the visited nodes and elapsed times, once reached their destination nodes, ants retraces their paths backwards and update the phereomone trails and trip vectors. We have the following equation :

<a href="https://www.codecogs.com/eqnedit.php?latex=p_{ijd}^k&space;(t)&space;=&space;f(\tau_{ijd}(t),&space;\eta_{ij}(t))" target="_blank"><img src="https://latex.codecogs.com/gif.latex?p_{ijd}^k&space;(t)&space;=&space;f(\tau_{ijd}(t),&space;\eta_{ij}(t))" title="p_{ijd}^k (t) = f(\tau_{ijd}(t), \eta_{ij}(t))" /></a>

Where here \tau_{ijd} is the pheromone trail normalized to one for all possible neighboring nodes. Also \eta_{ij} is a heuristic evaluation of link (i,j) which introduces problem specific information. We have routing table element R_i which memorizes the probabilities of choosing each neighbor nodes for each possible final destination. The trip vectors \Gamma_i, contains statistics about ants' trip times from current node i to each destination node d. 

Standard measures of performance are the throughput which have units bits/sec, and the average packet delay in seconds. Non-max throughput means retransmissions, error notifications, augmented congestions. Why do ant-based systems work ? There are three important components. The time, because a shorter path receives pheromones quicker, the quality, a shorter path receives more pheromones.

A metaheuristicis a set of algorithmic concepts that can be used to define or organize heuristic methods applicable to a wide set of different problems. Ant System and AntNet have been extended so that they can be applied to any shortest path problems on graphs. The resulting extension is called the Ant Colony Optimization metaheuristic. The corresponding procedure is as follows :

```
procedure ACO-metaheuristics()
  while (non termination criterion)
    schedule sub-procedures
      generate-&-manage-ants()
      execute-daemon-actions()
      update-pheromones()
    end schedule sub-procedures
  end while
end procedure
```

**Week 3**

There are different degrees in the autonomy. We see that the more the task is complex the more the humans must guide the robots. The perception to action loop is reactive and deliberative. We have that the sensors perceive, there is a computation done, an action taken and the environment is affected. We have different types of sensors : propioceptive which are related to the body and exteroceptive related to the environment. There are passive and active sensors which actually interact with the environment. Examples include tactile sensors, wheel and motor sensors and heading sensors, ground-based beacons, motion/speed sensors and vision-based sensors.

When we talk about sensors, the dynamic range is the ratio between the lower and the upper limits, this is usually in decibels when it concernes the power and the amplitude. There is also the resolution which is the minimum difference between two values. The lower limit of the dynamic range is the resolution. The bandwidth or the frequency is the speed with which a sensor can provide a stream of readings. The sensors performance is influences by the following charactersitics : the sensitivity, the cross-sensitivity, the error/accuracy. There can also be systematic errors related to for example the calibration of the sensor, there can also be random errors for which there is no deterministic prediction before hand. There is the measure of the precision, which calculates the reproducibility of the sensor results.

The webots principles are that the simulator process leads to the controller process, which lead to the controller code. A proximal architecture is close to sensors and actuators, it has a high flexibility in shaping the behavior and it is difficult to engineer in a human-guided way. Distal architectures have self-contained behavioral blocks, there is less flexibillity in shaping the behavior.

The subsumption architecture is such that the input lines go into the behavioral module, the inhibitor blocks the transmission, there are output lines which also some pass through the supressor. The supressor blocks the transmission and replaces the signal with the suppressing message. The subsumption has support for parallelism, each behavioral layer can run independently and asynchronously.

**Week 4**

An introduction to localization methods for mobile robots. Robot localization is a key task for path planning, mapping, referencing, coordination. There are indoor positioning sytems : motion capture systems MCSs, Impulse Radio Ultra Wide Band (IR-UWB). There are also overhead multi-camera systems. The MCSs consist of 10-50 cameras, with 4 to 5 passive markers per object to be tracked needed. The IR-UWB system on the other hand is based on the time-of-flight system. Emitters can be unsynchronized, positioning can be fed back to the robots using a standard narrow-band channel. In infrared and radio technology, there is a belt of IR emitters LED and receivers (photodiodes). The range is the measurement of the received signal strength intensity RSSI. The bearing is the signal correlation over multiple receivers.

Let's study now the Global Positioning System GPS. The location of any GPS receiver is determined through a time of flight measurement. Odometry is the us of proprioceptive sensory data influenced by the movement of actuators to estimate change in pose over time. Odometry can be studied with wheel encoders. Optical encoders measure the displacement of the wheels. Recall the following rule for mechanical physics :

<a href="https://www.codecogs.com/eqnedit.php?latex=v&space;=&space;\omega&space;r&space;=&space;\dot{\phi}&space;r" target="_blank"><img src="https://latex.codecogs.com/gif.latex?v&space;=&space;\omega&space;r&space;=&space;\dot{\phi}&space;r" title="v = \omega r = \dot{\phi} r" /></a>

Where here omega is the rotational speed and the derivative of phi is the derivative of the rotation angle. Here v is the tangential speed. We have the following linear speed for the car with two wheels :

<a href="https://www.codecogs.com/eqnedit.php?latex=v&space;=&space;\frac{r&space;\dot{\phi}_1}{2}&space;&plus;&space;\frac{r&space;\dot{\phi}_2}{2}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?v&space;=&space;\frac{r&space;\dot{\phi}_1}{2}&space;&plus;&space;\frac{r&space;\dot{\phi}_2}{2}" title="v = \frac{r \dot{\phi}_1}{2} + \frac{r \dot{\phi}_2}{2}" /></a>

We also have the following rotational speed for the car with two wheels :

<a href="https://www.codecogs.com/eqnedit.php?latex=\omega&space;=&space;\frac{r&space;\dot{\phi}_1}{2l}&space;&plus;&space;\frac{-r&space;\dot{\phi}_2}{2l}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\omega&space;=&space;\frac{r&space;\dot{\phi}_1}{2l}&space;&plus;&space;\frac{-r&space;\dot{\phi}_2}{2l}" title="\omega = \frac{r \dot{\phi}_1}{2l} + \frac{-r \dot{\phi}_2}{2l}" /></a>

Given the kinematic forward model and assuming no slip on both wheels we have : 

<a href="https://www.codecogs.com/eqnedit.php?latex=\epsilon_I(T)&space;=&space;\epsilon_{I_0}&space;&plus;&space;\int_0^T&space;\dot{\epsilon_I}&space;dt&space;=&space;\epsilon_{I_0}&space;&plus;&space;\int_0^T&space;R^{-1}(\theta)&space;\dot{\epsilon}_R&space;dt" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\epsilon_I(T)&space;=&space;\epsilon_{I_0}&space;&plus;&space;\int_0^T&space;\dot{\epsilon_I}&space;dt&space;=&space;\epsilon_{I_0}&space;&plus;&space;\int_0^T&space;R^{-1}(\theta)&space;\dot{\epsilon}_R&space;dt" title="\epsilon_I(T) = \epsilon_{I_0} + \int_0^T \dot{\epsilon_I} dt = \epsilon_{I_0} + \int_0^T R^{-1}(\theta) \dot{\epsilon}_R dt" /></a>

There are non-deterministic error sources in odometry based on wheel encoders. This is because of the variation of the contact point of the wheel, and because there is unequal floor contact due to the wheel slip, nonplanar surfaces. There are three types of odometric error types : range error (sum of the wheel movements), turn error (difference of wheel motion), drift error (difference between wheel errors lead to heading error).

Actuator noise leads to poise noise. First we precompute \Sigma_{\Delta}, compute the mapping actuator-to-poise noise incremental F_{\Delta rl}, and compute the mapping pose propagation noise over step F_p. We then have the following equation:

<a href="https://www.codecogs.com/eqnedit.php?latex=\Sigma_p^{(t=(k&plus;1)\Delta&space;t)}&space;=&space;F_p&space;\Sigma_p^{(t=k\Delta&space;t)}F_p^T&space;&plus;&space;F_{\Delta&space;rl}&space;\Sigma_{\Delta}&space;F_{\Delta&space;rl}^T" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\Sigma_p^{(t=(k&plus;1)\Delta&space;t)}&space;=&space;F_p&space;\Sigma_p^{(t=k\Delta&space;t)}F_p^T&space;&plus;&space;F_{\Delta&space;rl}&space;\Sigma_{\Delta}&space;F_{\Delta&space;rl}^T" title="\Sigma_p^{(t=(k+1)\Delta t)} = F_p \Sigma_p^{(t=k\Delta t)}F_p^T + F_{\Delta rl} \Sigma_{\Delta} F_{\Delta rl}^T" /></a>

**Week 5**

The flocking phenomenon is such that there are no collisions between members, there are reactivities to predators and obstacles. The benefits of flocking are that energy is saved, navigation accuracy. Boid's flight model says that there is momentum conservation, maximal acceleration, maximal speed via viscuous friction, some gravity and aerodynamic lift, wings flapping independently. We have the following Reynolds' rules for flocking : separation (avoid collisions with nearby flockmates), alignment (attempt to match velocity), cohesion (attempt to stay close to nearby flockmates).

The arbitraring rules say that time-constant linear weighted sum did not work in front of obstacles, whilst time-varying nonlinear weighted sum worked much better. Here we have that separation is more important than alignment which is more important than cohesion. Consider the sensory system for teammate detection. Here we have local, almost omni-directional sensory systems. Here we have the perfect relative range and bearing system. We have one perception-to-action loop, a homogeneous system and natural nonlinearities.

In general homogeneous systems impossible, immediate response impossible. Identifier for each teammate possible but scalability issues. Depending on the system used for range and bearing : occlusion is possible. Nonlinearities are determined by the underlying technology. We have motor-schema-based formation control where you move to the goal, avoid static obstacles, avoid the robot and maintain the formation. The formtion taxonomy can be as follows : unit-center-referenced, leader-referenced, neighbor-referenced.

Fredslung and Mataric have said in 2002 that the architecture is neighbor-references and is based on an on-board relative positioning, single leader always. In 2002 Fredslung and Mataric have created a combined use of laser range finder LRF and a pan camera. Formations can be divided into two categories : location-based (robots group must maintain fixed locations between teammates), heading-based (robots must maintain fixed location nd headings relative to teammates). We have the following modes for the formation localization modes. 

Mode 1 : no relative positioning - robots follow pre-programmed course with no closed-loop feedback
Mode 2 : relative positionning - robots observe teammates with relative positioning module and attempt to maintain proper locations
Mode 3 : relative positioning with communication - robots observe and share information with leader robot using relative positioning and wireless radio

Let's study the continuous consensus algorithms, the graph-based distributed control. Consider the following definition of the incidence matrix :

Define I in R^{|V| x |E|} as :

<a href="https://www.codecogs.com/eqnedit.php?latex=I(i,j)&space;=&space;\left\{\begin{matrix}&space;-1&space;&&space;\textrm{if&space;ej&space;leaves&space;ni}\\&space;&plus;1&space;&&space;\textrm{if&space;ej&space;enters&space;ni}\\&space;0&space;&&space;\textrm{otherwise}&space;\end{matrix}\right." target="_blank"><img src="https://latex.codecogs.com/gif.latex?I(i,j)&space;=&space;\left\{\begin{matrix}&space;-1&space;&&space;\textrm{if&space;ej&space;leaves&space;ni}\\&space;&plus;1&space;&&space;\textrm{if&space;ej&space;enters&space;ni}\\&space;0&space;&&space;\textrm{otherwise}&space;\end{matrix}\right." title="I(i,j) = \left\{\begin{matrix} -1 & \textrm{if ej leaves ni}\\ +1 & \textrm{if ej enters ni}\\ 0 & \textrm{otherwise} \end{matrix}\right." /></a>

We define the following weight matrix, where W in R^{|E| x |E|} as :

<a href="https://www.codecogs.com/eqnedit.php?latex=W(i,j)&space;=&space;\left\{\begin{matrix}&space;w_i&space;&&space;\textrm{if&space;i=j}\\&space;0&space;&&space;\textrm{otherwise}&space;\end{matrix}\right." target="_blank"><img src="https://latex.codecogs.com/gif.latex?W(i,j)&space;=&space;\left\{\begin{matrix}&space;w_i&space;&&space;\textrm{if&space;i=j}\\&space;0&space;&&space;\textrm{otherwise}&space;\end{matrix}\right." title="W(i,j) = \left\{\begin{matrix} w_i & \textrm{if i=j}\\ 0 & \textrm{otherwise} \end{matrix}\right." /></a>

Consider the laplacian matrix where L in R^{|V| x |V|} as :

<a href="https://www.codecogs.com/eqnedit.php?latex=L&space;=&space;I&space;\cdot&space;W&space;\cdot&space;I^T" target="_blank"><img src="https://latex.codecogs.com/gif.latex?L&space;=&space;I&space;\cdot&space;W&space;\cdot&space;I^T" title="L = I \cdot W \cdot I^T" /></a>

One way to solve the rendez-vous problem is to use the laplacian matrix and write:

<a href="https://www.codecogs.com/eqnedit.php?latex=\dot{x}(t)&space;=&space;-L&space;x(t)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dot{x}(t)&space;=&space;-L&space;x(t)" title="\dot{x}(t) = -L x(t)" /></a>

Holonomic robots is such that the total number of degrees of freedom is the number of controllable degrees of freedom. A robot is holonomic if it can move in any direction at any point in time. The Laplacian method gives the direction vector at each point in time.

<a href="https://www.codecogs.com/eqnedit.php?latex=\left\{\begin{matrix}&space;u&space;=&space;K_u&space;\sqrt{\dot{x}^2&space;&plus;&space;\dot{y}^2}&space;cos(atan2(\dot{y},&space;\dot{x}))\\&space;w&space;=&space;K_w&space;atan2(\dot{y},&space;\dot{x})&space;\end{matrix}\right." target="_blank"><img src="https://latex.codecogs.com/gif.latex?\left\{\begin{matrix}&space;u&space;=&space;K_u&space;\sqrt{\dot{x}^2&space;&plus;&space;\dot{y}^2}&space;cos(atan2(\dot{y},&space;\dot{x}))\\&space;w&space;=&space;K_w&space;atan2(\dot{y},&space;\dot{x})&space;\end{matrix}\right." title="\left\{\begin{matrix} u = K_u \sqrt{\dot{x}^2 + \dot{y}^2} cos(atan2(\dot{y}, \dot{x}))\\ w = K_w atan2(\dot{y}, \dot{x}) \end{matrix}\right." /></a>

We can also use the relative range and bearing in the non-holonomicity as follows:

<a href="https://www.codecogs.com/eqnedit.php?latex=\left\{\begin{matrix}&space;u&space;=&space;K_u&space;e&space;cos&space;\alpha\\&space;w&space;=&space;K_w&space;\alpha&space;\end{matrix}\right." target="_blank"><img src="https://latex.codecogs.com/gif.latex?\left\{\begin{matrix}&space;u&space;=&space;K_u&space;e&space;cos&space;\alpha\\&space;w&space;=&space;K_w&space;\alpha&space;\end{matrix}\right." title="\left\{\begin{matrix} u = K_u e cos \alpha\\ w = K_w \alpha \end{matrix}\right." /></a>

By adding a bias vector, we can modify the state (or assumed position) as follows:

<a href="https://www.codecogs.com/eqnedit.php?latex=\dot{x}&space;=&space;-&space;L(x(t)&space;-&space;B)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dot{x}&space;=&space;-&space;L(x(t)&space;-&space;B)" title="\dot{x} = - L(x(t) - B)" /></a>

In obstacle avoidance, each robot updates its neighbors list if necessary by adding a repulsive agent.  Positive weights will attract vehicles together, negative weights will create a repulsion mechanism. The proportional, integral controller says that :

<a href="https://www.codecogs.com/eqnedit.php?latex=\left\{\begin{matrix}&space;u&space;=&space;K_u&space;e&space;cos&space;\alpha&space;&plus;&space;K_I&space;\int_0^t&space;e&space;dt\\&space;w&space;=&space;K_w&space;\alpha&space;\end{matrix}\right." target="_blank"><img src="https://latex.codecogs.com/gif.latex?\left\{\begin{matrix}&space;u&space;=&space;K_u&space;e&space;cos&space;\alpha&space;&plus;&space;K_I&space;\int_0^t&space;e&space;dt\\&space;w&space;=&space;K_w&space;\alpha&space;\end{matrix}\right." title="\left\{\begin{matrix} u = K_u e cos \alpha + K_I \int_0^t e dt\\ w = K_w \alpha \end{matrix}\right." /></a>

**Week 6**

Modelling is needed to understand the interplay of the various elements of the system, and formally analyze the system properties. There are four types of modelling choices : the gray-box approach, the probabilistic approach, the multi-level approach and the bottom-up approach. The gray-box approach allows us to easily incorporate a priori information and aims at modelling interpretability. The probabilistic approach aims to capture noisy interactions, noisy robotic components, stochastic control policies and enables the aggragation schemes towards abstraction. The multi-level approach aims to represent explicitly different design choices, trade off computational speed and faithfulness to reality. The bottom-up approach aims to start from the physical reality and increases the abstraction level until the highest abstraction level. The multi-level modelling methodology consists of four possible layers, we have the macroscopic layer (representation of the whole swarm), the microscopic level (multi-agent models where only relevant robot features are captured), the submicroscopic model (meaning intra-robot and the environment), the target system (information on the controller).

We have the following invariant experimental features : short-range, proximity sensing, local communication and teammate sensing, full mobility and basic navigation, reactive and behavior-based control, not overcrowded arenas, multiple runs for the same experimental parameters. Non-spatial metrics are for the collective performance. Suppose that the Markov property is fulfilled then we have the following equality where p(n,t) is the probability of an agent to be in the state n at time t.

<a href="https://www.codecogs.com/eqnedit.php?latex=\Delta&space;p(n,t)&space;=&space;p(n,&space;t&space;&plus;&space;\Delta&space;t)&space;-&space;p(n,t)&space;=&space;\sum_{n'}&space;p(n,&space;t&space;&plus;&space;\Delta&space;t&space;|&space;n',t)&space;p(n',t)&space;-&space;\sum_{n'}&space;p(n',&space;t&plus;\Delta&space;t&space;|&space;n,t)&space;p(n,t)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\Delta&space;p(n,t)&space;=&space;p(n,&space;t&space;&plus;&space;\Delta&space;t)&space;-&space;p(n,t)&space;=&space;\sum_{n'}&space;p(n,&space;t&space;&plus;&space;\Delta&space;t&space;|&space;n',t)&space;p(n',t)&space;-&space;\sum_{n'}&space;p(n',&space;t&plus;\Delta&space;t&space;|&space;n,t)&space;p(n,t)" title="\Delta p(n,t) = p(n, t + \Delta t) - p(n,t) = \sum_{n'} p(n, t + \Delta t | n',t) p(n',t) - \sum_{n'} p(n', t+\Delta t | n,t) p(n,t)" /></a>

Then we have the following equality :

<a href="https://www.codecogs.com/eqnedit.php?latex=\frac{d&space;N_n&space;(t)}{dt}&space;=&space;\sum_{n'}&space;W(n&space;|&space;n',&space;t)&space;N_{n'}(t)&space;-&space;\sum_{n'}&space;W(n'&space;|&space;n,t)&space;N_n(t)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{d&space;N_n&space;(t)}{dt}&space;=&space;\sum_{n'}&space;W(n&space;|&space;n',&space;t)&space;N_{n'}(t)&space;-&space;\sum_{n'}&space;W(n'&space;|&space;n,t)&space;N_n(t)" title="\frac{d N_n (t)}{dt} = \sum_{n'} W(n | n', t) N_{n'}(t) - \sum_{n'} W(n' | n,t) N_n(t)" /></a>

Where here n and n' are the states of the agents (all the possible states at each instant), and where N_n is the average fraction of agents in state n at time t. Where we have the following equality :

<a href="https://www.codecogs.com/eqnedit.php?latex=W(n&space;|&space;n',&space;t)&space;=&space;\lim_{\Delta&space;t&space;\rightarrow&space;0}&space;\frac{p(n,&space;t&plus;\Delta&space;t&space;|&space;n',t)}{\Delta&space;t}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?W(n&space;|&space;n',&space;t)&space;=&space;\lim_{\Delta&space;t&space;\rightarrow&space;0}&space;\frac{p(n,&space;t&plus;\Delta&space;t&space;|&space;n',t)}{\Delta&space;t}" title="W(n | n', t) = \lim_{\Delta t \rightarrow 0} \frac{p(n, t+\Delta t | n',t)}{\Delta t}" /></a>

We have the following time-discrete rate equation :

<a href="https://www.codecogs.com/eqnedit.php?latex=N_n((k&plus;1)T)&space;=&space;N_n(kT)&space;&plus;&space;\sum_{n'}&space;TW(n&space;|&space;n',&space;kT)&space;N_{n'}(kT)&space;-&space;\sum_{n'}&space;TW(n'&space;|&space;n,&space;kT)&space;N_n(kT)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?N_n((k&plus;1)T)&space;=&space;N_n(kT)&space;&plus;&space;\sum_{n'}&space;TW(n&space;|&space;n',&space;kT)&space;N_{n'}(kT)&space;-&space;\sum_{n'}&space;TW(n'&space;|&space;n,&space;kT)&space;N_n(kT)" title="N_n((k+1)T) = N_n(kT) + \sum_{n'} TW(n | n', kT) N_{n'}(kT) - \sum_{n'} TW(n' | n, kT) N_n(kT)" /></a>

Where k is the iteration index, T is the time step, TW is the transition probability per time step. Consider the time disretization algorithm : assess what’s the time resolution needed for your system performance metrics, choose whenever possible the most computationally efficient model,  a single common sampling rate can be defined among different modeling levels. 

<a href="https://www.codecogs.com/eqnedit.php?latex=N_n(k&plus;1)&space;=&space;N_n(k)&space;&plus;&space;\sum_{n'}&space;P(n&space;|&space;n',&space;k)&space;N_{n'}(k)&space;-&space;\sum_{n'}&space;P(n'&space;|&space;n,k)&space;N_n(k)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?N_n(k&plus;1)&space;=&space;N_n(k)&space;&plus;&space;\sum_{n'}&space;P(n&space;|&space;n',&space;k)&space;N_{n'}(k)&space;-&space;\sum_{n'}&space;P(n'&space;|&space;n,k)&space;N_n(k)" title="N_n(k+1) = N_n(k) + \sum_{n'} P(n | n', k) N_{n'}(k) - \sum_{n'} P(n' | n,k) N_n(k)" /></a>

There are time-discrete vs time-continuous models. Assess what’s the time resolution needed for your system's performance metrics. Advantage of time-discrete models: a single common sampling rate can be defined among different modeling levels. The model parameters have incremental calibration. There are different methods for this : ad hoc experiments, system identification techniques, statistical verification techniques. Micro and macroscopic models have essentially two parameter types : state durations, state transition probabilities. Linear model has a probabilistic delay :

<a href="https://www.codecogs.com/eqnedit.php?latex=N_s(k&plus;1)&space;=&space;N_s(k)&space;-&space;p_a&space;N_s(k)&space;&plus;&space;p_s&space;N_a(k)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?N_s(k&plus;1)&space;=&space;N_s(k)&space;-&space;p_a&space;N_s(k)&space;&plus;&space;p_s&space;N_a(k)" title="N_s(k+1) = N_s(k) - p_a N_s(k) + p_s N_a(k)" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=N_a(k&plus;1)&space;=&space;N_0&space;-&space;N_s(k&plus;1)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?N_a(k&plus;1)&space;=&space;N_0&space;-&space;N_s(k&plus;1)" title="N_a(k+1) = N_0 - N_s(k+1)" /></a>

The steady state analysis would require that : N_n(k+1) = N_n(k) for all the states of the system n.

<a href="https://www.codecogs.com/eqnedit.php?latex=N^*_s&space;=&space;\frac{N_0}{1&plus;p_a&space;T_a}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?N^*_s&space;=&space;\frac{N_0}{1&plus;p_a&space;T_a}" title="N^*_s = \frac{N_0}{1+p_a T_a}" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=N^*_a&space;=&space;\frac{N_0&space;p_a&space;T_a}{1&plus;p_a&space;T_a}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?N^*_a&space;=&space;\frac{N_0&space;p_a&space;T_a}{1&plus;p_a&space;T_a}" title="N^*_a = \frac{N_0 p_a T_a}{1+p_a T_a}" /></a>

We consider the swarm performance metric. We have the following mean number of collaborations at iteration k :

<a href="https://www.codecogs.com/eqnedit.php?latex=C(k)&space;=&space;p_{g2}&space;N_s&space;(k&space;-&space;T_{ca})&space;N_g&space;(&space;k&space;-&space;T_{ca})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?C(k)&space;=&space;p_{g2}&space;N_s&space;(k&space;-&space;T_{ca})&space;N_g&space;(&space;k&space;-&space;T_{ca})" title="C(k) = p_{g2} N_s (k - T_{ca}) N_g ( k - T_{ca})" /></a>

We have the following mean collaboration rate over T_e :

<a href="https://www.codecogs.com/eqnedit.php?latex=C_t&space;(k)&space;=&space;\frac{&space;\sum_{k=0}^{T_e}&space;C(k)}{T_e}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?C_t&space;(k)&space;=&space;\frac{&space;\sum_{k=0}^{T_e}&space;C(k)}{T_e}" title="C_t (k) = \frac{ \sum_{k=0}^{T_e} C(k)}{T_e}" /></a>

We have the following reduced macroscopic model :

<a href="https://www.codecogs.com/eqnedit.php?latex=N_s(k&plus;1)&space;=&space;N_s(k)&space;-&space;p_{gl}[M_0&space;-&space;N_g(k)]&space;N_s(k)&space;&plus;&space;p_{g2}&space;N_g(k)N_s(k)&space;&plus;&space;p_{gl}&space;[M_0&space;-&space;N_g(k-T_g)]\Gamma(k;0)&space;N_s(k-T_g)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?N_s(k&plus;1)&space;=&space;N_s(k)&space;-&space;p_{gl}[M_0&space;-&space;N_g(k)]&space;N_s(k)&space;&plus;&space;p_{g2}&space;N_g(k)N_s(k)&space;&plus;&space;p_{gl}&space;[M_0&space;-&space;N_g(k-T_g)]\Gamma(k;0)&space;N_s(k-T_g)" title="N_s(k+1) = N_s(k) - p_{gl}[M_0 - N_g(k)] N_s(k) + p_{g2} N_g(k)N_s(k) + p_{gl} [M_0 - N_g(k-T_g)]\Gamma(k;0) N_s(k-T_g)" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=N_g(k&plus;1)&space;=&space;N_0&space;-&space;N_s(k&plus;1)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?N_g(k&plus;1)&space;=&space;N_0&space;-&space;N_s(k&plus;1)" title="N_g(k+1) = N_0 - N_s(k+1)" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=\Gamma&space;(k;0)&space;=&space;\prod_{j&space;=&space;k&space;-&space;T_g}^k&space;[1&space;-&space;p_{g2}&space;N_s(j)]" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\Gamma&space;(k;0)&space;=&space;\prod_{j&space;=&space;k&space;-&space;T_g}^k&space;[1&space;-&space;p_{g2}&space;N_s(j)]" title="\Gamma (k;0) = \prod_{j = k - T_g}^k [1 - p_{g2} N_s(j)]" /></a>

We have that T^{opt}_g can be computed analytically as follows :

<a href="https://www.codecogs.com/eqnedit.php?latex=T_g^{opt}&space;=&space;\frac{1}{ln(1&space;-&space;p_{g1}&space;R_g&space;\frac{N_0}{2})}&space;ln&space;\frac{1&space;-&space;\beta/2(1&plus;R_g)}{1-\beta/2}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?T_g^{opt}&space;=&space;\frac{1}{ln(1&space;-&space;p_{g1}&space;R_g&space;\frac{N_0}{2})}&space;ln&space;\frac{1&space;-&space;\beta/2(1&plus;R_g)}{1-\beta/2}" title="T_g^{opt} = \frac{1}{ln(1 - p_{g1} R_g \frac{N_0}{2})} ln \frac{1 - \beta/2(1+R_g)}{1-\beta/2}" /></a>

Where here \beta = N_0 / M_0 is the ratio of robots to sticks. 

**Week 7**

So today we learn Multi-Level Modeling: Calibration and Combination with Machine Learning. The submicroscopic model says that the libraries and APIs or middleware intrefaces such as ROS are usually provided in standard open-source simulators. State granularity arbitrary but (non spatial) performance metrics must be computable explicitly at all modeling levels. Exploit conservation laws (e.g. # of robots in an enclosed arena) to simplify the representation of the dynamical system. Measure all interaction times of interest in your system, i.e. those which might influence the system performance metrics. Geometric probabilities can be considered normalized detection areas (normalized over the total area of the experiment). Calculate the encountering rate ri [s-1] for the object i from the geometric probabilities gi.

<a href="https://www.codecogs.com/eqnedit.php?latex=r_i&space;=&space;\frac{v&space;W_s}{A_s}&space;g_i" target="_blank"><img src="https://latex.codecogs.com/gif.latex?r_i&space;=&space;\frac{v&space;W_s}{A_s}&space;g_i" title="r_i = \frac{v W_s}{A_s} g_i" /></a>

Where A_s is the detection area of the smallest object, v is the mean robot speed, W_s is the robot's detection width for the smallest object. . For time-discrete models, calculate the encountering probabilities pi (per time step) from the encountering rates:

<a href="https://www.codecogs.com/eqnedit.php?latex=p_i&space;=&space;r_iT" target="_blank"><img src="https://latex.codecogs.com/gif.latex?p_i&space;=&space;r_iT" title="p_i = r_iT" /></a>

Geometric probability g: example of transition in space from search to obstacle avoidance. The seed-assembling case study says that there is Reactive, non- communicating, nonadaptive behavior. State granularity choices say that you split the search state in search (loaded) and search(free) and keep dedicated avoidance/interferences states for each of the split states. The motivation is that the status loaded (carrying a seed) or free (not carrying a seed) can only change in a deterministic fashion (e.g., cannot be changed by an avoidance operation) and only by going to one of the seed dropping or picking states → can be explicitly represented with
a larger state space and deterministic transitions. Such enlarged state space facilitates the writing of the ODEs.

In the examples of assembled structures the noise in S&A and poor navigation
capabilities do not allow for precise, controllable structure building. The rationale for the combined methods is that. Any level of modeling (submicro, micro, or macro) allow us to consider certain parameters and leave others; models, as expression of reality abstraction, can be considered as more or less coarse “filters” of the reality.

We can analyze the entropy as follows. We have the simple entropy :

<a href="https://www.codecogs.com/eqnedit.php?latex=H(R)&space;=&space;-&space;\sum_{i=1}^m&space;p_i&space;log&space;p_i" target="_blank"><img src="https://latex.codecogs.com/gif.latex?H(R)&space;=&space;-&space;\sum_{i=1}^m&space;p_i&space;log&space;p_i" title="H(R) = - \sum_{i=1}^m p_i log p_i" /></a>

Where p_i is the proportion of the agents in cluster i. And m is the number of clusters in total. And h is the taxonomic level parameter. The social entropy is :

<a href="https://www.codecogs.com/eqnedit.php?latex=D(R)&space;=&space;\int_0^{\infty}&space;H(R,&space;h)&space;dh" target="_blank"><img src="https://latex.codecogs.com/gif.latex?D(R)&space;=&space;\int_0^{\infty}&space;H(R,&space;h)&space;dh" title="D(R) = \int_0^{\infty} H(R, h) dh" /></a>

The euclidean distance can be described as follows :

<a href="https://www.codecogs.com/eqnedit.php?latex=D_{eu}&space;=&space;\frac{1}{N(N-1)}&space;\sum_a&space;[\sum_{b&space;\neq&space;a}&space;d(a,b)]" target="_blank"><img src="https://latex.codecogs.com/gif.latex?D_{eu}&space;=&space;\frac{1}{N(N-1)}&space;\sum_a&space;[\sum_{b&space;\neq&space;a}&space;d(a,b)]" title="D_{eu} = \frac{1}{N(N-1)} \sum_a [\sum_{b \neq a} d(a,b)]" /></a>

Social entropy is looking for possible clustering of the vectors (looking for
possible castes) while Euclidian diversity is just looking how spread
out/diverse in general are the vectors.

**Week 8**

There are metaheuristic methods. There are design and optimization techniques
robust to noise, nonlinearities, discontinuities, non-convexity, multi-modal functions. There are different machine learning techniques. We have the classification axis 1 method. Here in supervised learning, training and test data are separated. In Unsupervised learning, no distinction between training and test data set; structure extraction from the data set. Reinforcement-based learning tries out possible design solutions, does mistakes, and tries to learn from its mistakes. The classification axis 2 says that there is in simulation techniques (reproduces the real scenario in simulation and applies there machine-learning techniques), hybrid, hardware-in-loop. There is a third classification axis : on-board, off-board (the machine-learning algorithm runs off-board and the system to be learned just serves as embodied
implementation of a candidate solution). The fourth classification axis says that : it can be population-based (population of candidate solutions is maintained by the algorithm, for instance Genetic Algorithms (individuals), Particle Swarm Optimization (particles), Particle Filter (particles), Ant Colony Optimization (ants/tours)), or it can be hill-climbing (the machine-learning algorithm work on a single candidate solution and try to improve on it).

There is particle swarm optimization, it is a competitive metaheuristic especially on
continuous optimization problems, it appears to deliver competitive results with
population sizes smaller than those of other metaheuristic methods. We have the following terminology related to particle swarm optimization.

- Swarm: pool of candidate solutions tested in one time step, consists of m
particles (typical # of particles: 20 to 50).
- Particle: represents a candidate solution; it is characterized by a velocity vector
v and a position vector x in the hyperspace of dimension D.
- Neighborhood: set of particles with which a given particle share performance
information
- Iteration: at the end of an iteration, a new pool of candidate solutions after the
metaheuristic operators have been applied is available (typical # of iterations:
50, 100, 1000).
- Fitness function: measurement of efficacy of a given candidate solution during
the evaluation span.
- Evaluation span: evaluation period of each candidate solution during a single
time step (algorithm iteration); the evaluation span might take more or less time
depending on the experimental scenario.
- Life span: number of iterations a candidate solution is present in the pool.
- Swarm manager: update velocities and position for each particle according to
the main PSO loop

The main PSO loop says that :

```
At each iteration t
  At each particle i
    For each component j
      Update the velocity as follows
```
<a href="https://www.codecogs.com/eqnedit.php?latex=v_{ij}(t&plus;1)&space;=&space;w&space;v_{ij}(t)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?v_{ij}(t&plus;1)&space;=&space;w&space;v_{ij}(t)" title="v_{ij}(t+1) = w v_{ij}(t)" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=c_p&space;rand()(x_{ij}^*&space;-&space;x_{ij})&space;&plus;&space;c_n&space;rand()&space;(x_{i'j}^*&space;-&space;x_{ij})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?c_p&space;rand()(x_{ij}^*&space;-&space;x_{ij})&space;&plus;&space;c_n&space;rand()&space;(x_{i'j}^*&space;-&space;x_{ij})" title="c_p rand()(x_{ij}^* - x_{ij}) + c_n rand() (x_{i'j}^* - x_{ij})" /></a>
```
      The move to :
```
<a href="https://www.codecogs.com/eqnedit.php?latex=x_{ij}(t&plus;1)&space;=&space;x_{ij}(t)&space;&plus;&space;v_{ij}(t&plus;1)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?x_{ij}(t&plus;1)&space;=&space;x_{ij}(t)&space;&plus;&space;v_{ij}(t&plus;1)" title="x_{ij}(t+1) = x_{ij}(t) + v_{ij}(t+1)" /></a>

There are different neighborhood types : local (only k neighbors considered over m particles in the population) and global. There are also genetic algorithms, it is designed for discrete optimization problems. The inspiration is from natural evolution, where organisms adapt to their environments – better able to survive over time. There are three aspects of the evolution : Survival of the fittest, Genetic combination in reproduction, Mutation. The generation loop says that : the crossover mutation leads to population replenishment which leads to the decoding of the genotype to the phenotype, which leads to fitness measurements and encoding, which leads to the selection and again to the crossover mutation. GA still much more widely used in general research community and robust to continuous and discrete optimization problems. We now define performance, in obstacle avoidance.

<a href="https://www.codecogs.com/eqnedit.php?latex=\phi&space;=&space;V(1-\sqrt{\Delta&space;V})(1-i)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\phi&space;=&space;V(1-\sqrt{\Delta&space;V})(1-i)" title="\phi = V(1-\sqrt{\Delta V})(1-i)" /></a>

Where V is the main speed of the wheels, delta V is the absolute algebraic difference between the wheel speeds, i is the activation value of the sensor with the
highest activity. Controller architecture can be of any type but worth using
GA/PSO if the number of parameters to be tuned is important. Suppose you want to evolve homing behavior. Then the fitness function is given by :

<a href="https://www.codecogs.com/eqnedit.php?latex=\phi&space;=&space;V(1-i)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\phi&space;=&space;V(1-i)" title="\phi = V(1-i)" /></a>

Faithful realistic simulators enable to explore design solution which encompasses co-evolution (co-design) of control and morphological characteristics (body shape, number of sensors, placement of sensors, etc. ).

**Week 9**

Two fundamental reasons making robot control optimization expensive in terms of time: Time for evaluation of candidate solutions, Noisy performance evaluations disrupt the
optimization process and require multiple evaluations for assessing the actual performance of a candidate solution. Suppose you want to reduce the evaluation time. You can exploit more abstracted, calibrated representations. Noise comes from sensors,
actuators, initial conditions, other robots. Noise causes decreased convergence speed and residual error. In particular in a population-based optimization method (but similar
mechanisms are applicable to hill-climbing evaluative learning), we want to dedicate more computational time to evaluate promising solutions and eliminate as quickly as possible the “lucky” ones.  Use dedicated functions for aggregating multiple evaluations: e.g., minimum and average or more generalized aggregation functions
(e.g., quasi-linear weighted means), perhaps combined with a statistical test for comparing resulting aggregated performances. The Bernoulli noise model says that :

<a href="https://www.codecogs.com/eqnedit.php?latex=f_i^b&space;(x)&space;=&space;\frac{f_i(x)}{\max&space;f_i}&space;&plus;&space;A&space;\cdot&space;B(p)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?f_i^b&space;(x)&space;=&space;\frac{f_i(x)}{\max&space;f_i}&space;&plus;&space;A&space;\cdot&space;B(p)" title="f_i^b (x) = \frac{f_i(x)}{\max f_i} + A \cdot B(p)" /></a>

Extended-time optimization requires a fair test: same total evaluation time
for all the algorithms. Hibrid optimization moves from high-fidelity simulation (Webots) to real robots after 90% of the total number of iterations. Noise-resistance helps manage transition. There are three axes for co-optimization : performance evaluation (individual vs group), solution sharing (private vs republic), team diversity (homogeneous vs heterogeneous). Heterogeneity allowed but eventually roughly homogeneous solution via shuffle around of candidate solutions. The distributed optimization has a new approach: distributed also the population manager on the robots (on-board) and share candidate solutions within the neighborhood through communication channels.

Increasing Number of Robots: Impact of Noise Resistance. Performance decreases with
number of robots (more difficult to avoid in overcrowded arenas). Noise-resistance make the difference in high density (i.e. noisier) scenarios. Impact of limited-time optimization : 1. Evaluation span include at least 1 interaction, 2. Swarm size = dimension of parameter space, 3. Use noise-resistant algorithms, 4. Dedicate max time budget to iterations. Consider the following PSO pbest algorithm :

```
Initialize particles
For N_i iterations do
  For N_p particles do
    Update particle position
    Evaluate particle
    Re-evaluate personal best
    Aggregate previous best
    Share personal best
  End for
End for
```

Consider the third approach called OCBA :

<a href="https://www.codecogs.com/eqnedit.php?latex=\frac{N_i}{N_j}&space;=&space;(\frac{\sigma_i&space;/&space;\delta_{b,i}}{\sigma_j&space;/&space;\delta_{b,j}}),&space;i&space;\neq&space;j&space;\neq&space;b" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{N_i}{N_j}&space;=&space;(\frac{\sigma_i&space;/&space;\delta_{b,i}}{\sigma_j&space;/&space;\delta_{b,j}}),&space;i&space;\neq&space;j&space;\neq&space;b" title="\frac{N_i}{N_j} = (\frac{\sigma_i / \delta_{b,i}}{\sigma_j / \delta_{b,j}}), i \neq j \neq b" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=N_b&space;=&space;\sigma_b&space;\sqrt{\sum_{i=1,&space;i&space;\neq&space;b}^k&space;\frac{N_i^2}{\sigma_i^2}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?N_b&space;=&space;\sigma_b&space;\sqrt{\sum_{i=1,&space;i&space;\neq&space;b}^k&space;\frac{N_i^2}{\sigma_i^2}}" title="N_b = \sigma_b \sqrt{\sum_{i=1, i \neq b}^k \frac{N_i^2}{\sigma_i^2}}" /></a>

Proven that maximizes probability of correct selection of best candidate b for infinite total number of samples (evaluations) T. Consider the following Centralized PSO OCBA.
 
```
Initialize particles
For N_i iterations do
  For N_p particles do
    Evaluate new particle position n_0 times
  End for
  remaining budget := iteration budget - n_0 N_p
  while remaining budget > 0 do
    allocate delta samples among current positions and personal bests using OCBA
    evaluate allocated samples
    recalculate mean and variance for new evaluations
    remaining budget := remaining budget - delta
  end while
  For N_p particles do
    Update personal best
    Update neighborhood best
    Update particle position
  End for
End for
```

**Week 10**

The division of labor and control.Talking about division of labor means that there is a potential redundancy in the role/contribution of individuals and coordination in
role/task allocation helps in decreasing redundancy and increasing efficiency as a group. We have three castes : physical, temporal (temporal polyethism) and behavioral. Temporal polyethism says that young individuals work on internal tasks (brood care
and nest maintenance). Older individuals forage for food and defend the nest. The flexibility of social roles : the number of individuals performing different tasks and the nature of the tasks to be done are subject to constant change in the course of the life of a colony. A threshold-based behavioral response appears to be a biologically
plausible mechanism for achieving the flexibility of social roles. The lower the threshold, the lower can be the stimulus for achieving a given response; respectively, the lower the threshold, the higher will be the response of an individual for a given stimulus. The ratio of the majors in a colony artificially increased. Majors replace missing minors confirming the idea of the response threshold. Use a model to explain the wilson's results. Here you have a nonspatial model, i.e. equiprobable exposure of individuals to the stimuli associated with the task and stimulus homogeneously distributed over space. Below we have the dynamics of the fraction of active individuals in each caste:

<a href="https://www.codecogs.com/eqnedit.php?latex=\frac{dx}{dt}&space;=&space;\frac{s^2}{s^2&space;&plus;&space;\theta_1^2}(1-x_1)&space;-&space;r_a&space;x_1" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{dx}{dt}&space;=&space;\frac{s^2}{s^2&space;&plus;&space;\theta_1^2}(1-x_1)&space;-&space;r_a&space;x_1" title="\frac{dx}{dt} = \frac{s^2}{s^2 + \theta_1^2}(1-x_1) - r_a x_1" /></a>

Here s is the intensity of the stimuli associated with the task. Then \theta_i is the  response threshold of caste i to the task. r_a is the rate of task abandoning (i.e. probability per infinitesimal time interval that an active individual abandons the task on which it is engaged for moving to idle). Then we have the dynamics of demand associated with the task :

<a href="https://www.codecogs.com/eqnedit.php?latex=\frac{ds}{dt}&space;=&space;\delta&space;-&space;\frac{\alpha}{N}(n_1&space;&plus;&space;n_2)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{ds}{dt}&space;=&space;\delta&space;-&space;\frac{\alpha}{N}(n_1&space;&plus;&space;n_2)" title="\frac{ds}{dt} = \delta - \frac{\alpha}{N}(n_1 + n_2)" /></a>

Where \delta is the rate of the stimulus increase, and \alpha is the normalized effectiveness rate for the individual contribution on the task (in this case an active individual of type 1 contribute in the same way as an active individual of type 2 when performing the task). Polists are primitive eusocial species. These species do not show morphological differences between castes in the adult stages, nor any control or physiological determination of the role an individual will play in the colony as an adult. The integration and coordination of individual activities is achieved through
the interactions which occur between the members of the colony, and between
the members of the colony and the local environment. The lower the threshold, the lower can be the stimulus for achieving a given response; respectively, the lower the threshold, the higher will be the response of an individual for a given stimulus.

Individual behavioral algorithm : 

<a href="https://www.codecogs.com/eqnedit.php?latex=\theta_i&space;\rightarrow&space;\theta_i&space;-&space;\epsilon" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\theta_i&space;\rightarrow&space;\theta_i&space;-&space;\epsilon" title="\theta_i \rightarrow \theta_i - \epsilon" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=\theta_i&space;\rightarrow&space;\theta_i&space;&plus;&space;\phi" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\theta_i&space;\rightarrow&space;\theta_i&space;&plus;&space;\phi" title="\theta_i \rightarrow \theta_i + \phi" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=T_{\theta_i}(s)&space;=&space;\frac{s^2}{s^2&space;&plus;&space;\theta_i^2}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?T_{\theta_i}(s)&space;=&space;\frac{s^2}{s^2&space;&plus;&space;\theta_i^2}" title="T_{\theta_i}(s) = \frac{s^2}{s^2 + \theta_i^2}" /></a>

Where s is the intensity of stimuli associated with the task, \theta_i is the response threshold of individual i to the task, \epsilon is the incremental learning parameter, \phi is the incremental forgetting parameter. Example of dynamics of the overall stimulus (demand) associated with a task.

<a href="https://www.codecogs.com/eqnedit.php?latex=\frac{ds}{dt}&space;=&space;\delta&space;-&space;\frac{\alpha}{N}(\sum_{i=1}^m&space;n_i)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{ds}{dt}&space;=&space;\delta&space;-&space;\frac{\alpha}{N}(\sum_{i=1}^m&space;n_i)" title="\frac{ds}{dt} = \delta - \frac{\alpha}{N}(\sum_{i=1}^m n_i)" /></a>

Where \delta is the rate of the stimulus increase, \alpha is the normalized effectiveness rate for the individual contribution to the task (in this case all the active individual belonging to different castes contribute in the same way), n_i is the number of active individual belonging to caste i. Special type of aggregation: linear structure assembly; only seeds at the tips of a cluster can be manipulated. Evolution of manipulation sites: number of manipulation sites decreases with progress of aggregation → more competition (interference) among robots for the same manipulation sites. End criterion: a power-efficient building system should stop working when the task is accomplished. Increasing the final cluster size: at the end all the seeds should belong to the single cluster (only those on the ground count for the aggregation metrics). Threshold based algorithm : there are key choices for minimizing complexity and maximizing interchangeabilit. Robots have the same capabilities, no reason to have different threshold, probabilistic response even with a single threshold will suffice to regulate the activity, not all the robots stop at the same time, when one drop the work, direct influence on aggregation demand, how do we implement: deterministic response + noise = probabilistic response → exploit local perception based on on-board sensors as noise generator. Chosen stimulus: time needed to find a seed to manipulate; the larger the time, the lower the stimulus associated with the aggregation demand. Robots model the market economy : Resources, tasks have precise worth, They are traded over the market, Accomplish task can lead to paid revenue, Consume resources incurs cost, the goal is to maximize profit (i.e. revenue - cost).

The task allocation problem and solution depend heavily on the task description. There is threshold based and market based approaches for multi-robot coordination. The metrics consider the cost of the production allocation (communication frequency, computation) and the quality of the allocation (number of events handles, total distance travelled). The market-based approach tries to assign events to the nearest robot. The threshold-based approach is such that the robot handles closest events w/o duplicating work. For this there are different axes : the communication range, the accuracy in state estimation, the accuracy in task estimation, the range in task estimation.

**Week 11**

Wireless sensor networks will enable spatially and temporally dense environmental monitoring, will enable precise, realtime alarm triggering. The sensorscope for example measures the temperature, the humidity, the precipitation, the solar radiation, the soil moisture. The sensorscopes are connected to a general packet radio service. But the closest server access point may be quite far from the stations. The Friis law says that:

<a href="https://www.codecogs.com/eqnedit.php?latex=\frac{P_t}{P_r}&space;\propto&space;(\frac{4&space;\pi&space;d}{\lambda})^2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{P_t}{P_r}&space;\propto&space;(\frac{4&space;\pi&space;d}{\lambda})^2" title="\frac{P_t}{P_r} \propto (\frac{4 \pi d}{\lambda})^2" /></a>

The pros of multi-hop WSNs are that : only one car battery in the network, extended spatial coverage of the network, multiple routes for stations to communicate with the sink, auto configurable network (robustness). Multi-hop WSNs implementation is : neighborhood discovery followed by data routing, followed by time synchronization, followed by duty-cycling. The time synchronization says that weather conditions, especially temperature and humidity, may have a significant effect on hardware. Time synchronization, says that nodes need to know the time to timestamp packets, synchronize actions. How do we get time, fully decentralized (every node gets the time itself), partially centralized (time is propagated from reference nodes).

Energy saving is a crucial driver for the design of WSN. Sensing data are typically only collected for a particular application and rarely used to control node actions: WSN are
typically data-agnostic! The modified Friis law says that :

<a href="https://www.codecogs.com/eqnedit.php?latex=\frac{P_t}{P_r}&space;=&space;G_t&space;G_r&space;(\frac{\lambda}{4&space;\pi})^2&space;(\frac{1}{R})^n" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{P_t}{P_r}&space;=&space;G_t&space;G_r&space;(\frac{\lambda}{4&space;\pi})^2&space;(\frac{1}{R})^n" title="\frac{P_t}{P_r} = G_t G_r (\frac{\lambda}{4 \pi})^2 (\frac{1}{R})^n" /></a>

Low power listening protocols are key for saving power. Is there a way to make a stronger overlap between WSN and DIS, in particular more closed-loop control at the node level, more distributed control at the network level? The quadtree is a hierarchical topology, we have bottom-up measurements and top-down control. Consider the state machine, it says that there is a layer increment: if all child nodes are processed, an idle node: if pruned by clusterhead, and Data: clusterhead replaces pruned children. Now we study the spatiotemporal suppression of data reporting. We save communication energy via efficient reporting. Temporal suppression is when you ask yourself : has my value changed recently ? Spatial suppression is when you ask : are my neighbors reporting similar measurements ? The suppression-based algorithm says there is constraint chaining.

**Week 12**
We study the motivation for dense sensing and the mobility in environmental monitoring. Some examples of mobile sensor networks : pipeprobe (a mobile sensor droplet for mapping the hidden pipeline). 

**Week 13**
