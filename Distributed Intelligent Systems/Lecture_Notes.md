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

We have the following AntNet algorithm : 
