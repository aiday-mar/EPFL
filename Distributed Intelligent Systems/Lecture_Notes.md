# Distributed Intelligent Systems

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

Here we have that alpha is the parameter controlling the includence of the virtual pheromone, whilst beta is the parameter controlling the influence of the local visibility. 
