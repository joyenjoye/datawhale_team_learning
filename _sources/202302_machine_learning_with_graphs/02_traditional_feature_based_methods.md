# Task 2: Traditional Feature-based Methods

Using effective features over graphs is the key to achieving good test performance. Traditional Machine Learning Pipeline manually designs features for nodes/links/graphs. 

This posts will cover traditional features for node/link/graph level prediction for undirected graphs.

## Node Level Tasks and Features

Node classification is a supervised machine learning (ML) approach whereby existing nodes with known classes can be used to train a model that will learn the classes for nodes where they are unknown[\[3\]](https://neo4j.com/developer/graph-data-science/node-classification/).

Characterize the structure and position of a node in the network

### Node Degree

the degree $k_v$ of node $v$ is the number of edges (neighboring nodes) the node has. Treats all neighboring nodes equally

### Node centrality

Node centrality $C_v$ takes the node importance in a graph into account. There are different ways to model importance.

**Engienvector centrality** model the centrality of node $v$ as the sum of the centrality of neighboring nodes:

$$c_v = \frac{1}{\lambda}\sum_{u \in N(v)}{c_u}$$

Where $\lambda$ is normalization constant. It will turn out to be the largest eigenvalue of A.

Rewrite the recursive equation in the matrix form.

$$\lambda c = A\mathbb{c}$$

Where A is adjacent matrix, $A_uv=1$ if $u \in N(v)$. $c$ is centrality vector and $\lambda$ is eigenvalue.

By Perron-Frobenius Theorem, The largest eigenvalue $\lambda_{max}$ is always positive and unique. And The eigenvector $c_{max}$  corresponding to $\lambda_{max}$ is used for centrality.


The intution behind **Betweenness centrality** is that A node is important if it lies on many shortest
paths between other nodes. Mathematically, It is defined as follows:

$$c_v = \sum_{s\not =v\not ={t}}\frac{n_1}{n_2}$$

Where $n_1$ is the number of shorted paths between $s$ and $t$ that contains $v$. and $n_2$ is the number of shorted paths between $s$ and $t$.

Closeness centrality.






## Link Prediction Task and Features

Link Prediction is to predict new links based on the existing links. At test time, node pairs (with no existing links) are ranked, and top $K$ node pairs are predicted. The key is to design features for a pair of nodes.

Two formulations of the link prediction task:
- Links missing at random: Remove a random set of links and then aim to predict them. 
- Links over time: given a graph $G[t_{0},t_{0}^{'}]$ defined by edges up to time $t_{0}^{'}$ output a ranked List L of edges that not in $G[t_{0},t_{0}^{'}]$ that are predicted to appear in time $G[t_{1},t_{1}^{'}]$. To evaluate, count how many edges in $L$ are real new edge that appear during the test period.

To do link prediction, it takes the following steps:
1. For each pair of nodes $(x,y)$ compute score $c(x,y)$ For example, $c(x,y)$ could be the # of common neighbors of $x$ and $y$
2. Sort pairs $(x,y)$ by the decreasing score
$c(x,y)$
3. Predict top $n$ pairs as new links 
4. See which of these links actually appear in $G[t_{1},t_{1}^{'}]$

The question now become what is the formulation of $c$
- Distance based: shortest-path distance between 2 nodes
- Local Neighborhood overlap: number of neighboring nodes shared between two nodes $v_1$ and $v_2$. 
  - Common Neighters
  - Jaccard's Coefficient
  - Adamic-Adar Index
  
- Global Neighborhood overlap
  - Katz index: count the number of walks of all lengths between a given pair of nodes.


## Graph


## References

\[1\][Stanford CS224W: Machine Learning with Graphs | 2021](https://www.youtube.com/watch?v=P-m1Qv6-8cI&list=PLoROMvodv4rPLKxIpqhjhPgdQy7imNkDn&index=1)

\[2\][传统图机器学习的特征工程-节点【斯坦福CS224W】](https://www.bilibili.com/video/BV1HK411175s/?spm_id_from=333.788&vd_source=fccde7883fbf93ac15b03dee298f9f18)

\[3\][传统图机器学习的特征工程-连接【斯坦福CS224W】](https://www.bilibili.com/video/BV1r3411m7sD/?spm_id_from=333.788)

\[4\][传统图机器学习的特征工程-全图【斯坦福CS224W】](https://www.bilibili.com/video/BV1r3411m7sD/?spm_id_from=333.788)


