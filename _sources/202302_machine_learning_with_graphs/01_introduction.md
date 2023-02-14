# Task 1: Introduction

Graphs are a general language for describing and analyzing entities with relations/Interactions


## Types of Data are Graphs

Many types of data can be represented as graphs and modeling these graphical relations allows to capture the relationships between different objects or entities. Through which, we can understand the underlying phenomena in different domains. 

- Social Networks
- Economic Networks
- Communication Networks
- Citation Networks
- Internet
- Network for Neurons
- Knowledge Graphs: describe knowledge 
- Regulatory Networks: describe the regulatory mechanisms as processes governed by the connections between different entities.
- Scene Graphs 
- Code Graphs: describe the software as a graph of calls among functions 
- Molecules
- 3D Shapes

How do we take advantage of relational structure for better prediction?
Complex domains have a rich relational structure, which cna be represented as relational graph. By explicitly modeling relationships we achieve better performance. 

Modern Deep learning tool box is designed for simple sequences(Text) & grids(Imges). Networks are complex with the following characteristics:
- Arbitrary size and complex topological structure
- No fixed ordering or reference points
- often dynamic and have multi-modal features
  
How can we develop neural networks that are much more broadly applicable? so that we can use for graphs. 

- Input: Networks
- Output/Predictions: Node Labels, New Links, Generated graphs and subgraphs

Traditional Machine Learning spend a lot of time on feature engineering to capture the structure of the data so that machine learning models can take advantage of. 

In this case, it will mostly about representation learning where feature engineer is removed. But instead, we can automatically learn a good representation so that it can be used for down stream machine learning algorithm. 

Representation Learning is to map nodes to d-dimensional embeddings such that similar nodes in the network are embedded close together in the embedding space. 

The courses covers various topic in Machine Learning and representation Learning for Graph Structured data:
- Traditional methods: Graphlets, Graph kernels
- Node Embeddings: DeepWalk, Node2Vec
- Graph Neural Networks: GCN, GraphSAGE, GAT, Theory of GNNs.
- Knowledge Graphs and reasoning: TransE, BetaE
- Deep Generative Model for Graphs
- Applications in Biomedicine, Science and fraud detection, recommender system. 

## Graph Algorithms

- Pathfinding & Search
- Centrality / Importance
- Community Detection
- Link prediction
- Similarity
- Embeddings

## Applications of Graph ML 

- Node Classification: predict a property of a node. e.g. Categorize online users or items
- Link Prediction: predict whether there are missing links between nodes. e.g. Knowledge graph completion
- Graph classification: categorize different graphs. Molecule property prediction
- Clustering: detect if nodes form a community. e.g. Social Circle detection.
- Graph Generation: Drug discovery
- Graph Evolution: Physical Simulation. 

### Node level Example: Protein Folding

Every protein is made up a sequence of amino acids bonded together.  There amino acids interact locally 

- nodes: amino acids in a protein sequence 
- Edges: 


### Link level Example: Recommender Systems
Predict whether two nodes in a graph are related.

Users interacts with items
- Watch moives 

### Link level Example: Drug Side Effects

Many patients take multiple drugs to treat complex or co-existing disease.

- 46% people ages 70-79 take more 5 drugs 
- Many patients take more than 20 drugs

Task: predict adverse side effect for a given pair of drugs. e.g. How likely will Simvastatin and Ciprofloxacin when taken together , break down muscle tissue?

The graph can be constructed:
- Nodes: Drugs & Proteins
- Edges: Interactions 

The task is essentially predict missing edge/link.


### Subgraph level Example: Traffic Prediction

### Graph Classification: Drug Discovery

Antibiotics are small molecular graphs
  - Nodes: Atoms
  - Edges: Chemical Bonds

Task: predict promising molecules from a pool of candidates. 

The predicted promising molecules can be further tested in the lab. This helps with discover drugs quickly and more efficiently. 


### Graph Generation: Generating Novel Molecules

Graph generation can be used for the following two use cases:
- Generate new molecules as graphs in a targeted way. For example, generate novel molecules with high drug likeness.
- Optimize existing molecules to have desirable properties.

### Graph Evolution: Physical Simulation

Physical simulation as a graph
- Nodes: Particles
- Edges: Interaction between particles

The simulation repeats the following process
1. construct a graph with nodes being particles and edge being proximity between nodes
2. predict nodes' position and velocities based on current position and velocities
3. move the particles to new positions 


## Choice of Graph representation

Network or graphs $G(N,E)$ consists of 
- Objects: nodes, vertices. Denoted as $N$
- Interactions: links, edges. Denoted as  $E$

We may have different options for constructing networks. Choice of the proper network representation of a given domain determines our ability to use network successfully. 

### Options when constructing graphs

Graphs can be undirected or directed. Examples of undirected graphs are collaborations, friendship on facebook. Examples of directed graphs are phone calls, financial transactions and following on Twitter.

Node degrees. For undirected, node degree is the number of edges adjacent to node $i$ For directed graphs, we can further define in-degree and out-degree. The degree of a node is sum of in and out degrees. 

Bipartite is a type of graph structure whose nodes can be divided into two disjoint set U and V such that every links connect a node in U to one in V.  Examples of Bipartite graphs are Authors-to-Papers(they are authored), Actors-to-Movies(they are appeared in), Users-to-Movies(they rated), Recipes-to-Ingredients(they contain). Bipartite can be projected as Folded networks. For example Authors-to-Papers(they are authored) can be projected as author collaboration networks. 
