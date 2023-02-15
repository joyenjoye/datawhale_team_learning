# Task 1: Introduction

## Why Graphs?

Graphs are a general language for describing and analyzing entities with relations/Interactions. Many types of data can be represented as graphs. Here are examples:
- Computer Networks
- Disease Pathways
- Food Webs
- Particle Networks
- Underground Networks
- Event Graphs
- Social Networks
- Economic Networks
- Communication Networks
- Citation Networks
- Internet
- Network for Neurons
- Knowledge Graphs: 
- Regulatory Networks
- Scene Graphs 
- Code Graphs
- Molecules
- 3D Shapes

Modeling these graphical relations allows us to capture the relationships between different objects or entities. Through which, we can understand the underlying phenomena in different domains. 

## How to model relational structure in graphs?

Complex domains have a rich relational structure, which can be represented as relational graph. By explicitly modeling relationships we achieve better performance. 

Traditional Machine Learning spend a lot of time on feature engineering to capture the structure of the data so that machine learning models can take advantage of. 

Alternatively, deep neural networks can be used to automatically learn a good representation of the graph. The learned representation can be used for down stream machine learning algorithm. 

The problem is that modern Deep learning tool box is designed for simple sequences(Text) & grids(Images). But graphs are complex because of the following characteristics:
- Arbitrary size and complex topological structure
- No fixed ordering or reference points
- often dynamic and have multi-modal features
  
How can we develop neural networks that are much more broadly applicable? so that we can use for graphs. 

The courses covers various topic in Machine Learning and representation Learning for Graph Structured data:
- Traditional methods: Graphlets, Graph kernels
- Node Embeddings: DeepWalk, Node2Vec
 - Graph Neural Networks: GCN, GraphSAGE, Graph attention Network, Theory of GNNs.
- Knowledge Graphs and reasoning: TransE, BetaE
- Deep Generative Model for Graphs: Graph RNN.
- Applications in Biomedicine, Science, and fraud detection, recommender system. 

## Graph ML Tools
- Graph Neural Networks: [PyG(PyTorch Geometric)](http://www.pyg.org/), GraphGym, [DGL](https://www.dgl.ai/)
- Network Analytics: SNAP.PY, [NetworkX](networkx.org)
- Graph Visualizations: [Echarts](echarts.apache.org/examples/zh/index.html#chart-type-graphGL), [AntV](graphin.antv.vision)
- Graph Databases: Neo4j, Microsoft Azure Cosmos DB

## Applications of Graph ML 

50+ graph algorithms are implemented in Neo4j[3]. There are different types of tasks:
- Node Classification: predict a property of a node. e.g. Categorize online users or items
- Link Prediction: predict whether there are missing links between nodes. e.g. Knowledge graph completion
- Graph classification: categorize different graphs. Molecule property prediction
- Clustering: detect if nodes form a community. e.g. Social Circle detection.
- Graph Generation: generate a new graph. e.g. Drug discovery
- Graph Evolution: Physical Simulation. 

### Node Classification: Protein Folding

Every protein is made up a sequence of amino acids bonded together.  There amino acids interact locally to form shapes like helices and sheets. These shapes fold up on a larger scales to form the full three-dimensional protein structure. Proteins can interact with other proteins perform functions such as signaling and transcribing DNA. 

[AlphaFold](https://www.deepmind.com/blog/putting-the-power-of-alphafold-into-the-worlds-hands) is an AI system developed by DeepMind that predicts a protein’s 3D structure from its amino acid sequence. 

The graph can be constructed:
- nodes: amino acids in a protein sequence 
- Edges: Proximity between amino acids (residues)



### Link Prediction: Recommender Systems

In the scenario of watching movies, buying products, and listening to music, we want to recommend items users might like. 

The graph can be constructed:
- Nodes: Users and items
- Edges: User-item interactions

Predict whether two nodes in a graph are related.


### Graph Link Prediction: Drug Side Effects

Many patients take multiple drugs to treat complex or co-existing disease.

- 46% people ages 70-79 take more 5 drugs 
- Many patients take more than 20 drugs to treat heart disease, depression, insomnia, etc. 

Task: predict adverse side effect for a given pair of drugs. e.g. How likely will Simvastatin and Ciprofloxacin when taken together, break down muscle tissue?

The graph can be constructed:
- Nodes: Drugs & Proteins
- Edges: Interactions 

The task is essentially predict missing edge/link.

### Subgraph Prediction: Traffic Prediction

From a starting point/location to a destination, we want to predict how long it takes to arrive the destination.

The graph can be constructed:
- Nodes: Road segments
- Edges: Connectivity between road segments


### Graph Classification: Drug Discovery

In the scenario of predicting promising molecules from a pool of candidates. The predicted promising molecules can be further tested in the lab. This helps with discover drugs quickly and more efficiently. 

Antibiotics are small molecular graphs
  - Nodes: Atoms
  - Edges: Chemical Bonds

The task is essentially graph classification.

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

### Constructing Graphs

Graphs can be undirected or directed. Examples of undirected graphs are collaborations, friendship on facebook. Examples of directed graphs are phone calls, financial transactions and following on Twitter.

Node degrees. For undirected, node degree is the number of edges adjacent to node $i$ For directed graphs, we can further define in-degree and out-degree. The degree of a node is sum of in and out degrees. 

Bipartite is a type of graph structure whose nodes can be divided into two disjoint set U and V such that every links connect a node in U to one in V.  Examples of Bipartite graphs are Authors-to-Papers(they are authored), Actors-to-Movies(they are appeared in), Users-to-Movies(they rated), Recipes-to-Ingredients(they contain). Bipartite can be projected as Folded networks. For example Authors-to-Papers(they are authored) can be projected as author collaboration networks. 

## References
\[1\][Stanford CS224W: Machine Learning with Graphs | 2021](https://www.youtube.com/watch?v=P-m1Qv6-8cI&list=PLoROMvodv4rPLKxIpqhjhPgdQy7imNkDn&index=1)

\[2\][斯坦福CS224W图机器学习、图神经网络、知识图谱【同济子豪兄】](https://www.bilibili.com/video/BV1pR4y1S7GA/?spm_id_from=333.788&vd_source=fccde7883fbf93ac15b03dee298f9f18)

\[3\][Graph Algorithms, Neo4j.](https://neo4j.com/docs/graph-data-science/current/algorithms/)