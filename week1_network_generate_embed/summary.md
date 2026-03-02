# Network Embedding — Session Summary

---

## Part 1: Network Generation Models

The core idea across all 3 models is the same:

> Define a **probability matrix P** where each cell $P_{ij}$ = probability of an edge existing between node $i$ and node $j$. Then randomly sample edges based on those probabilities to generate the actual network.

The models differ in **how they construct P**.

---

### Stochastic Block Model (SBM)

**Key idea:** Nodes are divided into discrete communities. The probability of connection depends only on which communities the two nodes belong to.

$$P_{ij} = B_{\tau_i \tau_j}$$

- $B$ is a small $K \times K$ matrix of community-to-community connection probabilities
- Every node within the same community has **identical** connection probability to every other node
- Results in a clean, uniform block structure in the heatmap

**Visually:** Perfect flat blocks on the diagonal — textbook clean community structure.

---

### Degree-Corrected SBM (DCSBM)

**Key idea:** Same as SBM, but each node also gets its own "popularity" parameter $\theta_i$. Some nodes are hubs (high $\theta$), some are loners (low $\theta$).

$$P_{ij} = \theta_i \cdot \theta_j \cdot B_{\tau_i \tau_j}$$

- Community structure still exists, but within each community nodes have **varying degrees**
- $\theta$ is randomly sampled per node (e.g. from uniform distribution)
- More realistic than basic SBM — real networks always have degree heterogeneity

**Visually:** Blocks still visible but with bright rows/columns (hubs) and sparse rows/columns (loners). Loner nodes float away as outliers in the network graph.

---

### Random Dot Product Graph (RDPG)

**Key idea:** Instead of discrete community labels, each node gets a continuous **latent position vector** in some low-dimensional space. The probability of connection = dot product of their vectors.

$$P_{ij} = x_i \cdot x_j = x_i^T x_j$$

- No hard community boundaries — communities are regions in a continuous latent space
- Nodes in the same community get vectors sampled near the same base vector (with some noise)
- The full probability matrix is just $P = XX^T$ where $X$ is the matrix of all latent vectors
- Most flexible and general of the three models

**Visually:** Smoothest probability matrix — probabilities gradually vary rather than jumping between discrete block values. Communities may overlap if latent vectors are close in space.

---

### How the 3 Models Relate

$$\text{SBM} \subset \text{DCSBM} \subset \text{RDPG}$$

SBM is a special case of DCSBM (all $\theta = 1$). Both are special cases of RDPG (discrete community vectors are just a rigid version of continuous latent positions).

---

## Part 2: Embedding Algorithms

Once you have a network, embedding converts each node into a **vector** so you can run standard ML on it. The goal: nodes that are similar in the network should end up close together in vector space.

---

### Adjacency Spectral Embedding (ASE)

**What it does:** Apply SVD to the adjacency matrix $A$, keep only the top $d$ components, and use those as node vectors.

$$\hat{X} = U_d \Sigma_d^{1/2}$$

- Math-based, directly connected to RDPG — ASE is literally the method for estimating RDPG latent positions from observed data
- The top $d$ singular values capture the signal (true structure), smaller ones are noise — so keeping only top $d$ is a denoising operation
- Works best when the network has genuine low-rank latent structure (like RDPG)

---

### Node2Vec (DeepWalk family)

**What it does:** Generate random walks on the network (treat them like sentences), then train a Word2Vec neural network to predict which nodes appear near each other in those walks. The learned weights become the node vectors.

- Inspired by NLP — stolen directly from Word2Vec
- Nodes that tend to co-occur in random walks get similar vectors
- Works best when community structure is sharp and discrete (like SBM/DCSBM) because walks stay within communities naturally
- Key parameters: walk length, number of walks, and $q$ (controls local vs global exploration)

---

## Part 3: What We Observed

We applied both ASE and Node2Vec to all 3 generated networks and compared the results.

| | ASE | Node2Vec |
|---|---|---|
| **SBM** | 3 clusters, reasonable separation | Very clean separation |
| **DCSBM** | Elongated clusters (degree heterogeneity visible) | Clean separation |
| **RDPG** | Best recovery of latent structure | Struggled — communities overlap too much |

**Key insight:** Node2Vec crushes SBM/DCSBM because sharp discrete communities make random walks stay within communities naturally — clear co-occurrence signal. ASE is better for RDPG because it's mathematically designed to recover continuous latent geometry.

> Neither method is universally better — they capture **different aspects** of network structure. Choose based on what kind of network you have.

---

## Part 4: Real World — Cora Citation Network

Applied both methods to Cora — a real academic citation network with 2708 papers and 7 research topic categories (all ML-related).

- Network graph: completely mixed — all ML topics cite each other freely, no visual separation
- ASE: many small scattered clusters, captures local subgraph structure
- Node2Vec: slightly better topic grouping visible for some categories

**Takeaway:** Real networks are messy and hard — topics genuinely overlap. The fact that any clustering tendency is visible at all is meaningful. This is why Cora is a classic benchmark in graph ML research.

---

## Big Picture Summary

```
Real/Simulated Network (adjacency matrix A)
        ↓
Choose a generation model (SBM / DCSBM / RDPG)
        ↓
Apply embedding algorithm (ASE / Node2Vec)
        ↓
Each node → vector in low-dimensional space
        ↓
Use UMAP to visualize in 2D
        ↓
Nodes in same community cluster together (hopefully!)
```