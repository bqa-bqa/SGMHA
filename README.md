 # SGMHA: Semantic Graph Reconstruction with
Multi-Head Attention for Gene Regulatory
Network Inference

## Overview
SGMHA is a two-stage framework thatintegrates graph masked autoencoder (GraphMAE) pre-training with multi-head atention for supervisedfine-tuning to infer gene regulatory networks (GRNs) from single-cell RNA-seq data.

- Self-supervised Pre-training: Learns robust gene representations through self supervised reconstruction of masked node features. Thisphase captures intrinsic expression patterns and topological dependencies without relying on labeled links.
- Supervised Fine-tuning: Pre-trained embeddings are dynamically integrated with the original graph features and raw expression datavia a multi-head attention mechanism. This fusion enables the model to jointly capture rich semantic information and topologicalcontext for precise link prediction.
## File Structure
- `main.py` - Main program entry
- `scGNN.py` - Graph neural network model definition
- `PytorchTools.py` - Data processing tools
- `utils.py` - Utility functions and evaluation metrics

## Installation
```bash
pip install torch pandas numpy scipy scikit-learn tqdm
```

## Usage
Data Input Format

SGMHA requires the single-cell RNA-seq data to be formatted as follows:

Gene Expression Matrix: A matrix $X\in\mathbb{R}^{N\times M}$ where $N$ is the number of genes and $M$ is the number of cells.



Prior Adjacency Matrix: A matrix $A\in\{0,1\}^{N\times N}$ containing known or prior gene interactions.

Running the Model

To train and run the model, simply execute the main script in the root directory:

```Bash
python main.py
```

## Contact
For any questions or further communication, please contact the corresponding authors:
- Email: 2152608@tongji.edu.cn; jhguan@tongji.edu.cn; 23310342@tongji.edu.cn
