## Hierarchical VAE network-based molecular generation and virtual screening of CYP2C19 inhibitors
## Installation
Python 3.7.9
Pytorch 1.7.1
RDKit >= 2019.03
...

### MMNN CYP2C19 inhibitor prediction model
We trained the three models independently, then passed the outputs of the three models through a fully connected layer to obtain the final output.



### HVAE generative model
The source code for the generative model is referenced from：The generative models Hierarchical Generation of Molecular Graphs using Structural Motifs:[GitHub](https://github.com/wengong-jin/hgraph2graph)
