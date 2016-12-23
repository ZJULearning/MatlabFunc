
Hashing Algorithms for Approximate Nearest Neighbor Search
=============

We provide here the matlab codes of some popular hashing algorithms. Most of the core functions are written by the author of the original paper, we simply add some wrapper code for the interface.

These codes are used in our paper: [A Revisit of Hashing Algorithms for Approximate Nearest Neighbor Search](http://arxiv.org/abs/1612.07545)

Currently, the following hashing algorithms are included:

* LSH (Locality Sensitive Hashing)
* SH (Spectral Hashing)
* KLSH (Kernelized Locality Sensitive Hashing)
* BRE (Binary Reconstructive Embeddings)
* USPLH (Unsupervised Sequential Projection Learning Hashing)
* ITQ (Iterative Quantization)
* AGH (Anchor Graph Hashing)
* SpH (Spherical Hashing)
* IsoH (Isotropic Hashing)
* CH (Compressed Hashing)
* CPH (Complementary Projection Hashing)
* HamH (Harmonious Hashing)

If you have a new hashing algorithm and want to share with us, welcome to send the matlab code to me and I will include it in this package.

All these hashing algorithms are in the **Unsupervised** folder

* The *HashingRun.m* is used to call a specified hashing algorithm (without landmarks input, eg, LSH, SH, BRE, USPLH, ITQ, SpH, IsoH) to generate the binary code of the base set and the query set.
* The *HashingRunLandmark.m* is used to call a specified hashing algorithm (with landmarks input, eg, KLSH, AGH, CH, CPH, HamH) to generate the binary code of the base set and the query set.
* The *GenLandmarks.m* is used to generate and save landmarks.

The binary code of the base and query sets can be save in a TXT or binary format. Please read the code for details.

The binary code file can be used as the hash index of the [Search with Hash Index](https://github.com/fc731097343/efanna/tree/master/samples_hashing) algorithm.

If you have some problems or find some bugs in the codes, please email: dengcai AT gmail DOT com
