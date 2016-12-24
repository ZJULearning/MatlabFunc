
Hashing Algorithms for Approximate Nearest Neighbor Search
=============

We provide here the matlab codes of some popular hashing algorithms. Most of the core functions are written by the author of the original paper, we simply add some wrapper code for the interface.

These codes are used in our paper: [A Revisit of Hashing Algorithms for Approximate Nearest Neighbor Search](http://arxiv.org/abs/1612.07545)

Currently, the following hashing algorithms are included:

* LSH (Locality Sensitive Hashing) [1]
* SH (Spectral Hashing) [2]
* KLSH (Kernelized Locality Sensitive Hashing) [3]
* BRE (Binary Reconstructive Embeddings) [4]
* USPLH (Unsupervised Sequential Projection Learning Hashing) [5]
* ITQ (Iterative Quantization) [6]
* AGH (Anchor Graph Hashing) [7]
* SpH (Spherical Hashing) [8]
* IsoH (Isotropic Hashing) [9]
* CH (Compressed Hashing) [10]
* CPH (Complementary Projection Hashing) [11]
* HamH (Harmonious Hashing) [12]
* DSH (Density Sensitive Hashing) [13]

If you have a new hashing algorithm and want to share with us, welcome to send the matlab code to me and I will include it in this package.

All these hashing algorithms are in the **Unsupervised** folder

* The *HashingRun.m* is used to call a specified hashing algorithm (without landmarks input, eg, LSH, SH, BRE, USPLH, ITQ, SpH, IsoH) to generate the binary code of the base set and the query set.
* The *HashingRunLandmark.m* is used to call a specified hashing algorithm (with landmarks input, eg, KLSH, AGH, CH, CPH, HamH) to generate the binary code of the base set and the query set.
* The *GenLandmarks.m* is used to generate and save landmarks.

The binary code of the base and query sets can be save in a TXT or binary format. Please read the code for details.

The binary code file can be used as the hash index of the [Search with Hash Index](https://github.com/fc731097343/efanna/tree/master/samples_hashing) algorithm.

If you have some problems or find some bugs in the codes, please email: dengcai AT gmail DOT com



1. Aristides Gionis, Piotr Indyk, Rajeev Motwani: Similarity Search in High Dimensions via Hashing. VLDB 1999: 518-529   
2. Yair Weiss, Antonio Torralba, Robert Fergus: Spectral Hashing. NIPS 2008: 1753-1760   
3. Brian Kulis, Kristen Grauman: Kernelized locality-sensitive hashing for scalable image search. 2130-2137, ICCV 2009   
4. Brian Kulis, Trevor Darrell: Learning to Hash with Binary Reconstructive Embeddings. NIPS 2009: 1042-1050   
5. Jun Wang, Sanjiv Kumar, Shih-Fu Chang: Sequential Projection Learning for Hashing with Compact Codes. ICML 1127-1134, 2010   
6. Yunchao Gong, Svetlana Lazebnik: Iterative quantization: A procrustean approach to learning binary codes. CVPR 2011   
7. Wei Liu, Jun Wang, Sanjiv Kumar, Shih-Fu Chang: Hashing with Graphs. ICML 2011   
8. Jae-Pil Heo, Youngwoon Lee, Junfeng He, Shih-Fu Chang, Sung-Eui Yoon: Spherical hashing. CVPR 2012   
9. Weihao Kong, Wu-Jun Li: Isotropic Hashing. NIPS 1655-1663 2012   
10. Yue Lin, Rong Jin, Deng Cai, Shuicheng Yan, Xuelong Li: Compressed Hashing. CVPR 2013   
11. Zhongming Jin, Yao Hu, Yue Lin, Debing Zhang, Shiding Lin, Deng Cai, Xuelong Li: Complementary Projection Hashing. 257-264 ICCV 2013   
12. Bin Xu, Jiajun Bu, Yue Lin, Chun Chen, Xiaofei He, Deng Cai: Harmonious Hashing. 1820-1826 IJCAI 2013   
13. Zhongming Jin, Cheng Li, Yue Lin, Deng Cai: Density Sensitive Hashing. IEEE Trans. Cybernetics 44(8): 1362-1371 (2014)   
