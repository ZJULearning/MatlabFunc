
Hashing Algorithms for Approximate Nearest Neighbor Search
=============

We provide here the matlab codes of some popular hashing algorithms. Most of the core functions are written by the author of the original paper, we simply add some wrapper codes for the interface.

These codes are used in our paper: [A Revisit of Hashing Algorithms for Approximate Nearest Neighbor Search](http://arxiv.org/abs/1612.07545)

Currently, the following hashing algorithms are included:

* LSH (Locality Sensitive Hashing) [1] [2]
* SH (Spectral Hashing) [3]
* KLSH (Kernelized Locality Sensitive Hashing) [4]
* BRE (Binary Reconstructive Embeddings) [5]
* USPLH (Unsupervised Sequential Projection Learning Hashing) [6]
* ITQ (Iterative Quantization) [7]
* AGH (Anchor Graph Hashing) [8]
* SpH (Spherical Hashing) [9]
* IsoH (Isotropic Hashing) [10]
* CH (Compressed Hashing) [11]
* CPH (Complementary Projection Hashing) [12]
* HamH (Harmonious Hashing) [13]
* DSH (Density Sensitive Hashing) [14]

If you have a new hashing algorithm and want to share with us, welcome to send the matlab code to me and I will include it in this package.

All these hashing algorithms are in the **Unsupervised** folder

* The *HashingRunLongCode.m* is used to call a specified hashing algorithm (eg, LSH, SH, BRE ...) to generate the binary code of the base set and the query set, and the long code will be stored in multiple 32-bits tables for convenience.
* The *GenKmeansPartitions.m* is used to generate kmeans partitions of the data.

The binary format code file and the partition file can be used as the hash index of the [Search with Hash Index](https://github.com/ZJULearning/hashingSearch) algorithm.

If you have some problems or find some bugs in the codes, please email: dengcai AT gmail DOT com



1. Aristides Gionis, Piotr Indyk, Rajeev Motwani: Similarity Search in High Dimensions via Hashing. VLDB 1999: 518-529   
2. Moses S. Charikar: Similarity estimation techniques from rounding algorithms. Proceedings of the thiry-fourth annual ACM symposium on Theory of computing, 2002.  
3. Yair Weiss, Antonio Torralba, Robert Fergus: Spectral Hashing. NIPS 2008: 1753-1760   
4. Brian Kulis, Kristen Grauman: Kernelized locality-sensitive hashing for scalable image search. 2130-2137, ICCV 2009   
5. Brian Kulis, Trevor Darrell: Learning to Hash with Binary Reconstructive Embeddings. NIPS 2009: 1042-1050   
6. Jun Wang, Sanjiv Kumar, Shih-Fu Chang: Sequential Projection Learning for Hashing with Compact Codes. ICML 1127-1134, 2010   
7. Yunchao Gong, Svetlana Lazebnik: Iterative quantization: A procrustean approach to learning binary codes. CVPR 2011   
8. Wei Liu, Jun Wang, Sanjiv Kumar, Shih-Fu Chang: Hashing with Graphs. ICML 2011   
9. Jae-Pil Heo, Youngwoon Lee, Junfeng He, Shih-Fu Chang, Sung-Eui Yoon: Spherical hashing. CVPR 2012   
10. Weihao Kong, Wu-Jun Li: Isotropic Hashing. NIPS 1655-1663 2012   
11. Yue Lin, Rong Jin, Deng Cai, Shuicheng Yan, Xuelong Li: Compressed Hashing. CVPR 2013   
12. Zhongming Jin, Yao Hu, Yue Lin, Debing Zhang, Shiding Lin, Deng Cai, Xuelong Li: Complementary Projection Hashing. 257-264 ICCV 2013   
13. Bin Xu, Jiajun Bu, Yue Lin, Chun Chen, Xiaofei He, Deng Cai: Harmonious Hashing. 1820-1826 IJCAI 2013   
14. Zhongming Jin, Cheng Li, Yue Lin, Deng Cai: Density Sensitive Hashing. IEEE Trans. Cybernetics 44(8): 1362-1371 (2014)   
