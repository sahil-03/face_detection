# Face Detection

There are several ways we can do face detection. For this sort of image classification, we can train something like a CNN by providing it labeled examples of my face and training it to detect my face. However, this would require careful data collection and a heavier model consisting of several layers (convolution, pooling, linear, etc...). 

In matrix theory, there is a neat trick for dimensionality reduction called Singular-Value Decomposition (SVD). This is when an inout matrix X is broken into U, S, V^T (X = USV^T), where U is an orthonormal basis of the column space, S is a diagonal matrix consisting of singular values describing the transformation of singular vectors by X in their respective dimensions, and V^T is the orthonormal basis of the row space. 

This proves to be handy when trying to do image classification. We can line up a dataset of faces into the columns X and then decompose it. The U matrix is most helpful as it is an orthonormal basis of the column space, therefore, behaves as the representation space of all faces. Projecting a new face into this orthonormal basis returns a unique vector in space that represents that face (alpha = U^T @ x). To reconstruct the face, we can do x_recon = U @ (U^T @ x). 

In this mini-project, I create light-weight face detection using opencv and SVD to try to uniquely represent faces, perform classification, and reconstruct faces from the latent space of all faces represented by decomposing the original input matrix into U (and other components).  

*Note: this was meant for experimentation rather than production-code. 

