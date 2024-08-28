# T-SNE


t-SNE (t-Distributed Stochastic Neighbor Embedding) is a non-linear dimensionality reduction technique primarily used for visualizing high-dimensional data in a low-dimensional space (typically 2D or 3D). It is particularly effective for preserving the local structure of the data, meaning that it tries to ensure that points that are close together in the high-dimensional space remain close together in the low-dimensional space.



High-Dimensional Space: The original space where the data points reside. Each data point is represented as a vector in this space.

Low-Dimensional Space: The space in which we want to embed the data points, typically 2D or 3D for visualization purposes.

Perplexity: A parameter related to the number of nearest neighbors that is used to balance between local and global aspects of the data. It is typically chosen in the range [5, 50].

Joint Probability Distribution: t-SNE aims to match the distribution of pairwise similarities in the high-dimensional space with the distribution of pairwise similarities in the low-dimensional space.



Input Data: Let X be a dataset with n data points in d-dimensional space. Each data point is represented as a vector x_i in R^d.

Pairwise Affinities in High-Dimensional Space: The pairwise affinities (or similarities) between data points in the high-dimensional space are modeled using a Gaussian distribution. The similarity between points x_i and x_j is defined as:
p_ij = exp(-||x_i - x_j||^2 / 2*sigma_i^2) / sum_k!=i exp(-||x_i - x_k||^2 / 2*sigma_i^2)
where:
||x_i - x_j||^2 is the squared Euclidean distance between x_i and x_j.
sigma_i is the bandwidth of the Gaussian kernel for point x_i, determined such that the perplexity of the distribution equals a predefined value.
The joint probability P_ij is symmetrized as:
P_ij = (p_ij + p_ji) / 2n

Pairwise Affinities in Low-Dimensional Space: In the low-dimensional space, the pairwise affinities are modeled using a Student's t-distribution with one degree of freedom (which has heavier tails than a Gaussian). The similarity between points y_i and y_j is defined as:
q_ij = (1 + ||y_i - y_j||^2)^(-1) / sum_k!=l (1 + ||y_k - y_l||^2)^(-1)
where y_i and y_j are the low-dimensional embeddings of x_i and x_j, respectively.

Kullback-Leibler (KL) Divergence: The objective of t-SNE is to minimize the KL divergence between the joint probability distributions P (in high-dimensional space) and Q (in low-dimensional space). The KL divergence is defined as:
KL(P || Q) = sum_i sum_j P_ij * log(P_ij / Q_ij)
The optimization aims to adjust the low-dimensional embeddings Y to minimize this divergence, thus preserving the pairwise similarities.

Gradient Descent: The embeddings Y are updated iteratively using gradient descent to minimize the KL divergence. The gradient with respect to the positions y_i is computed as:
dKL/dy_i = 4 * sum_j (P_ij - Q_ij) * (y_i - y_j) * (1 + ||y_i - y_j||^2)^(-1)


Compute Pairwise Affinities in High-Dimensional Space: For each pair of points in the high-dimensional space, compute the probability P_ij using a Gaussian distribution. For each pair of points in the low-dimensional space, compute the probability Q_ij using a Student's t-distribution. Use gradient descent to adjust the low-dimensional embeddings Y to minimize the KL divergence between P and Q.
