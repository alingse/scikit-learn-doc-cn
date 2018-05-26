.. _biclustering:

============
双向聚类(Biclustering)
============

:mod:`sklearn.cluster.bicluster` 模块可以进行双向聚类。
双向聚类算法同时对数据矩阵的行和列进行聚类。这些行和列上的聚类结果就被称为双聚类。
每一个聚类结果使用一些必需的特性确定了源数据矩阵的一个子矩阵。
比如，给定一个shape为 ``(10, 10)``的矩阵，一个可能的聚类结果是有3行2列，即一个shape为``(3, 2)``的子矩阵::

    >>> import numpy as np
    >>> data = np.arange(100).reshape(10, 10)
    >>> rows = np.array([0, 2, 3])[:, np.newaxis]
    >>> columns = np.array([1, 2])
    >>> data[rows, columns]
    array([[ 1,  2],
           [21, 22],
           [31, 32]])

为了可以可视化，给定一个双向聚类结果，数据矩阵的行和列可能会被重排以便让双向聚类连续。

算法在怎么定义双向聚类有区别。一些常见的类型包含:

* 固定数值，固定行，固定列
* 异常高的或低数值
* 具有低方差的子矩阵
* 相关的行或相关的列

算法也在行和列如何分配给聚类上有差别，这样会导致不同的聚类结构。行和列分配给不同的分区时，会出现快对角线或者棋盘结构。

如果每行或者每一列仅属于某个聚类时，重新排列数据矩阵的行和列会展示出对角线上的双聚类结果。
这儿有一个这种结构的例子，双聚类结果比其他的行或列有更高的平均值:

.. figure:: ../auto_examples/bicluster/images/plot_spectral_coclustering_003.png
   :target: ../auto_examples/bicluster/images/plot_spectral_coclustering_003.png
   :align: center
   :scale: 50

   一个双向聚类的例子，由分块的行和列组成。

在棋盘模式下，每个行属于所有的列向聚类，每个列属于所有的行聚类（译者：大致是任一行可以划成若干个部分属于所有的列方向聚类，反之亦然）。 这里是一个这种结构的例子，每个聚类结果里的值的方差都小:

.. figure:: ../auto_examples/bicluster/images/plot_spectral_biclustering_003.png
   :target: ../auto_examples/bicluster/images/plot_spectral_biclustering_003.png
   :align: center
   :scale: 50

   一个棋盘的双向聚类集例子.

适配一个模型之后，行和列的聚合关系能够由 ``rows_`` and ``columns_`` 属性发现， ``rows_[i]`` 是一个二值向量，非零的条目对应该行属于双向聚类结果 ``i`` 。相似的，``columns_[i]`` 表明哪一列属于双向聚类结果 ``i`` 。

一些模型也会有 ``row_labels_`` 和 ``column_labels_`` 属性。
这些模型分割行和列，就和块对角和棋盘 双向聚类结构一样。

.. note::

    双向聚类在不同的领域有许多其他的名字，包括
    联合聚类(co-clustering),双模式聚类(two-mode clustering),
    两路聚类(two-way clustering),块聚类(block clustering),
    耦合两路聚类(coupled two-way clustering), 等等。一些算法的名字，比如
    谱联合聚类,反映出这些替代的名字(原文：The names of some algorithms reflect these alternate names.)。


.. currentmodule:: sklearn.cluster.bicluster


.. _spectral_coclustering:

谱联合聚类(Spectral Co-Clustering)
===================================

谱联合聚类 :class:`SpectralCoclustering` 算法会找出
那些值高于相应的其他行和列中的值的聚类结构，由于每行和每列只会属于某一个聚类结构的，
所以重新排列行和列以使较高的值分块连续显示在这个聚类矩阵的对角线上。

.. note::

  该算法将输入数据矩阵视为二分图(bipartite graph)：矩阵的行和列对应于两组顶点，
  并且每个条目对应于行和列之间的一条边。
  该算法逼近该图的归一化切分(normalized cut)以找到重子图(heavy subgraphs)。


数学描述(Mathematical formulation)
---------------------------------

通过二分图的拉普拉斯算子的广义特征值分解，可以找到一种最佳归一化切分(optimal normalized cut)的近似解决方案。
通常这意味着要直接使用拉普拉斯算子矩阵(Laplacian matrix)。

如果原始数据矩阵 :math:`A` 尺寸为 :math:`m \times n`, 那么对应二分图的拉普拉斯算子矩阵(Laplacian matrix)
的尺寸为 :math:`(m + n) \times (m + n)`。但是，在这种情况下
可能直接使用矩阵 ：math：`A`，它的尺寸更小且更高效。

The input matrix :math:`A` is preprocessed as follows:

.. math::
    A_n = R^{-1/2} A C^{-1/2}

Where :math:`R` is the diagonal matrix with entry :math:`i` equal to
:math:`\sum_{j} A_{ij}` and :math:`C` is the diagonal matrix with
entry :math:`j` equal to :math:`\sum_{i} A_{ij}`.

The singular value decomposition, :math:`A_n = U \Sigma V^\top`,
provides the partitions of the rows and columns of :math:`A`. A subset
of the left singular vectors gives the row partitions, and a subset
of the right singular vectors gives the column partitions.

The :math:`\ell = \lceil \log_2 k \rceil` singular vectors, starting
from the second, provide the desired partitioning information. They
are used to form the matrix :math:`Z`:

.. math::
    Z = \begin{bmatrix} R^{-1/2} U \\\\
                        C^{-1/2} V
          \end{bmatrix}

where the the columns of :math:`U` are :math:`u_2, \dots, u_{\ell +
1}`, and similarly for :math:`V`.

Then the rows of :math:`Z` are clustered using :ref:`k-means
<k_means>`. The first ``n_rows`` labels provide the row partitioning,
and the remaining ``n_columns`` labels provide the column partitioning.


.. topic:: Examples:

 * :ref:`example_bicluster_plot_spectral_coclustering.py`: A simple example
   showing how to generate a data matrix with biclusters and apply
   this method to it.

 * :ref:`example_bicluster_bicluster_newsgroups.py`: An example of finding
   biclusters in the twenty newsgroup dataset.


.. topic:: References:

 * Dhillon, Inderjit S, 2001. `Co-clustering documents and words using
   bipartite spectral graph partitioning
   <http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.140.3011>`__.


.. _spectral_biclustering:

Spectral Biclustering
=====================

The :class:`SpectralBiclustering` algorithm assumes that the input
data matrix has a hidden checkerboard structure. The rows and columns
of a matrix with this structure may be partitioned so that the entries
of any bicluster in the Cartesian product of row clusters and column
clusters is are approximately constant. For instance, if there are two
row partitions and three column partitions, each row will belong to
three biclusters, and each column will belong to two biclusters.

The algorithm partitions the rows and columns of a matrix so that a
corresponding blockwise-constant checkerboard matrix provides a good
approximation to the original matrix.


Mathematical formulation
------------------------

The input matrix :math:`A` is first normalized to make the
checkerboard pattern more obvious. There are three possible methods:

1. *Independent row and column normalization*, as in Spectral
   Co-Clustering. This method makes the rows sum to a constant and the
   columns sum to a different constant.

2. **Bistochastization**: repeated row and column normalization until
   convergence. This method makes both rows and columns sum to the
   same constant.

3. **Log normalization**: the log of the data matrix is computed: :math:`L =
   \log A`. Then the column mean :math:`\overline{L_{i \cdot}}`, row mean
   :math:`\overline{L_{\cdot j}}`, and overall mean :math:`\overline{L_{\cdot
   \cdot}}` of :math:`L` are computed. The final matrix is computed
   according to the formula

.. math::
    K_{ij} = L_{ij} - \overline{L_{i \cdot}} - \overline{L_{\cdot
    j}} + \overline{L_{\cdot \cdot}}

After normalizing, the first few singular vectors are computed, just
as in the Spectral Co-Clustering algorithm.

If log normalization was used, all the singular vectors are
meaningful. However, if independent normalization or bistochastization
were used, the first singular vectors, :math:`u_1` and :math:`v_1`.
are discarded. From now on, the "first" singular vectors refers to
:math:`u_2 \dots u_{p+1}` and :math:`v_2 \dots v_{p+1}` except in the
case of log normalization.

Given these singular vectors, they are ranked according to which can
be best approximated by a piecewise-constant vector. The
approximations for each vector are found using one-dimensional k-means
and scored using the Euclidean distance. Some subset of the best left
and right singular vector are selected. Next, the data is projected to
this best subset of singular vectors and clustered.

For instance, if :math:`p` singular vectors were calculated, the
:math:`q` best are found as described, where :math:`q<p`. Let
:math:`U` be the matrix with columns the :math:`q` best left singular
vectors, and similarly :math:`V` for the right. To partition the rows,
the rows of :math:`A` are projected to a :math:`q` dimensional space:
:math:`A * V`. Treating the :math:`m` rows of this :math:`m \times q`
matrix as samples and clustering using k-means yields the row labels.
Similarly, projecting the columns to :math:`A^{\top} * U` and
clustering this :math:`n \times q` matrix yields the column labels.


.. topic:: Examples:

 * :ref:`example_bicluster_plot_spectral_biclustering.py`: a simple example
   showing how to generate a checkerboard matrix and bicluster it.


.. topic:: References:

 * Kluger, Yuval, et. al., 2003. `Spectral biclustering of microarray
   data: coclustering genes and conditions
   <http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.135.1608>`__.


.. _biclustering_evaluation:

.. currentmodule:: sklearn.metrics

Biclustering evaluation
=======================

There are two ways of evaluating a biclustering result: internal and
external. Internal measures, such as cluster stability, rely only on
the data and the result themselves. Currently there are no internal
bicluster measures in scikit-learn. External measures refer to an
external source of information, such as the true solution. When
working with real data the true solution is usually unknown, but
biclustering artificial data may be useful for evaluating algorithms
precisely because the true solution is known.

To compare a set of found biclusters to the set of true biclusters,
two similarity measures are needed: a similarity measure for
individual biclusters, and a way to combine these individual
similarities into an overall score.

To compare individual biclusters, several measures have been used. For
now, only the Jaccard index is implemented:

.. math::
    J(A, B) = \frac{|A \cap B|}{|A| + |B| - |A \cap B|}

where :math:`A` and :math:`B` are biclusters, :math:`|A \cap B|` is
the number of elements in their intersection. The Jaccard index
achieves its minimum of 0 when the biclusters to not overlap at all
and its maximum of 1 when they are identical.

Several methods have been developed to compare two sets of biclusters.
For now, only :func:`consensus_score` (Hochreiter et. al., 2010) is
available:

1. Compute bicluster similarities for pairs of biclusters, one in each
   set, using the Jaccard index or a similar measure.

2. Assign biclusters from one set to another in a one-to-one fashion
   to maximize the sum of their similarities. This step is performed
   using the Hungarian algorithm.

3. The final sum of similarities is divided by the size of the larger
   set.

The minimum consensus score, 0, occurs when all pairs of biclusters
are totally dissimilar. The maximum score, 1, occurs when both sets
are identical.


.. topic:: References:

 * Hochreiter, Bodenhofer, et. al., 2010. `FABIA: factor analysis
   for bicluster acquisition
   <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2881408/>`__.
