

.. _example_applications_plot_out_of_core_classification.py:


======================================================
Out-of-core classification of text documents
======================================================

This is an example showing how scikit-learn can be used for classification
using an out-of-core approach: learning from data that doesn't fit into main
memory. We make use of an online classifier, i.e., one that supports the
partial_fit method, that will be fed with batches of examples. To guarantee
that the features space remains the same over time we leverage a
HashingVectorizer that will project each example into the same feature space.
This is especially useful in the case of text classification where new
features (words) may appear in each batch.

The dataset used in this example is Reuters-21578 as provided by the UCI ML
repository. It will be automatically downloaded and uncompressed on first run.

The plot represents the learning curve of the classifier: the evolution
of classification accuracy over the course of the mini-batches. Accuracy is
measured on the first 1000 samples, held out as a validation set.

To limit the memory consumption, we queue examples up to a fixed amount before
feeding them to the learner.



.. rst-class:: horizontal


    *

      .. image:: images/plot_out_of_core_classification_001.png
            :scale: 47

    *

      .. image:: images/plot_out_of_core_classification_002.png
            :scale: 47

    *

      .. image:: images/plot_out_of_core_classification_003.png
            :scale: 47

    *

      .. image:: images/plot_out_of_core_classification_004.png
            :scale: 47


**Script output**::

  Test set is 982 documents (90 positive)
    Passive-Aggressive classifier :          994 train docs (   121 positive)    982 test docs (    90 positive) accuracy: 0.939 in 1.38s (  721 docs/s)
            Perceptron classifier :          994 train docs (   121 positive)    982 test docs (    90 positive) accuracy: 0.934 in 1.38s (  719 docs/s)
                   SGD classifier :          994 train docs (   121 positive)    982 test docs (    90 positive) accuracy: 0.928 in 1.39s (  717 docs/s)
        NB Multinomial classifier :          994 train docs (   121 positive)    982 test docs (    90 positive) accuracy: 0.908 in 1.41s (  702 docs/s)
  
  
    Passive-Aggressive classifier :         3820 train docs (   520 positive)    982 test docs (    90 positive) accuracy: 0.969 in 3.72s ( 1027 docs/s)
            Perceptron classifier :         3820 train docs (   520 positive)    982 test docs (    90 positive) accuracy: 0.958 in 3.72s ( 1026 docs/s)
                   SGD classifier :         3820 train docs (   520 positive)    982 test docs (    90 positive) accuracy: 0.945 in 3.73s ( 1025 docs/s)
        NB Multinomial classifier :         3820 train docs (   520 positive)    982 test docs (    90 positive) accuracy: 0.919 in 3.75s ( 1017 docs/s)
  
  
    Passive-Aggressive classifier :         6759 train docs (   902 positive)    982 test docs (    90 positive) accuracy: 0.967 in 6.09s ( 1110 docs/s)
            Perceptron classifier :         6759 train docs (   902 positive)    982 test docs (    90 positive) accuracy: 0.926 in 6.09s ( 1109 docs/s)
                   SGD classifier :         6759 train docs (   902 positive)    982 test docs (    90 positive) accuracy: 0.963 in 6.10s ( 1108 docs/s)
        NB Multinomial classifier :         6759 train docs (   902 positive)    982 test docs (    90 positive) accuracy: 0.931 in 6.12s ( 1103 docs/s)
  
  
    Passive-Aggressive classifier :         9158 train docs (  1148 positive)    982 test docs (    90 positive) accuracy: 0.960 in 8.30s ( 1103 docs/s)
            Perceptron classifier :         9158 train docs (  1148 positive)    982 test docs (    90 positive) accuracy: 0.953 in 8.31s ( 1102 docs/s)
                   SGD classifier :         9158 train docs (  1148 positive)    982 test docs (    90 positive) accuracy: 0.954 in 8.31s ( 1102 docs/s)
        NB Multinomial classifier :         9158 train docs (  1148 positive)    982 test docs (    90 positive) accuracy: 0.930 in 8.34s ( 1098 docs/s)
  
  
    Passive-Aggressive classifier :        11953 train docs (  1510 positive)    982 test docs (    90 positive) accuracy: 0.962 in 10.73s ( 1114 docs/s)
            Perceptron classifier :        11953 train docs (  1510 positive)    982 test docs (    90 positive) accuracy: 0.962 in 10.73s ( 1113 docs/s)
                   SGD classifier :        11953 train docs (  1510 positive)    982 test docs (    90 positive) accuracy: 0.968 in 10.73s ( 1113 docs/s)
        NB Multinomial classifier :        11953 train docs (  1510 positive)    982 test docs (    90 positive) accuracy: 0.939 in 10.76s ( 1110 docs/s)
  
  
    Passive-Aggressive classifier :        14338 train docs (  1778 positive)    982 test docs (    90 positive) accuracy: 0.969 in 12.96s ( 1105 docs/s)
            Perceptron classifier :        14338 train docs (  1778 positive)    982 test docs (    90 positive) accuracy: 0.959 in 12.97s ( 1105 docs/s)
                   SGD classifier :        14338 train docs (  1778 positive)    982 test docs (    90 positive) accuracy: 0.958 in 12.97s ( 1105 docs/s)
        NB Multinomial classifier :        14338 train docs (  1778 positive)    982 test docs (    90 positive) accuracy: 0.937 in 13.00s ( 1102 docs/s)
  
  
    Passive-Aggressive classifier :        17255 train docs (  2123 positive)    982 test docs (    90 positive) accuracy: 0.965 in 15.35s ( 1124 docs/s)
            Perceptron classifier :        17255 train docs (  2123 positive)    982 test docs (    90 positive) accuracy: 0.963 in 15.35s ( 1124 docs/s)
                   SGD classifier :        17255 train docs (  2123 positive)    982 test docs (    90 positive) accuracy: 0.958 in 15.35s ( 1123 docs/s)
        NB Multinomial classifier :        17255 train docs (  2123 positive)    982 test docs (    90 positive) accuracy: 0.940 in 15.38s ( 1121 docs/s)



**Python source code:** :download:`plot_out_of_core_classification.py <plot_out_of_core_classification.py>`

.. literalinclude:: plot_out_of_core_classification.py
    :lines: 25-

**Total running time of the example:**  16.83 seconds
( 0 minutes  16.83 seconds)
    