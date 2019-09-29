# CyTOF-annotator

Mass Cytometry by time-of-flight (CyTOF) is a commonly used technology to study the variation in immune cell populations by simultaneously measuring the expression of 40-50 protein markers in millions of single cells. Traditionally, for the identification of cell types, a clustering method is employed which uses cell surface marker expression profiles to group similar cell-types. While being instrumental in analyzing the high-dimensional CyTOF datasets, current clustering-based strategies face a number of limitations. For instance, for larger datasets, sub-sampling is routinely performed (e.g. often only 10% or even less of all events are used), and randomly selected cells are assumed to be the representative of entire cell population.  The primary reason of sub-sampling is to reduce computational time and memory use, which consequently reduces the probability of annotating non-canonical cells with small population size along with significant data loss. Moreover, the clustering event of a cell to a given group varies with respect to neighboring cells, making the cell annotation difficult. This statistical reoccurrence of a given cell within a single cell-type cluster in spite of varying neighboring cells could be utilized for assigning it to a statistically most probable cell-type.


# Algorithm

Model training: 

The algorithm begins with the following multi-step procedure to calculate local and non-linear boundaries for each cell type using the hand-gated cells:
  1. Splitting: The hand-gated cells are pooled and then split into- training (80%) and test set (20%), wherein the cell type label proportion remains the same in both the sets.
  2. Landmarks: In the training set, the cell-cell closeness per cell type are identified by decomposing the high-dimensional data points into two Principal Components (PCs) followed by estimating the kernel densities of each cell. Using this, the total number of landmark cells per cell type is determined as the set of non-similar cells from the training set that are able to predict the expected cells in the test set using nearest neighborhood identification.

Quadratic Discriminant Analysis: 

    Using the training data set, the non-linear boundaries for cell type separation are modeled by performing discriminant analysis, QDA, followed by model evaluation on test data set.

  Cell type prediction:

    Index Blocking: Each sample of independent live cells dataset is split into a defined number of blocks of equal numbers of randomly selected unique cell.

  Bootstrapping: 

    In each epoch, randomly selected blocks, one from each sample, are shuffled to create a unique sub-sample that undergoes cell type identification steps. Next, the approximated nearest neighbor cells for each landmark are determined. The approximated landmark neighbors are further clustered by GMM. The cells that cluster together with landmarks are shortlisted for QDA classification, whereby its posterior probability for a given cell type is used to compute its final score across all epochs.

Scoring: 

    The final score for each cell, one for each cell type, is computed using the following equation:
    where, p1 and p2 are the GMM and QDA posterior probabilities respectively, in each iteration i of total epochs e. 𝛾 is the error rate (QDA) for cell type ct. The credibility of the prediction for each cell is determined by evaluating the variance in the cell types found to be associated with it across all epochs.

   Cluster stability:
   
      The cluster stability and the core cells associated with stable clusters are predicted by enumerating the variations associated with its cells. For downstream biological analysis only stable cells enriched within a given cell type may be used for reliable estimation of the differential behavior in their functional markers.

