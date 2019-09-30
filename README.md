# CyTOF-annotator (under development)

Abstract:

Mass Cytometry by time-of-flight (CyTOF) is a commonly used technology to study the variation in immune cell populations by simultaneously measuring the expression of 40-50 protein markers in millions of single cells. Traditionally, for the identification of cell types, a clustering method is employed which uses cell surface marker expression profiles to group similar cell-types. While being instrumental in analyzing the high-dimensional CyTOF datasets, current clustering-based strategies face a number of limitations. For instance, for larger datasets, sub-sampling is routinely performed (e.g. often only 10% or even less of all events are used), and randomly selected cells are assumed to be the representative of entire cell population.  The primary reason of sub-sampling is to reduce computational time and memory use, which consequently reduces the probability of annotating non-canonical cells with small population size along with significant data loss. Moreover, the clustering event of a cell to a given group varies with respect to neighboring cells, making the cell annotation difficult. This statistical reoccurrence of a given cell within a single cell-type cluster in spite of varying neighboring cells could be utilized for assigning it to a statistically most probable cell-type.

The detailed workflow can be found in the Workflow.pdf 

The sample dataset: https://drive.google.com/file/d/1JD-LreUVSCrxa6ZCigPUKksAcnbDjdgz/view?usp=sharing
