'''@author: kaushik, abhinav'''
from functions import *
#import numpy as np
#import pandas as pa
#import os
#import fcsparser


'''////// input Arguments //////////'''
Fileinfo= "Panorama/FileInfo.csv" ## Mandatory Live cells to be tested for annotation
handgated="Panorama/HandGatedFileInfo.csv" ## Mandatory hand-gated cells to be used for training
### this is for PANORMA
relevantMarkers = ['Ter119','CD45.2','Ly6G','IgD','CD11c','F480','CD3','NKp46','CD23','CD34','CD115','CD19','120g8',
'CD8','Ly6C','CD4','CD11b','CD27','CD16_32','SiglecF','Foxp3','B220','CD5','FceR1a','TCRgd','CCR7','Sca1','CD49b',
'cKit','CD150','CD25','TCRb','CD43','CD64','CD138','CD103','IgM','CD44','MHCII']

#### this is for POISED
Fileinfo= "handgated_POISED_BL_dropouts/FileInfo.csv" ## Mandatory Live cells to be tested for annotation
handgated="handgated_POISED_BL_dropouts/HandGatedFileInfo.csv" ## Mandatory hand-gated cells to be used for training
relevantMarkers = ['CD19','CD49b','CD4','CD8','CD20','LAG3','CD123','CD3','HLA.DR','CD69',
                   'CD33','CD11c','CD14','CD127','CCR7','CD25','CD56','TCRgd','CD16',
                   'CD40L','CD45RA','CD40L']

head = 'infer' ## or None ; does the csv expression files contain header (marker name), instaed of metal ion or any random number
normalizeCell=False ## or True
filterCells = False ## or True
postProb=0.5 # #if the prob. is less than this value the cell will be assigned as "Unlnown" else whatever predicted by the algorithm; this will increase or confidence
UMAPplot=False ## or False ## warning: may be slow depending upon number of cells
cellCount = 200 ## minimum number of cells a celltype must have in handgated cells to be used; all handgated cells with cell count less than this number will be removed
bn = 10
blockNum = 10 ## how many fixed number of blocks you want in each FCS file; useful only when limit = "auto"; otherwise ignored no matter what you write
smallSample=True ## if False, CSV/FCS files with small cell count will be automatically ignore; if True program will halt and raise an error ## if celimit = 'auto', then smallSample = true , i.e. will be taken into analysis
limit = 'auto' ## 'auto' or number of cells to be used in each block, i.e. number of cells in block is fixed, block count varies for each file
epoch = 'auto' ## number of  times ## set 'auto' it will automatically decide the number of epochs required so that all the blocks get iterated atleast 'allowed' number of iterations
allowed = 100 ## this is an imp. term, when calculating number of epochs, it get multiplies with the total number blocks present in sample having maximum number of blocks
              ## if this is 100 it means that all the blocks will be iterated 100 times in the sample having highest number of blocks ; yet samples with samller number of blocks may get repeated
nlandMarks = 10 ## number of landmarks cells you need from each cell type.
LandmarkPlots = False ## scatter plots that shows where are high density cells ; estimated using kernel density estimation
Maxlimit = 100000 ## maximum number of cells to be allowed in each cell-type for fitting GMM model and land mark identification
shrinkage = None ## or True ; for LDA only ; if True then =>>  solver='lsqr', shrinkage='auto' will be used for LDA : enable only for testing; its a good approach when you have less cells than markers :D
herenthersh = 3 ## 'Here N there' threshold; cells moving to celltypes larger than this number will be considered as randomly moving cells; artifacts; called as Here N there cells
##********************##


'''////// Program start //////////'''
train = method0(handgated,filterCells,normalizeCell,relevantMarkers,cellCount,head,UMAPplot)
test = method1(Fileinfo,normalizeCell,relevantMarkers,'infer')
OutPut2=e2b(test=test,
    train=train,
    limit="auto",
    blockNum = bn,
    smallSample=True,
    relevantMarkers=relevantMarkers,
    shrinkage=shrinkage,
    Maxlimit=Maxlimit,
    nlandMarks=nlandMarks,
    allowed=allowed,
    epoch=epoch,
    LandmarkPlots=LandmarkPlots,
    herenthersh=herenthersh)











