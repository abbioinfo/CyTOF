#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 15:07:39 2019

@author: abhinavkaushik
"""
#import gc
#import pickle
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import sys
import warnings
import random
import itertools
import fcsparser
import umap
import faiss
import time
import re
import math
import collections
import numpy as np
import pandas as pa
from fastKDE import *
from scipy import mean
from scipy import stats
from scipy import nanstd
from scipy import nanmean
from scipy.stats import sem, t
from scipy import spatial
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
#from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns; sns.set(color_codes=True)
from sklearn.manifold import TSNE
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn import preprocessing
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture as GMM
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
#from sklearn import svm, grid_search
#from operator import itemgetter
#from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
#from sklearn.manifold import TSNE
#from sklearn.model_selection import LeaveOneOut
#from sklearn.model_selection import train_test_split
#from sklearn.base import BaseEstimator, ClassifierMixin
#from sklearn.neighbors import KernelDensity
#from sklearn.model_selection import GridSearchCV
#from sklearn.svm import LinearSVC
#from scipy.stats import gaussian_kde
#from sklearn.linear_model import LinearRegression
#from sklearn.metrics import confusion_matrix
#from random_walk_classifier import *
#from cell_type_annotation import *
#from IPython.display import clear_output
#from sklearn.model_selection import KFold
#from sklearn.model_selection import StratifiedKFold
'''/////// Function ////////'''

class Sample:
    X = None
    y = None
    z = None
    l = None
    o = None
    def __init__(self, X, y = None, z = None, l = None, o = None):
        self.X = X ## expression (and meta data)
        self.y = y ## orignal index
        self.z = z ## column header (marker and meta data)
        self.l = l
        self.o = o

def method0(handgated,filterCells,normalizeCell,relevantMarkers,cellCount,head,umap):
    ## this function read the handgated file info and load the csv/fcs files from the path given and creates a dic with key as sample_id and value as xpression set and celltypes for each row in expression set
    ct2expr = {}
    md = pa.read_csv(handgated,header=0)
    #md = md.iloc[range(28),:]
    inputs = md.loc[:,"fpath"]
    labels = md.loc[:,"cell_type"]
    ct2indx = {CT:(i+1) for i, CT in enumerate(list(np.unique(labels)))} ## +1 ensures that first celltype will not get 0 as cell typen name ; this is helpful in SVM
    indx2ct = {value:key for key, value in ct2indx.items() } ## get celltype name back from its index
    print ("Reading handgated cells...")
    for c,f in enumerate(inputs):
        if os.path.exists(f):
            match = re.search(r'fcs$', f)
            match2 = re.search(r'csv$', f)
            key=md.loc[c,"sample_id"] ## sampleID
            #cellType = labels[c]
            cellType = ct2indx[labels[c]] ## instead of actual celltypr name use a number to
            if match: ## if file is FCS
                _, exp = fcsparser.parse(f, reformat_meta=True)
                print(key,labels[c])
            elif match2: ## if file is CSV
                exp = pa.read_csv(f, delimiter=",", header=head)
                print(key,labels[c])
            else:
                print(str("Line: " + c + "- Unknown File format, must be CSV or FCS only!!!"))
                sys.exit(0)
            exp = exp.loc[:,relevantMarkers]
            indxrm = None
            if filterCells == True:
                indxrm = filterCells(exp) ## this will give indexes to remove bad quality cell;
                exp.drop(indxrm, axis = 0, inplace=True) ## this will remove the bad quality cells
            if cellType in ct2expr.keys(): ## checking if this celltype has already been used for gathering expression from other sample_id
                exp = pa.concat([ct2expr[cellType].X,exp]) ## concatenate (rbind) with existsing expression dataframe
                ct2expr[cellType] = Sample(X=exp)
            else:
                ct2expr[cellType] = Sample(X=exp)
        else:
            print(str("Error: File " + f + " does not exists"))
            sys.exit(0)
    tmp= {}
    for ct,expr in ct2expr.items():
        if expr.X.shape[0] > cellCount: ## removing celltypes with poor cell count; less equal to than cellCount
            tmp[ct] = expr
    ct2expr = tmp
    if normalizeCell == True:
        ct2expr = preprocess(ct2expr,markers=relevantMarkers) ### FUNCTION NEED TO BE FORMATTED ACCORDING TO ct2expr
    if umap == True:
        plotUMAP(ct2expr,relevantMarkers)
    return Sample(X=ct2expr,y=indx2ct)

def method1(Fileinfo,normalizeCell,relevantMarkers,header):
    ''' This can be used to load panda dataframe 50 times faster'''
    '''https://blog.esciencecenter.nl/irregular-data-in-pandas-using-c-88ce311cb9ef'''
    test = {}
    md = pa.read_csv(Fileinfo,header=0)
    inputs = md.loc[:,"fpath"]
    print ("Reading live cells for annotation...")
    for c,f in enumerate(inputs):
        if os.path.exists(f):
            match = re.search(r'fcs$', f)
            match2 = re.search(r'csv$', f)
            key=md.loc[c,"sample_id"] ## sampleID
            if match: ## if file is FCS
                _, exp = fcsparser.parse(f, reformat_meta=True)
                print(f)
            elif match2: ## if file is CSV
                exp = pa.read_csv(f, delimiter=",", header=header)
                print(f)
            CT=exp.loc[:,'cluster'] ## this is temporary since we have cluster info for test s
            exp = exp.loc[:,relevantMarkers]
            #test[key] = Sample(X=exp)
            test[key] = Sample(X=exp,y=CT) ## this is temporary since we have cluster info for test set
        else:
            print(str("Line: " + c + "- Unknown File format, must be CSV or FCS only!!!"))
            sys.exit(0)
    if normalizeCell == True:
        test = preprocess(test,markers=relevantMarkers)
    return test

def ReduceDFbyLabel(ct2expr,limit):
    print("Reducing expression matrix")
    data = pa.DataFrame()
    labels = pa.DataFrame()
    for ct, V in ct2expr.items():
        Xi = V.X
        print("for CellType : " + str(ct) + "::" + str(Xi.shape[0]) + " Cells")
        if Xi.shape[0] > limit:
            Xi = Xi.sample(n=limit)
        data = pa.concat([data,Xi])
        ytmp = pa.DataFrame([ct] * Xi.shape[0])
        labels = pa.concat([labels,ytmp])
    return Sample(X=data, y=labels)

def plotUMAP(ct2expr,relevantMarkers):
    print("UMAPing")
    dataSample=ReduceDFbyLabel(ct2expr,3000)
    embedding = umap.UMAP(n_neighbors=15, min_dist=.25).fit_transform(dataSample.X.values)
    classes = list(ct2expr.keys())
    ct2idx = {key:idx for idx, key in enumerate(classes)}
    target = np.array([ct2idx[x] for x in dataSample.y.values.ravel()])
    ### Plotting ####
    fig, ax = plt.subplots(1, figsize=(14, 10))
    plt.title('Hangated UMAP')
    plt.scatter(*embedding.T, s=0.1, c=target, cmap='Spectral', alpha=1.0)
    plt.setp(ax, xticks=[], yticks=[])
    cbar = plt.colorbar(boundaries=np.arange(25)-0.5)
    cbar.set_ticks(np.arange(24))

def plotSNE(ct2expr):
    dataSample=ReduceDFbyLabel(ct2expr,3000)
    tsne = TSNE(n_components=2, random_state=0)
    vis_data = tsne.fit_transform(dataSample.X)
    # Visualize the data
    classes = list(ct2expr.keys())
    ct2idx = {key:idx for idx, key in enumerate(classes)}
    target = np.array([ct2idx[x] for x in dataSample.y.values.ravel()])

    ### Plotting ####
    plt.title('Hangated tSNE')
    fig, ax = plt.subplots(1, figsize=(14, 10))
    plt.scatter(*vis_data.T, s=0.1, c=target, cmap='Spectral', alpha=1.0)
    plt.setp(ax, xticks=[], yticks=[])
    cbar = plt.colorbar(boundaries=np.arange(25)-0.5)
    cbar.set_ticks(np.arange(24))

def findrange(data,confidenceInt=0.80):
    n = len(data)
    m = mean(data)
    std_err = sem(data)
    h = std_err * t.ppf((1 + confidenceInt) / 2, n - 1)
    here = m - h
    return abs(here)

def CountFrequency(y):
    freq = {} # Creating an empty dictionary
    for item in y.ravel():
        if (item in freq):
            freq[item] += 1
        else:
            freq[item] = 1
    return freq

def common_member(a, b):
    a_set = set(a)
    b_set = set(b)
    if (a_set & b_set):
        return(a_set & b_set)
    else:
        return("Error: No common elements")

def index2ct(idx2ct,y_pred_index):
    ct_pred = [0] * len(y_pred_index)
    for idx, ct in enumerate(idx2ct):
        #print(str(idx) + " " + str(ct))
        found =  (y_pred_index == idx).tolist()
        res = [i for i, val in enumerate(found) if val]
        for get in res:
            ct_pred[get] = ct
    return ct_pred

def preprocess(train,filterD,transform):
    #exp = pa.DataFrame()
    for key, value in train.items():
        print(key)
        #exp = value.X
        exp = np.arcsinh((value - 1.0)/5.0)
        train[key].X = exp
    return train

def filterCells(exp): ##TO DO LIST : instaed of removing it should just give index to be removed in each matrix
    #for key,value in eset.items():
    #    exp = value.X
    markers  = exp.columns
    print('removing cell events with very high intensity in any given marker')
    to_remove = []
    for i in range(len(markers)):
        print("reading..."+ markers[i])
        marker_expr = exp.loc[:,markers[i]]
        threshold = np.percentile(marker_expr, 99.99) ## change it to 99.99
        cell_index = marker_expr[marker_expr >= threshold].index
        to_remove.append(cell_index.values.tolist())
    merged = list(itertools.chain(*to_remove)) ## reduce list of lists to singel list
    remove = np.unique(merged) ## these cell indeces have atleast one marker that is outlier and these cells will be discarded from analysis
    per_cells_removed = (len(remove) / exp.shape[0]) * 100
    print("Level 1: Cell can be removed " , per_cells_removed, "%(" ,len(remove), ")")
    min_max_scaler = preprocessing.MinMaxScaler() ## scaling column wise expression between 0 to 1
    x_scaled = min_max_scaler.fit_transform(exp)
    x=pa.DataFrame(x_scaled, columns=exp.columns) ## converting numpy array to panda df
    info = x.sum(axis=1) ## sum of each row or cell(info score)
    threshold = 0.05 ## INFOi threshold ## cells with score less than this value have no use
    to_remove = [] ## this list will hold indexes of all those cells that are needed to be removed
    tmp = info[info <= threshold].index.tolist() ## these cells have INFO score very less and these cells contribute nothing so removed
    to_remove.append(tmp)
    df = x.max(axis=1)
    tmp = df[(info * 0.5)< df].index.tolist()
    to_remove.append(tmp)
    merged = list(itertools.chain(*to_remove,remove.tolist())) ## merging the index from level 1 of filtering and level2
    remove = np.unique(merged)
    #exp.drop(remove.tolist(), axis = 0, inplace=True) ## removing the cells from main matrix ## role of scaled matrix is over
    print("Level 2: Cell can be removed " , per_cells_removed, "%(" ,len(remove), ")")
    #eset[key].X = exp
    #return eset ## this is a dict with key as filname and value has sample class. The sample class contains refrence/pointers to the corresponding expression matrix (.X) ; marker columns (.y) and sample info (.z)
    return remove.tolist() ## list of indexes that can be removed

def countEpoch(blockInfo,allowed):
    counts = []
    for key,value in blockInfo.items():
        blocksCount = len(blockInfo[key]) ## blockInfo is hash of hash so its value is also an hash whose length is the
        counts.append(blocksCount)
    epoch = max(counts) * allowed ## each block in the sample with maximum number of blocks must cover 'allowed' iterations to generate reliable probability
    return epoch

def select_block(blockinfo):
    vals = blockinfo.values()
    minVal = min(vals)
    minVals = []
    for k in blockinfo.keys():
        if blockinfo[k] == minVal:
            minVals.append((k, minVal))
        #elif blockinfo[k] == maxVal:
        #    maxVals.append((k, maxVal))
    ind = np.random.choice(range(len(minVals)),1)
    return(minVals[ind[0]][0]) ## return the randonly selected block ID with lowest freq

def getModel(models,counter,ongoing, blocknumber,lda):
    if counter == blocknumber:
        mod = models[ongoing].X
        ct = models[ongoing].y
        counter = 0
        ongoing = ongoing + 1
    else:
        mod = lda
        ct = models[ongoing].y
    return Sample(X=mod,y=counter,z=ongoing,l=ct)

def makePrediction(E,M,lda,CellTypes,postProb):
    X = E.X ## this is the expression matrix
    X = X.loc[:,M] ## M = relevantMarkers
    pred = lda.predict(X)
    pred_prob = lda.predict_proba(X) ## Posterior probability lower threshold, below which the prediction will be 'unknown',
    max_prob = np.amax(pred_prob, axis=1) ## maximum probability associated with each cell (row) for a given cell type among all celltypes
    #indexes = np.argmax(pred_prob, axis=1) ## indexes of the column to which maximum prob is associated; each column ~ a celltype; the cell type with maximum prob is the one we assigned to each cell
    unk_index = np.nonzero(max_prob < postProb) ## getting index of cells when maximum post. prob of cell to be associated with a given celltype is less than threshold
    pred[unk_index] = "Unknown" ## replacing the assigningment of cells as "unknwon" if their post prob. is less than threshold
    unassignedCellsProp  = (np.shape(unk_index)[0]/np.shape(pred)[0])*100 ##%age of cells that are likely to be of unknown category
    ## TO DO: compute the cluster size; average distance; average silhouette ; euclidean distance
    return Sample(X=pred,y=max_prob,z=unassignedCellsProp)

def findcelltype(blck2ct,rowsInfo,epoch,f1):
    indx2ct ={}
    for b in blck2ct.keys(): ## 'b' is the bloc number
        df = blck2ct[b] ## the cell types predicted for each cell in a given block (remember this is running for each file)
        ##df.groupby('a').count() ## counting the number of
        x = epoch - df.apply(pa.value_counts, axis=1).fillna(0) ## for each cell count the number of times this cell type occuered in all possibel cell types i.e. in each row calculate the freq of all known cell types
        x = x/epoch ## This x has the score of each cell in all possible CTs ; score = (epoch - number of times CT found across all epoch) / epoch ; lower the better
        ct = x.idxmin(axis=1) ## extract the name of cell type with min score
        ct_pval = x.min(axis=1) ## extarct the min score
        varianc = df.var(axis=1)
        df = pa.DataFrame({'Celltype':ct,'Minimum_P-value':ct_pval,'Cell type Variance':varianc})
        cell_index = rowsInfo[f1][b]
        indx2ct = dict(zip(cell_index, df)) ## a dict where key is the cell index and value is the 3-column df data frame
    return(Sample(indx2ct))

def printACC(test,OutPut1,OutPut2):
    for sm in test.keys():
        lables = test[sm].y ## orignal labels
        predictedLabels = OutPut2[sm].loc[:,'cluster']
        celltypes = OutPut1[sm].columns.tolist() ## celltypes analyzed
        for ct in celltypes:
            ind1  = list(np.where(lables == ct)[0])
            ind2  = list(np.where(predictedLabels == ct)[0])
            common = list(set(ind1) & set(ind2))
            acc = (len(common)/len(ind1)) * 100
            print ("in sample " + str(sm) + "; accuracy of cell type " + str(ct) + " is " + str(acc) + ' with ' + str(len(ind1)) + ' cells ')

def FindCelltypePerCell(dicti,rd,counter,Cellimit,rowsInfo,Runcollection,result,result2):
    start = 0
    CTs = dicti.X ## get the cell type labels for the dataset in this run
    PostProb = dicti.y ## posterior prob. associated with each cell for the predicted cell type.
    tmp=rd.y ## which block is used in which file during this epoch number
    acc_pred = []
    for f1 in Runcollection.l: ## fileorder: reading the filenames in the same order as per they exists in the expression matrix ; this has file names (see makerun function)
        print(f1)
        df1 = pa.DataFrame()
        if f1 in result.keys(): ## if this file f1 has already been processd in someother epoch number
            blck2ct = result[f1] ## read the block number of each file: key2 and gives the cell types of all cells if block already run under any epoch(s) before
            blck2pp = result2[f1] ## read the block number of each file: key2 and gives the posterior prob. if block already run under any epoch(s) before
        else:
            blck2ct = {}
            blck2pp = {}
        block = tmp[f1] ## for this file whats the block used
        itsIndecies = rowsInfo[f1][block] ## rowinfo will tell what are the indexes available in this block
        end = start + len(itsIndecies)
        region = range(start,end)
        en = ('e'+str(counter)) ## epochnumber
        ct = pa.DataFrame({en:CTs[region]}) ## a dataframe in which column name is the epoch number (e1, e2..en) and value is the cell type predicted
        pp = pa.DataFrame({en:PostProb[region]})
        cellsFound = ct.shape[0]
        ct[en] = ct[en].astype('category') ## converting the cell type to categorical data
        print(str(CTs.shape[0]) + " / " + str(start) + " / " + str(end)  + " / " + str(cellsFound))
        if block in blck2ct.keys(): ## if blck2ct[block] exists then update the vlues else
            df1 = blck2ct[block]
            blck2ct[block] = pa.concat([df1, ct], axis=1, sort=False) ## keep on appending the df cell types (column wise) according to the order of cells used in expression matrix
            ref = blck2ct[block].iloc[:,0]
            acc_pred.append(accuracy_score(ref, ct)) ## calculate accuracy at this point and keeping first column as ref and see the fluction of accuracy after all runs
            df1 = blck2pp[block]
            blck2pp[block] = pa.concat([df1, pp], axis=1, sort=False)
        else:
            blck2ct[block] = ct ## value (cell type) is the panda datafarame
            blck2pp[block] = pp
        result[f1]=blck2ct ##result{filename}{blocknumber}:celltye labels
        result2[f1]=blck2pp ##result2{filename}{blocknumber}:celltye labels post. prob.
        start = end
        if cellsFound != Cellimit:
            print ("Error 1: Row count doesnt match. Dont know what to do :( \n" +
            "epoch: " + str(counter) + "\n" +
            "expected rows: "+ str(Cellimit) + "\n" +
            "Found rows:" + str(np.shape(ct)[0]) + "\n")
            #exit()
    acc_overall = sum(acc_pred)/len(acc_pred) ## average accuracy per  block in each run
    return Sample(X=result,y=result2,z=acc_overall)

#def analyzeResult(labels,prob):
#    prob_thresh = 0.7 ## across iterations if there are no celltype that occured less than this %age of value will be marked as Uknown
#    output = {}
#    unstable = []
#    for f1 in labels.items():
#        output[f1] = {}
#        for b1 in labels[f1].items():
#            tmp={}
#            df = labels[f1][b1] ## a panda categorical dataframe of celltypes lables predicted in each epoch
#            rowDesc = df.apply(pa.Series.describe, axis=1) ## col1: count ; col2: unique ; col3: top ; col4: freq (of toptet.loc cel type)
#            df["prob"]= rowDesc.iloc[:,3]/rowDesc.iloc[:,0] ##3 is freq and 0 is tot number of epoch runs occured in this block
#            rowDesc.loc[rowDesc.prob < prob_thresh, 'top']  = "Unknown" ## finally marking unknown cell types in which none of the cell-type is dominating
#            tmp[b1] = rowDesc.top
#            output[f1][b1] = tmp ## Now top column has unknown if the cell type is unstable else the stable cell type (from describe())
#            # df.top has all the updated labels tet
#            unstable.append(df[rowDesc.unique > 1]) ## these cells/rows have more than one cell types; lets connect them; some of them may have significantly enriched one cell type only
#    #ct_net = create_Cell_Type_Network(unstable) ## will give the edge list with score and a network plot in PNG format
#    #return Sample(X=output,y=ct_net)


def runPCA(X):
    Xi = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2) ## since these are pre-sorted cells (hand-gated), we donot expect much variance
    X_pca = pca.fit_transform(Xi)
    return X_pca ## check for laoding

def makeBlocks(cells,limit,expr):
    blockscore = {}
    block = {}
    items = range(cells)
    if items == limit: ## for this file only one block can be made
        blockNumber =  ('block' + str(1))
        block[blockNumber] = [items]
        blockscore[blockNumber] = 0
    else : ## basicially it will keep on randomly collecting index from the file and start assigning them to block; if indexes are over and a block has remianed empty by > 1 cell, them remaining cell was assigned from any other block randomly
        count = int(cells/limit) + 1## total number of blocks to be made for ongoing file
        #items_copy = items.copy() ## making a copy for later use
        for i in range(count):  ## +1 beacuse we used int
            blockNumber =  ('block' + str(i))
            if len(items) < limit: ## if you have less number of cells available in dataset for this block than required; anyways we will fill this block to desired number of cells by randomly selecting the cells
                blck_indx = np.random.choice(items,len(items), replace=False)
                req = limit - len(blck_indx) ##if in the final block the number of cells are less than limits than add required number of cells ##
                subse = list(set(list(range(cells)))^set(list(blck_indx))) ## small subset of indexes from which remaining cells need to be identified randomly
                blck_indx = np.append(blck_indx, np.random.choice(subse,req, replace=False)) ## randomly capturing required number of cells from subset of cells indexes
            else :
                blck_indx = np.random.choice(items,limit, replace=False)
            exprlist = expr.X.iloc[blck_indx,:].values.tolist() ## this is the expression matrix made up of the indexes of the given block 'blockNumber' for the ongoing file
            block[blockNumber] = Sample(X=exprlist,y=blck_indx) ## updating block dict for the expression matrix and indexes that becomes the part of this block for the ongoing file
            blockscore[blockNumber] = 0 ## initiating frequency: number of times thois block has been used during iteration; this dict is imp and will keep on updating itself during the process
            items = list(set(items)^set(blck_indx)) ## now items hold only those indexes not used before
    blocks = Sample(block, blockscore)
    return blocks

def getBlockNum(blockInfo):
    blcounts=[]
    for samples in blockInfo:
        blcounts.append(len(blockInfo[samples]))
    blcounts.sort()
    return(blcounts[0])

def getSet(ct2expr):
    mydf = pa.DataFrame()
    annota = pa.DataFrame()
    for key, value in ct2expr.items():
        print(key)
        celltypes= pa.DataFrame([key] * value.X.shape[0])
        annota= pa.concat([annota,celltypes])
        mydf = pa.concat([mydf,value.X])
    out = Sample(X=mydf,y=annota)
    return out

def performLDA(ExprSet,relevantMarkers,shrinkage,nlandMarks,LandmarkPlots,ct2expr,indx2ct):
    #counter = 1
    #skf = StratifiedKFold(n_splits=2, shuffle = True, random_state=random.randint(1,100000))
    print("Q/LDA: Splitting into train (80%) and test set (20%)")
    skf = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=random.randint(1,100000))
    model={}
    # X is the feature set and y is the target
    for train_index, test_index in skf.split(ExprSet.X,ExprSet.y):
        X_train = ExprSet.X.iloc[train_index,:]
        X_test = ExprSet.X.iloc[test_index,:]
        y_train = ExprSet.y.iloc[train_index,:].values.ravel()
        y_test = ExprSet.y.iloc[test_index,:].values.ravel()
        print("Fitting QDA")
        qda = QDA(store_covariance=True)
        qda.fit(X_train, y_train)
        classes = qda.classes_ ## this is the list of cell types
        #y_train_pred = qda.predict(X_train) ## train
        y_test_pred = qda.predict(X_test)
        #accu1 = accuracy_score(y_train_pred, y_train) ## train
        accu1 = accuracy_score(y_test_pred, y_test)
        accu11 = f1_score(y_test, y_test_pred, average='weighted')
        prob = qda.predict_proba(X_test)
        y_test_Postprob = pa.DataFrame(prob, columns=classes)
        #CellTypeAcc(y_train_pred,y_train,"QDA", None)
        Dplot = CellTypeAcc(y_test_pred,y_test,"QDA",y_test_Postprob,indx2ct)
        #model[accu1] = Sample(X=qda, Y=)
        model[accu11] = Sample(X=qda,y=Dplot)
        #qda.predict_proba()
        print('QDA Model: ' + ' // Accuracy : {:.2f}'.format(accu1) + ' // Accuracy : {:.2f}'.format(accu11))
        print("Fitting LDA")
        if shrinkage== None:
            lda = LDA(n_components = len(relevantMarkers) -1)
        else:
            lda = LDA(solver='lsqr', shrinkage='auto')
        lda.fit(X_train, y_train)
        #y_train_pred = lda.predict(X_train)
        y_test_pred = lda.predict(X_test)
        y_test_Postprob = pa.DataFrame(lda.predict_proba(X_test), columns=classes)
        #accu2 = accuracy_score(y_train_pred, y_train)
        accu2 = accuracy_score(y_test_pred, y_test)
        accu22 = f1_score(y_test, y_test_pred, average='weighted')
        #CellTypeAcc(y_train_pred,y_train,"LDA",None)
        Dplot = CellTypeAcc(y_test_pred,y_test,"LDA",y_test_Postprob,indx2ct)
        print('LDA Model: ' + ' // Accuracy : {:.2f}'.format(accu2) + ' // F1-score : {:.2f}'.format(accu22))
        #model[accu2] = Sample(X=lda) ## enable this if you want train prediction to be also used for final model selection
        model[accu22] = Sample(X=lda, y=Dplot) ##<========
        #counter = counter + 1
        clf = GradientBoostingClassifier(n_estimators=130, learning_rate=0.1, max_depth=2, random_state=0).fit(X_train, y_train)
        y_test_pred = clf.predict(X_test)
        y_test_Postprob = pa.DataFrame(clf.predict_proba(X_test), columns=classes)
        #accu2 = accuracy_score(y_train_pred, y_train)
        accu2 = accuracy_score(y_test_pred, y_test)
        accu22 = f1_score(y_test, y_test_pred, average='weighted')
        #CellTypeAcc(y_train_pred,y_train,"LDA",None)
        Dplot = CellTypeAcc(y_test_pred,y_test,"LDA",y_test_Postprob,indx2ct)
        print('GB Model: ' + ' // Accuracy : {:.2f}'.format(accu2) + ' // F1-score : {:.2f}'.format(accu22))
        lmarks = findLandmarks(ct2expr,X_test,y_test,classes,nlandMarks,LandmarkPlots,relevantMarkers,indx2ct)
    scores = list(model.keys()) ## getting all the accuracu during LDA/QDA modelling
    scores.sort(reverse = True) ## soring the accuracy in decending order
    mod= model[scores[0]].X ## model with highest accuracy either LDA or QDA
    thresholds = model[scores[0]].y ## celltype specific threshold (from TR prediction of test set) from the selected model mod
    X_train=None; X_test=None; y_test_pred=None; y_test_Postprob=None;ExprSet=None; prob=None;scores=None;
    return Sample(X=mod,y=thresholds,z=lmarks,l=lmarks)

def fitGMM(ct2expr,Maxlimit,totalCells,indx2ct): ## this will create a separate 2-component GMM model for each cell type; this will help us in shortlisting bad neighbours preicted by faiss
    #Maxlimit = 100000 ## Maximum number of cells to make GMM fitting per cell type
    models = {}
    freqCT = {}
    print ("Fitting GMM and finding Landmarks (may be time consuming for some cell types) ")
    start_time = time.time()
    for cellType, V in ct2expr.items():
        Xi=V.X
        ct = indx2ct[cellType]
        print("for CellType : " + str(ct) + "::" + str(Xi.shape[0]) + " Cells")
        if Xi.shape[0] > Maxlimit:
            Xi = Xi.sample(n=Maxlimit)
        gmm = GMM(n_components=2).fit(Xi)
        models[cellType] = gmm
        freqCT[cellType] = V.X.shape[0]/totalCells
    print("--- %s seconds used ---" % (time.time() - start_time))
    return Sample(X=models,y = freqCT)

def makerun2(rowsInfo,blockInfo,relevantMarkers):
    runInfo = {}
    #expr = np.empty((0, dim))
    expr=[]
    orig = []
    sampl = []
    #annotation = pa.DataFrame() ## tmp varibale ; will be removed
    for f1 in rowsInfo.keys(): ## read each file one by one
        #print("fILENAME..",f1)
        blockinfo = blockInfo[f1] ## get the frequency info of each block of cells. this stores the information of how many times a given block has been used for analysis
        #selected_block1='block0'
        selected_block1 = select_block(blockinfo) ## this will give the block ID with minimum frequency (randomly chosen block)
        blockinfo[selected_block1] = blockinfo[selected_block1] + 1 ## update the frequency of block once its used so that other block woulg get preference
        blockInfo[f1] = blockinfo # update main dict
        indeces = rowsInfo[f1][selected_block1].y ## get the index of these cells in a given file f1 ## there can only be one block froma file so no worry about the order until filename order is intact
        runInfo[f1] = selected_block1 ## update which block ID was used for which file in this run
        expr += rowsInfo[f1][selected_block1].X ## a list of expression values from the selected block of this file
        orig += list(indeces) ## its orignal indexes
        sampl += ([f1] * len(indeces)) ## a list of samplenames with respect to the indexes
    IndexDict = dict(zip(range(len(sampl)),orig))
    sampleDict = dict(zip(range(len(sampl)),sampl))
    ## the speed can further be increased if somehow I can resolve X=pa.DataFrame(expr,columns=relevantMarkers) with some alternative
    runs = Sample(X=expr,y=runInfo,z=blockInfo,l=IndexDict,o=sampleDict) ## sample class object;
    return runs

def farthest_search(df, top):
    out = []
    print(".", end = '')
    #print(df)
    for _ in range(top-1):
        dist_mat = spatial.distance_matrix(df.values, df.values)         # get distances between each pair of candidate points
        i, j = np.unravel_index(dist_mat.argmax(), dist_mat.shape)         # get indices of candidates that are furthest apart
        z = [i] + [j] ## these two indexes are fathest from each other
        out += list(df.index[z]) ## getiing the orignal rowname/index of those indexes found to be farthest from teach other in this run
        df.drop(df.index[z], inplace=True)
    return out

def getCenter(data,name,nlandMarks,plot):
    data = pa.DataFrame(data)
    warnings.filterwarnings("ignore")
    myPDF,axes,z = fast_kde(data.iloc[:,0],data.iloc[:,1],sample=True) ## the modified KDE downloaded from https://gist.github.com/joferkington/d95101a61a02e0ba63e5
    z = pa.DataFrame(stats.zscore(z))
    quants = [i for i in np.arange(1.0, 0.1, -0.1)] ###  [0.99,0.89,0.79,0.69,0.59,0.49,0.39,0.29,0.19] ## qualtile ranges
    threshs = [np.quantile(z, q) for q in quants] ## maximum value of Kernel density within each qantile range
    desiredIndex = []
    ## Idea was, we split the kernel densitites into 10 quantiles; each quantile will contribute to 'defined' number of cells
    ## each quantile has its own density threshold; withoin which randomly defined number of cells will be picked
    print("optimizing landmarks")
    density = [] ## ******
    for th in threshs:
        LandMarkcell = list(z[z[0] >= round(th, 3)].index)
        dens = z[z[0] >= th]
        z.drop(LandMarkcell,inplace=True)
        if len(LandMarkcell) > nlandMarks:
            top = int(nlandMarks/2)
        else:
            top=1
        if top > 1:
            i = farthest_search(data.loc[LandMarkcell,:], top) ## among cells with almost equal densities (same quantile); retain only cells far from each other in 2D space
        else:
            i = LandMarkcell
        density = np.append(density, dens.loc[i,:]) ## ****** ## the index i belongs to the index of orignal panda dataframe i.e. data
        desiredIndex  = desiredIndex + i
    if plot == True:
        test0 = pa.DataFrame({"x":data.iloc[:,0],"y":data.iloc[:,1],"z":dens})
        #s = np.zeros(test0.shape[0]) ## this specifically highlights (VISUALLY) landmark cells in density plot. here we create a vector of zeros
        #np.put(s,LandMarkcell,[50]) ## replace the landmark cell index with size 50, that will be brush size in scatter plot below, so only landmark cells will be plotted rest will be hidden keeping the plot scale intact
        #ax=test.plot.scatter(x='x',y='y',c='z',s=s,colormap='viridis')
        ax=test0.plot.scatter(x='x',y='y',c='z',colormap='viridis') ## if you are running the above three lines; hash this line
        plt = ax.get_figure()
        plt.savefig(name)
    return Sample(X=desiredIndex,y=density) ## ******


def findLandmarks(X_train,y_train,X_test,y_test,CTs,nlandMarks,LandmarkPlots,relevantMarkers,indx2ct):
    ## Now also identify land marks for this cell type
    nlandMarks=6 ## EVEN NUMBER ONLY #### from each quantile out of 10 quantiles of kernel densitites (PC1 vs PC2) pick this much number of cells
    d = len(relevantMarkers) # number of markers
    np.random.seed(12)
    index = faiss.IndexFlatL2(d) ## initiating faiss database
    xb = np.ascontiguousarray(np.float32(X_test.values)) ## making faiss database of epoch-specific marker expression matrix
    index.add(xb) ## indexing the database
    landmarks = {}
    Ddensity = {}
    name = None
    for cellType in indx2ct.keys():
        Xi = X_train.values[y_train==cellType]
        name =  "PCA_" + str(cellType) + ".pdf"
        print(name)
        ct = indx2ct[cellType]
        data = runPCA(Xi) ## applying PCA on cell matrix for this cell type
        LMind = getCenter(data=data,name=name,nlandMarks=nlandMarks,plot=LandmarkPlots) ##LandmarkPlots: True or False if you want to have scatter plot of PCs with dense clsuetrs
        Testcellindex=np.where(y_test==cellType)[0] ## index of this cellype in this test dataset
        neighbours = len(Testcellindex)
        cells_DF=Xi[LMind.X,:] ## expression matrix of LM cells only ## ******
        xq = np.ascontiguousarray(np.float32(cells_DF))
        D, I = index.search(xq, neighbours) ##  ## get neighbours of all the landmarks cells for a given cell type ## I variable has rowindexes of cells that should belong to same distribution to the
        comparison={indx:None for indx in range(len(I))} ## an empty dict
        for indx in range(len(I)): ## read all the preicted neighbours of every LM one by one; reading the output of landmarks one by one
            TruePos = set(I[indx]) & set(Testcellindex) ## common indxes between predicted indexes for this LM and orignal TP indexes
            if (len(TruePos)/len(Testcellindex)) >= 0.20: ### atleast 20% of indexes are TP; otherwise I dont need that LM
                comparison[indx] = TruePos
        ### now that we know which LM has predicted what TP indexes; we will now try to find minimum number of LM required to effectively predict more than 98% of TP indexes in test datasets
        found={}## for each of celltype index found ; how many LMs can find  this index
        rows = []
        densities = [] ## ******
        for indx in range(len(I)-1):
            TP1 = comparison[indx] ## output of this landmark prediction; by deafult the very first index will always be used; no matter waht; its the most dense LM
            if TP1 is not None:
                #choice = [checkLM(TP1,comparison[indx2]) for indx2 in range(indx+1, len(I))] ##***** ## indexes that are TP predict by subsequent landmark
                comparison = {indx2:checkLM(TP1,comparison[indx2]) for indx2 in range(indx+1, len(I))} ## if the first TP1 is better than anyone of the subsequenct LM than replace that subsequent LM value with None; else keep it as such
                #print(sum([x is None for x in comparison.values()])) ## counting if None count is increaing over indx iterations
                #if sum(choice) == 0: ## hurray this TP1 is better than all other TP2
                rows = rows + [cells_DF[indx].tolist()] ## append the marker expression of this LM as list
                densities = densities + [LMind.y[indx]] ## ******
                found.update({tp:found.get(tp, 0) + 1 for tp in TP1})
            if (len(found.keys())/neighbours) > 0.99: ## the current landmark has already predicted more than 98% of expected indexes
                break ## if all the expected neigbours are already predicted; do not go ahead
        acc = (len(found.keys())/neighbours) * 100
        print('Achieved ' + str(acc) + '99% of cells in CELLTYPE ' + str(ct) + ' with ' + str(len(rows)) + ' / ' + str(len(LMind.X)) +  ' landmark cells')
        landmarks[cellType] = pa.DataFrame(rows, columns=X_train.columns)
        Ddensity[cellType] = densities ## list of LM densities (for each LM cell) in each cell type
        indexes = []
        for s in range(len(rows)):
            indexes.append("LM" + str(s)) ## adding rownames to landmark cells to probe them later ,e.g. LM0; LM1;LM2... dependng upon number of landmark cells
        landmarks[cellType].index = indexes
    return Sample(X=landmarks,y= Ddensity) ## a dict of cell type and best landmarks (panda df) with expression values

def checkLM(TP1,TP2):
    ##there are 2 criteria to call TP1 good than TP2
    ret = TP2 ## by deafult do not remove/change TP2 from dict
    if TP2 == None:
        ret= None ## already None; so let it be None; TP1 is good; TP2 will be automatically ignored from dict
    else:
        common  = set(TP1) & set(TP2) ## common True Postives predicted by these two LMs
        if len(common) / len(TP2) >= 0.90:  ## more than 90% of common TP exits within TP1; why would i NEED TP2 THEN; its a repetitive landmark. i dont need it
            ret = None ## REMOVING ITS PREDICTIONS ; so that this index (TP2 index) will be ignored for subsequent analysis; whereas TP1 will be used
    return ret


def svc_param_selection(X, y, nfolds,ct):
    Cs = [0.1]
    gammas = [0.001]
    print ("Searching SVM hyper-parameters for cell type", str(ct))
    param_grid = {'C': Cs, 'gamma' : gammas}
    grid_search = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=nfolds)
    grid_search.fit(X.iloc[1:5000,:], y[1:5000])
    print('Best score for ', str(ct) , ' : ' , grid_search.best_score_)
    return grid_search.best_estimator_

def QDAboost(X_train,y_train,X_test,y_test,ct,obj0,CTin):
    table = {}
    qda = QDA(store_covariance=True)
    qda.fit(X_train, y_train)
    y_test_pred = qda.predict(X_test)
    accu0 = accuracy_score(y_test_pred, y_test)
    accu00 = f1_score(y_test, y_test_pred, average='weighted')
    print('QDA Model for cell type: ' + str(ct) + ' // Accuracy : {:.2f}'.format(accu0) + ' // F1-score : {:.2f}'.format(accu00))
    clf = GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, max_depth=2, random_state=0).fit(X_train, y_train)
    y_test_pred = clf.predict(X_test)
    accu1 = accuracy_score(y_test_pred, y_test)
    accu11 = f1_score(y_test, y_test_pred, average='weighted')
    print('GradientBoosting for cell type: ' + str(ct) + ' // Accuracy : {:.2f}'.format(accu1) + ' // F1-score : {:.2f}'.format(accu11))
    y_test_pred = obj0.X.predict(X_test)
    y_test_pred = [CTin if x==CTin else 0 for x in y_test_pred]
    accu2 = accuracy_score(y_test_pred, y_test)
    accu22 = f1_score(y_test, y_test_pred, average='weighted')
    print('QDA (all vs all)  for cell type: ' + str(ct) + ' // Accuracy : {:.2f}'.format(accu2) + ' // F1-score : {:.2f}'.format(accu22))
    #accu2 = obj0.y[CTin] ## accuracy of this cell type in all vs all model
    #print('QDA (all vs all) for cell type: ' + str(ct) + ' // Accuracy : {:.2f}'.format(accu2) )
    table[accu0] = qda
    table[accu1] = clf
    table[accu2] = obj0.X
    bestAcc = max(list(table.keys())) ## model with maximumm accuracy
    model = table[bestAcc]
    return model

def CellTypeAcc(y_test_pred,y_test,text,y_test_Postprob,indx2ct):
    ## read all celltypes one by one ## get it from unique
    y_pred = np.array(y_test_pred) ## predicted labels
    y_test = np.array(y_test) ## orignal labels
    mlist = y_test_Postprob.columns.values
    Dplot={}
    mod = {}
    for ct in list(mlist):
        probs = pa.DataFrame(y_test_Postprob.loc[:,ct])
        ind1 = np.where(y_pred == ct)[0] ## predicted lables
        ind2 = np.where(y_test == ct)[0] ## orignal labels for this ct
        common = list(set(ind1).intersection(ind2)) ## true positive prediction; the indexes that are predicted to be this cell type (in y_pred) are also the same cell type in y_test
        probs = probs.iloc[common,:].values## prosterior probs observed in TP pedictions by LDA or QDA
        errorRate = (len(ind1) - len(common)) / len(ind1)
        Dplot[ct] = {'thesh':np.quantile(probs, 0.01),
             'minimum': np.amin(probs),
             'maximum': np.amax(probs),
             'Mean':np.mean(probs),
             'errorRate': errorRate}
        accu = len(common) / len(ind2) ## found among all the expected lables
        cELLt = indx2ct[ct]
        mod[ct] = accu
        print ('Acc for celltype : ' + str(cELLt) + ' with ' + str(len(ind2)) + ' cells is : {:.2f}'.format(accu) + '; Error rate: ' + str(errorRate) + '; method: ' + text)
        print ('###############################################################################')
    threshold = pa.DataFrame(Dplot).T
    threshold.plot.line()
    print(threshold)
    return mod

def AllvsAllQDA(X_train, y_train,X_test,y_test,indx2ct):
    qda = QDA(store_covariance=True)
    qda.fit(X_train, y_train)
    classes = qda.classes_ ## this is the list of cell types
    y_test_pred = qda.predict(X_test)
    accu1 = accuracy_score(y_test_pred, y_test)
    accu11 = f1_score(y_test, y_test_pred, average='weighted')
    prob = qda.predict_proba(X_test)
    y_test_Postprob = pa.DataFrame(prob, columns=classes)
    mod = CellTypeAcc(y_test_pred,y_test,"QDA",y_test_Postprob,indx2ct)
    print('QDA Model: ' + ' // Accuracy : {:.2f}'.format(accu1) + ' // Accuracy : {:.2f}'.format(accu11))
    return Sample(X=qda,y=mod)

def trainHandgated(ExprSet,ct2expr, relevantMarkers,CTs,nlandMarks,LandmarkPlots,indx2ct):
    print("Splitting into train (60%) and test set (40%)")
    skf = StratifiedShuffleSplit(n_splits=1, test_size=0.4, random_state=random.randint(1,100000))
    models={}
    freqCT={}
    # X is the feature set and y is the target
    for train_index, test_index in skf.split(ExprSet.X,ExprSet.y):
        X_train = ExprSet.X.iloc[train_index,:]
        X_test = ExprSet.X.iloc[test_index,:]
        y_train = ExprSet.y.iloc[train_index,:].values.ravel()
        y_test = ExprSet.y.iloc[test_index,:].values.ravel()
    obj0 = AllvsAllQDA(X_train, y_train,X_test,y_test,indx2ct)
    obj = findLandmarks(X_train,y_train,X_test,y_test,CTs,nlandMarks,LandmarkPlots,relevantMarkers,indx2ct) ### identify the best landmarks with non-redundant outcome and good density
    lmarks = obj.X
    ## Next step is to create celltype specific model
    d = len(relevantMarkers) # number of markers
    np.random.seed(12)
    index = faiss.IndexFlatL2(d) ## initiating faiss database
    xb = np.ascontiguousarray(np.float32(ExprSet.X.values)) ## making faiss database of epoch-specific marker expression matrix
    index.add(xb) ## indexing the database
    for CT in ct2expr.keys():
        neighbours = ct2expr[CT].X.shape[0]
        LMs = lmarks[CT]
        xq = np.ascontiguousarray(np.float32(LMs.values))
        D, I = index.search(xq, neighbours) ##  ## get neighbours of all the landmarks cells for a given cell type ## I variable has rowindexes of cells that should belong to same distribution to the
        indexes = I.ravel()
        ## label the indexes ## this will include cell types from multiple CTs with non-linear boundaries. SVM can help
        X = ExprSet.X.iloc[indexes,:]
        Y = ExprSet.y.iloc[indexes,:]
        Y = Y.applymap(lambda x: 0 if x != CT else x) # IN labels all celtypes other than this will be zero
        if X.shape[0] > 150000: ## TOO MANY CELLS ; REDUCE THE NUMBER SO THAT TRAINING SET GET 100K CELLS
            X = X.sample(150000)
            Y = Y.iloc[X.index,:]
        if len(np.unique(Y)) > 1:
            X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=1)
            models[CT] = QDAboost(X_train,y_train,X_test,y_test,indx2ct[CT], obj0,CT) ## output is a sample obj
        else: ## Faiss accurately predicted only one type of cell ; thats like impossible but lets just assume it happened
            gmm = GMM(n_components=1).fit(X)
            models[CT] = gmm
        freqCT[CT] = ct2expr[CT].X.shape[0]/ExprSet.X.shape[0]
    return Sample(X=obj, y=models, z=freqCT) ## obj.y is the densities of LM cells in obj.X

def CelltypePred(iter1,Runcollection,SelfObject,relevantMarkers):
    X = np.array(Runcollection.X)## a marker expression matrix; epoch specific; made by comboining one block from each cell
    ensemble_ = SelfObject.y ## Q/LDA model with highest F1 score
    obj = SelfObject.X ## a dict; chosen landmark cells (with expression values) for each cell type
    landmarks = obj.X
    densities = obj.y
    freqCT = SelfObject.z ##  frequency/proportion of a given cell type in hand-gated cells
    d = len(relevantMarkers) # number of markers
    np.random.seed(1234)
    index = faiss.IndexFlatL2(d) ## initiating faiss database
    xb = np.ascontiguousarray(np.float32(X)) ## making faiss database of epoch-specific marker expression matrix
    index.add(xb) ## indexing the database
    #m = {ct :{} for ct in freqCT.keys()} ## initializing dict of dict
    check ={}; updatedCT ={}; vari={}
    for CT, EN in freqCT.items(): ## this dict has cell-type as key and expected prop. as value  ##EN : number of neighbours you need; this can be the number of cells expected for a given cell type ## count the number of cells supposed to be there in the dataset ##
        #print("*", end='')
        ################## Nearest Neighbourhood approximation #############################
        neighbours = int(EN * len(X))
        xq = np.ascontiguousarray(np.float32(landmarks[CT].values)) ## getting from the dict ; should be panda data frame of landmark cells with column in same order as orignal expr dataframe ## for each celltype get the landmarks cells (as expression values of all markers from 'eset') ##
        D, I = index.search(xq, neighbours) ##  ## get neighbours of all the landmarks cells for a given cell type ## I variable has rowindexes of cells that should belong to same distribution to the
        indexes = list(np.unique(I.ravel())) ## remove the duplicate indexes predicted to be neighbours of all landmark cells
        tmpdf = np.concatenate((landmarks[CT].values,X[indexes]), axis=0) ## the first n cells in this case is landmark cell which is also used in modelling ## first cell is also the cell closest to the query cell in faiss
        ################## weights for each cell using LM cells #########
        d = {};
        for i,col in enumerate(I):
            for j,k in enumerate(col):
                weight = -1 * (np.log(((j+1)/(neighbours+2)))) * densities[CT][i] # <-------
                d.update({k:d.get(k, 0) + weight})
        ##################### scoring: weighted ensemble model ##########
        tmpdf = np.delete(tmpdf, slice(landmarks[CT].shape[0]), 0) ## removing landmark cells from the cell-type specific exp matrix
        pred  = ensemble_[CT].predict_proba(tmpdf) ## prob. prob to cells using cell type specific model
        column = np.argmax(ensemble_[CT].classes_) ## colun number of cell type in the model
        tmp1 = list(pred[:,column]) ## posterior prob. of cell to belong to this cell type using best classifier
        ################## Converting CT-specific ID to epoch-specific ID ##########
        score = {cid:np.nansum([d[cid] ,tmp1[ti]]) for ti,cid in enumerate(indexes)} ## note nansum is used; so if one vale is nan then it wont be a problem
        for ind in indexes:
            if ind in check:
                if check[ind] < score[ind]:
                    check[ind] = score[ind] ## for testing the condition & getting the score
                    updatedCT[ind] = CT ## will yeild final CT of a given index
                    vari[ind] += [CT] ## store all CT which is related with given index
            else:
                check[ind] = score[ind]
                updatedCT[ind] = CT
                vari[ind] =[CT]
    m = {ai:[check[ai],updatedCT[ai],vari[ai]] for ai in check.keys()} ##allindexes
    return m

def e2b(test, train,limit, blockNum, smallSample, relevantMarkers, shrinkage, Maxlimit,nlandMarks,allowed,epoch,LandmarkPlots,herenthersh):
    rowsInfo = {}
    blockInfo = {}
    print("Processing blocks within each file... this may take some time")
    for key,value in test.items(): ## key is filename
        #ExpressionSet[key] = value.X.values ## updated in main dict with key as filename
        cells = value.X.shape[0] # number of cells
        print(cells)
        if limit == "auto":
            limit = int(cells/blockNum) ## number of cells to be used in all the blocks in this FCS file; This number of cell will vary across all the FCS files
        elif cells < limit:
            if smallSample:  ## if smallSample == true then we will not ignore this file; rather all cells are assigned to 1 block
                limit=cells ## the total size of block in this file is equal to the number of cells available
            else:
                sys.exit('Error: Set limit == "auto" or Please choose less or equal number of cell limit in a block than total cells available in dataset:'+ cells + " or set smallSample=False" )
        blocks = makeBlocks(cells,limit,test[key]) ## limit is the number of cells you will have in each block; cells is the total number of cells in a given file (key) and last one is the expression matrix of this file in self object
        rowsInfo[key] = blocks.X ## for the given file 'key', here are the block numbers and respective index
        blockInfo[key] = blocks.y ## this would be only 0 at this point but later gets updated during randomization of blocks duting epochs
    blockNum=getBlockNum(blockInfo)
    ## running LDA/QDA and getting the best model on the basis of accuracy
    ExprSet = getSet(train.X) ## this training data is handgated cells which will also split futher into train and test set
    indx2ct = train.y ## key is the number and value is the cell type name
    CTs  = list(indx2ct.values())
    SelfObject = trainHandgated(ExprSet,train.X,relevantMarkers,CTs,nlandMarks,LandmarkPlots,indx2ct)
    print('Counting epochs')
    if epoch == 'auto':
        epoch = countEpoch(blockInfo,allowed)
    Sam2CT={}
    for sm in test.keys():
        Sam2CT[sm] = {}
    #mydict = Sam2CT.copy()
    for iter1 in range(epoch):
        Runcollection=None;
        start_time = time.time()
        print("Running epoch..",iter1)
        Runcollection=makerun2(rowsInfo,blockInfo,relevantMarkers) ## for a given epoch what is the final expression df (panda) [.X] ; what is the final runInformation i.e. which file used which block) [.y] and ; updated block information i.e. how many times the each block has been used across different epochs (.z)
        tmp = CelltypePred(iter1,Runcollection,SelfObject,relevantMarkers)
        Sam2CT = processArray(tmp,Sam2CT,Runcollection.o,Runcollection.l)
        print("Completed --- %s seconds ---" % (time.time() - start_time))
    print("analyzing...")
    output=report(Sam2CT,indx2ct,CTs,test,herenthersh)
    return output

def processArray(tmp,Sam2CT,Idx2sample,Idx2rowNum):
    for id1,data in tmp.items():
        sm = Idx2sample[id1]
        oid  = Idx2rowNum[id1]
        Sam2CT[sm].update({oid:Sam2CT[sm].get(oid, []) + [data]})
    return Sam2CT

def report(Sam2CT,indx2ct,CTs,test,herenthersh):
    ## Converting count values to probability ##
    herenthersh = 4 ## randomly moving cells; cells moving into more than this number of cell types will be classififed as here n there cells
    OutPut1={}
    OutPut2={}
    #template = pa.DataFrame([[0] * len(CTs)], columns=indx2ct.keys())
    for sm in Sam2CT.keys(): ## reading samples one by one from template (Sam2CT)
        print("Putting labels sample..." + sm)
        prob= [] ## this will be a 2D list one for each sample; rows will be cell and columns will be celltype and values will be score of each cell across all epochs in a given CT
        Evalue= [] ## one credibility score for each cell
        UNK=[] ## indexes of cells in this sample which fails to annotate
        HereNthere = [] ## cells moving to different cell types across all epochs in more than defined number of CTs
        for indx in range(test[sm].X.shape[0]):
            tmpCT={}; scr=[]
            if indx in Sam2CT[sm]: ## found in some cluster
                data = np.array(Sam2CT[sm][indx]) ## [1] score [2] CT/per epoch [3] all CTs predicted for this index in every epoch
                b = data[:,1]; bb = data[:,0]
                ## variability of index ##
                foundCTs = np.concatenate(data[:,2], axis=0 ) ## all CTs predicted for this index in every epoch; will help in calculating credibility
                Evalue.append(np.var(foundCTs)) #variance = np.var(foundCTs)
                if len(list(np.unique(foundCTs))) > herenthersh:
                    HereNthere.append(indx)
                for k in range(len(b)):
                    if b[k] in tmpCT:
                        tmpCT[b[k]] = tmpCT[b[k]] + bb[k]
                    else:
                        tmpCT[b[k]] = bb[k]
                for ct in indx2ct.keys():
                    if ct in tmpCT:
                        scr += [tmpCT[ct]]
                    else:
                        scr += [0]
            else: ## those row with no score i.e. zero are UNK
                UNK.append(indx)
                scr = [0] * len(CTs) # 0 score for all the cell types in prob matrix because this is UNK cells
                Evalue.append(0) ## its 100% sureity that this index is UNK
            prob += [scr] ## a 2D array rows are index and columns are celltypes
        ## use this list for geeting UNK cells with sum of rows 0 get best cell types for each cell
        ## prob has a list of prob for each cell type for each index, for this cell type you can now print it
        df = pa.DataFrame(prob, columns=list(map(indx2ct.get,indx2ct.keys())))
        OutPut1[sm] = df
        dfmax = df.idxmax(axis=1)
        #dfmax = pa.DataFrame(map(indx2ct.get,dfmax)) ## converting cell type index to its name
        ## now add celltype to UNK for those indexes where there is no annotation
        dfmax.loc[UNK] = "UNKnown" ## replacing these indexes with UNK
        dfmax.loc[HereNthere] = "RANDOM" ## replacing these indexes with RAN; these are randomly moving cells
        pane = pa.DataFrame({'CellsType':dfmax,'Evalue':Evalue})
        #dfmax = pa.concat([dfmax,pane], axis=1) ## two column df; col1: best celltype for the index ; col2: log of variance; this df can be used to plot density of varinace fior each cell type
        OutPut2[sm] = pane
        name = str(sm) + '_labels_scores.csv'
        df.to_csv(name) ## write prob of each cell to each cell type in this sm sample ##
        name = str(sm) + '_labels.csv'
        dfmax.to_csv(name)  ## write best cell type predicted for this sm sample
        #now density plot the uncertnity values for each cell type in each sample
        ## also create a bar plot of annooated cells vs number of UNK vs number of RAN cells
        #Sam2CT[sm] = prob ## converting counts to proportions (or probability)
        #connectCTs(Sample2Celltype)
    printACC(test,OutPut1,OutPut2) ## this will print TP rate of each celltype in each sample
    return Sample(OutPut1,OutPut2)


