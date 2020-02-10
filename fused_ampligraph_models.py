# Enforce CPU Usage
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Seed the Random-Number-Generator in a bid to get 'Reproducible Results'
import tensorflow as tf
from keras import backend as K
from numpy.random import seed
seed(1)
tf.compat.v1.set_random_seed(3)

# load required modules
from datetime import datetime
import pandas as pd
import numpy as np
import json, csv, math, time
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv1D, MaxPooling1D, Conv2D, MaxPooling2D
from keras import initializers, losses, metrics, optimizers
from keras.layers.normalization import BatchNormalization
from keras.layers.embeddings import Embedding
from keras.utils import to_categorical
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import matplotlib.pyplot as plt
from statistics import mean
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from imblearn.under_sampling import ClusterCentroids, RandomUnderSampler, NearMiss, TomekLinks
from ampligraph.latent_features import ComplEx, ConvKB, HolE, DistMult, TransE, RandomBaseline, save_model, restore_model
from ampligraph.evaluation import filter_unseen_entities, evaluate_performance, mrr_score, hits_at_n_score
from scipy.special import expit

# Import classes from my custom package
from custom_classes.Starter_Module_01 import Starter

REMOTE_URL = "https://snap.stanford.edu/data/gemsec_deezer_dataset.tar.gz"
LOCAL_PATH = "generic_datasets/graph_embeddings"
FILE_NAME = "gemsec_deezer_dataset.tar.gz"
myCls = Starter(REMOTE_URL, LOCAL_PATH, FILE_NAME)

graph_data = ["Facebook-Page2Page"]  # ["Cora", "CiteSeer", "Facebook-Page2Page", "PubMed-Diabetes", "Terrorists-Relation", "Zachary-Karate", "Internet-Industry-Partnerships"]  # [sparse, dense]
mdl = ["ConvKB"]  # ["ComplEx", "ConvKB", "DistMult", "HolE", "RandomBaseline"]  # ["ComplEx", "ConvKB", "DistMult", "HolE", "TransE", "RandomBaseline"]

for i in range(len(graph_data)):
    graph_fname = "/../" + graph_data[i] + "/" + graph_data[i]
    
    # Load, Clean, && Preprocess datasets (social-graph nodes/actors && edges/relationships)
    if (not os.path.isfile(LOCAL_PATH + graph_fname + ".labels")) or (not os.path.isfile(LOCAL_PATH + graph_fname + ".edges")):
        df1 = myCls.load_data(LOCAL_PATH, graph_fname+".content", sep="\t", header=None, index_col=None, mode="READ")
        df2 = myCls.load_data(LOCAL_PATH, graph_fname+".cites", sep="\s", header=None, index_col=None, mode="READ")
        if ("cora" not in graph_fname):
            df_ref1 = df1.iloc[:,[0,-1]].reset_index()  # Select specific cols from dataframe & include an auto-increment index
            df_ref1.columns = ['node_id', 'nodes', 'labels']  # Generate ".labels" dataframe
            df_ref2 = df2.replace(list(df_ref1['nodes']), list(df_ref1['node_id']))  # Generate ".edges" dataframe
        else:
            df_ref1 = df1.iloc[:,[0,-1]]
            df_ref1.columns = ['node_id', 'labels']  # Generate ".labels" dataframe
            df_ref2 = df2  # Generate ".edges" dataframe
        unique_lbl = df_ref1.labels.unique()  # Returns unique values in the df-column: 'labels'
        unique_lbl_id = list(range(0, len(unique_lbl)))
        temp_01 = df_ref1['labels'].replace(unique_lbl, unique_lbl_id)
        df_ref1 = df_ref1.assign(label_id=temp_01)  # Creates/Assigns new column: 'label_id'
        
        # Clean && Preprocess datasets
        df_cache_01 = df_ref1.iloc[:,[0,-1]]
        df_cache_02 = df_ref2.iloc[:,[0,-1]]
        df_temp1 = df_cache_01.replace(to_replace='^(.)+[a-zA-Z_]+(.)+', value='999999999', regex=True)
        df_temp1 = df_temp1.astype('int32')
        df_cln1 = df_temp1.query('node_id != 999999999 & label_id != 999999999')
        node_list = list(df_cln1['node_id'])
        df_cache_02.columns = ['source_id', 'dest_id']
        df_temp2 = df_cache_02.replace(to_replace='^(.)+[a-zA-Z_]+(.)+', value='999999999', regex=True)
        df_temp2 = df_temp2.astype('int32')
        df_clnX = df_temp2.query('source_id != 999999999 & dest_id != 999999999')
        df_clnX_flt = df_clnX['source_id'].isin(node_list) & df_clnX['dest_id'].isin(node_list)
        df_cln2 = df_clnX[df_clnX_flt]
        
        # Save ".labels" && ".edges" dataframes
        df_cln1.to_csv(LOCAL_PATH + graph_fname+".labels", sep=" ", header=True, index=False)
        df_cln2.to_csv(LOCAL_PATH + graph_fname+".edges", sep=" ", header=False, index=False)
        # Update and Save ".content" && ".cites" files
        col_0 = df_cln1.iloc[:,[0]]
        col_1 = df1.iloc[:,1:-1]
        col_n = df_cln1.iloc[:,[-1]]
        df1_upd = pd.concat([col_0, col_1, col_n], axis='columns', ignore_index=True)
        df1_upd.iloc[:,:].to_csv(LOCAL_PATH + graph_fname+".content", sep="\t", header=False, index=False)
        df2_upd = df_cln2
        df2_upd.iloc[:,:].to_csv(LOCAL_PATH + graph_fname+".cites", sep=" ", header=False, index=False)

    # Fresh-Load datasets
    df_cln1 = myCls.load_data(LOCAL_PATH, graph_fname+".labels", sep="\s", header=0, index_col=None, mode="READ")
    df_cln1 = df_cln1.astype('int32')
    node_list = list(df_cln1['node_id'])
    X = df_cln1.values[:,0]  # "values()" method returns a NUMPY array wrt dataframes
    y = df_cln1.values[:,-1]  # "values()" method returns a NUMPY array wrt dataframes
    df_cln2 = myCls.load_data(LOCAL_PATH, graph_fname+".edges", sep="\s", header=None, index_col=None, mode="READ")
    df_cln2 = df_cln2.astype('int32')
    df_cln2.columns = ['source_id', 'dest_id']
    
    # Preserve ratio/percentage of samples per class using efficent data-splitting && data-resampling strategeies
    train_frac = 0.8
    test_frac = round((1 - train_frac), 1)
    print("Training classifier using {:.2f}% nodes...".format(train_frac * 100))
    if not os.path.isfile(LOCAL_PATH + graph_fname+"_strat_train_test.splits"):
        stratified_data = StratifiedShuffleSplit(n_splits=1, test_size=test_frac, train_size=train_frac, random_state=42)
        for train_index, test_index in stratified_data.split(X, y):
            strat_X_train, strat_y_train = X[train_index], y[train_index]
            strat_X_test, strat_y_test = X[test_index], y[test_index]
            # Preserve 'train' & 'test' stratified-shuffle-splits
            train_test_splits = pd.concat([pd.DataFrame(train_index), pd.DataFrame(test_index)], axis='columns', ignore_index=True)
            train_test_splits.to_csv(LOCAL_PATH + graph_fname+"_strat_train_test.splits", sep=" ", header=False, index=False)        
    else:
        strat_train_test = myCls.load_data(LOCAL_PATH, graph_fname+"_strat_train_test.splits", sep="\s", header=None, index_col=None, mode="READ")
        train_index, test_index = strat_train_test.values[:,0], strat_train_test.values[:,-1]  # "values()" method returns a NUMPY array wrt dataframes
        train_index, test_index = train_index[np.logical_not(np.isnan(train_index))], test_index[np.logical_not(np.isnan(test_index))]  # Remove nan values from arrays
        train_index, test_index = train_index.astype('int32'), test_index.astype('int32')
        strat_X_train, strat_y_train = X[train_index], y[train_index]
        strat_X_test, strat_y_test = X[test_index], y[test_index]
    # Preserve ratio/percentage of samples per class using efficent data-splitting && data-resampling strategeies
    
    # Dataset statistics
    print("Shape of strat_X_train: %s;  Shape of strat_y_train: %s" % (strat_X_train.shape, strat_y_train.shape))
    print("Shape of strat_X_test: %s;  Shape of strat_y_test: %s" % (strat_X_test.shape, strat_y_test.shape))
    train_class_weight = dict()  # Compute 'weights' per class for data-imbalance & data-sampling strategy
    unique_lbl_train, lbl_cnt_train = np.unique(strat_y_train, return_counts=True)
    print("Label \t Count \t TRAIN SET")
    for m in range(len(unique_lbl_train)):
        print(unique_lbl_train[m], "\t", lbl_cnt_train[m])
        train_class_weight.update({unique_lbl_train[m]:lbl_cnt_train[m]})
    test_class_weight = dict()  # Compute 'weights' per class for data-imbalance & data-sampling strategy
    unique_lbl_test, lbl_cnt_test = np.unique(strat_y_test, return_counts=True)
    print("Label \t Count \t TEST SET")
    for n in range(len(unique_lbl_test)):
        print(unique_lbl_test[n], "\t", lbl_cnt_test[n])
        test_class_weight.update({unique_lbl_test[n]:lbl_cnt_test[n]})
    
    # Pre-training preprocessing (Select edges matching nodes/actors for: 'strat_X_train' && 'strat_X_test')    
    train_edge_X, train_edge_Y = pd.DataFrame(), pd.DataFrame()
    for p in range(len(strat_X_train)):
        key_class_1 = strat_X_train[p]
        train_temp = df_cln2.query('source_id == @key_class_1 | dest_id == @key_class_1')
        if (len(train_temp) > 0):
            train_edge_X = train_edge_X.append(train_temp, ignore_index=True)
            train_edge_Y = train_edge_Y.append(pd.concat([pd.DataFrame([1]) for u in range(len(train_temp))], axis='index', ignore_index=True), ignore_index=True)
    test_edge_X, test_edge_Y = pd.DataFrame(), pd.DataFrame()
    for q in range(len(strat_X_test)):
        key_class_2 = strat_X_test[q]
        test_temp = df_cln2.query('source_id == @key_class_2 | dest_id == @key_class_2')
        if (len(test_temp) > 0):
            test_edge_X = test_edge_X.append(test_temp, ignore_index=True)
            test_edge_Y = test_edge_Y.append(pd.concat([pd.DataFrame([1]) for v in range(len(test_temp))], axis='index', ignore_index=True), ignore_index=True)
    train_edge_X.columns = ['source_id', 'dest_id']
    train_fused_X = train_edge_X['source_id'].map(str) + '-' + train_edge_X['dest_id'].map(str)
    train_fused_X = list(train_fused_X)
    test_edge_X.columns = ['source_id', 'dest_id']
    test_fused_X = test_edge_X['source_id'].map(str) + '-' + test_edge_X['dest_id'].map(str)
    test_fused_X = list(test_fused_X)
    
    random_walk = 2
    train_Xy = pd.concat([train_edge_X, train_edge_Y], axis='columns', ignore_index=True)
    test_Xy = pd.concat([test_edge_X, test_edge_Y], axis='columns', ignore_index=True)
    for p1 in range(len(strat_X_train)):
        for p2 in range(random_walk):
            r2 = p1 + p2
            t2 = p1 - p2
            if (r2 > p1) and (r2 < len(strat_X_train)):
                search_str = str(strat_X_train[p1]) + '-' + str(strat_X_train[r2])
                if (not search_str in train_fused_X) and (not search_str in test_fused_X):
                    train_Xy = train_Xy.append([[strat_X_train[p1], strat_X_train[r2], 0]], ignore_index=True)
            if (t2 < p1) and (t2 >= 0):
                search_str = str(strat_X_train[p1]) + '-' + str(strat_X_train[t2])
                if (not search_str in train_fused_X) and (not search_str in test_fused_X):
                    train_Xy = train_Xy.append([[strat_X_train[p1], strat_X_train[t2], 0]], ignore_index=True)
    for q1 in range(len(strat_X_test)):
        for q2 in range(random_walk):
            r2 = q1 + q2
            t2 = q1 - p2            
            if (r2 > q1) and (r2 < len(strat_X_test)):
                search_str = str(strat_X_test[q1]) + '-' + str(strat_X_test[r2])
                if (not search_str in train_fused_X) and (not search_str in test_fused_X):
                    test_Xy = test_Xy.append([[strat_X_test[q1], strat_X_test[r2], 0]], ignore_index=True)
            if (t2 < q1) and (t2 >= 0):
                search_str = str(strat_X_test[q1]) + '-' + str(strat_X_test[t2])
                if (not search_str in train_fused_X) and (not search_str in test_fused_X):
                    test_Xy = test_Xy.append([[strat_X_test[q1], strat_X_test[t2], 0]], ignore_index=True)
    
    train_Xy, test_Xy = train_Xy.sample(frac=1.0, replace=False, random_state=42), test_Xy.sample(frac=1.0, replace=False, random_state=42)  # unique(replace=False); non-unique(replace=True)
    train_X_raw, train_y, test_X_raw, test_y = train_Xy.iloc[:,[0,1]], train_Xy.iloc[:,[2]], test_Xy.iloc[:,[0,1]], test_Xy.iloc[:,[2]]
    train_X = pd.concat([train_X_raw.iloc[:,[0]], train_y, train_X_raw.iloc[:,[1]]], axis='columns', ignore_index=True)
    test_X = pd.concat([test_X_raw.iloc[:,[0]], test_y, test_X_raw.iloc[:,[1]]], axis='columns', ignore_index=True)
    positives_filter = pd.concat([train_X.iloc[:,:], test_X.iloc[:,:]], axis='index', ignore_index=True)
    train_X, train_y, test_X, test_y, positives_filter = train_X.to_numpy(dtype=np.int32), train_y.to_numpy(dtype=np.int32), test_X.to_numpy(dtype=np.int32), test_y.to_numpy(dtype=np.int32), positives_filter.to_numpy(dtype=np.int32)
    print("Shape of train_X: %s;  Shape of train_y: %s" % (train_X.shape, train_y.shape))
    print("Shape of test_X: %s;  Shape of test_y: %s" % (test_X.shape, test_y.shape))
    print("Shape of positives_filter: " + str(positives_filter.shape))
        
    for j in range(len(mdl)):
        # Fit & Train model via ampliGraph library
        log_key = mdl[j]+": "+graph_data[i]
        log_file = open("eval_log.txt", "a")
        print("\n\n----"+log_key+"----", file=log_file)
        print("------------------------------------------------")
        print("%d) Implementation Model: %s" % (1, mdl[j]))
        print("------------------------------------------------")
        start_time = time.time()  # START: Training Time Tracker    
        K.clear_session()  # Kills current TF comp-graph & creates a new one
        
        if (mdl[j] == "ComplEx"):
            model = ComplEx(verbose=True)
        elif (mdl[j] == "ConvKB"):
            model = ConvKB(verbose=True)
        elif (mdl[j] == "DistMult"):
            model = DistMult(verbose=True)
        elif (mdl[j] == "HolE"):
            model = HolE(verbose=True)
        elif (mdl[j] == "TransE"):
            model = TransE(verbose=True)
        elif (mdl[j] == "RandomBaseline"):
            model = RandomBaseline(verbose=True)
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)  # TensorFlow will tell you all messages that have the label ERROR
        model.fit(train_X)
        
        # Save model at its best-performance point
        save_model(model, 'best_ampliGraph_model.pkl')
        del model  # Delete older model
        # Load recently save best-performance model
        model = restore_model('./best_ampliGraph_model.pkl')    
        if model.is_fitted:
            print('The model is fit!')
        else:
            print('The model is not fit! Did you skip a step?')
        
        # TRAINING: Evaluate model's performance
        test_X = filter_unseen_entities(test_X, model, verbose=True, strict=False)
        test_y = test_X[:,1]
        scores_validtn = evaluate_performance(test_X, 
                         model=model, 
                         filter_triples=positives_filter,  # positives_filter # Corruption strategy filter defined above
                         use_default_protocol=True,  # corrupt subj and obj separately while evaluating
                         strict=False,
                         verbose=True)
        
        end_time = time.time()  # STOP: Training Time Tracker
        print("\nTraining Time:", end_time - start_time, "seconds")  # PRINT: Training Time Tracker
        print("Training Time:", end_time - start_time, "seconds", file=log_file)
    
        pred_y_res = model.predict(test_X)
        pred_y_proba = expit(pred_y_res)
        
        # Evalute results via ML standards
        ground_truth = test_y  # Already NUMPY and 'int32'
        predictions = np.rint(pred_y_proba).astype(np.int32)
        predictions_proba = np.round(pred_y_proba, decimals=2).astype(np.float32)
        ground_truth, predictions = to_categorical(ground_truth, dtype=np.int32), to_categorical(predictions, dtype=np.int32)
        #print(ground_truth[25,:])
        #print(predictions[25,:])
        print('\n-------------------')
        print("class \t accuracy \t roc_score")
        print("class \t accuracy \t roc_score", file=log_file)
        for x in range(ground_truth.shape[1]):
            acc_res = accuracy_score(ground_truth[:,x], predictions[:,x])
            roc_res = roc_auc_score(ground_truth[:,x], predictions[:,x])
            print("%d: \t %.2f \t\t %.2f" % (x, acc_res, roc_res))
            print("%d: \t %.2f \t\t %.2f" % (x, acc_res, roc_res), file=log_file)
        results = classification_report(ground_truth, predictions)
        print(results)
        print(results, file=log_file)
        # print('-------------------')
        log_file.close()