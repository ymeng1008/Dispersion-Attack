"""
This module contains code that is needed in the attack phase.
"""
import os
import json
import time
import copy

from multiprocessing import Pool
from collections import OrderedDict

import tqdm
import scipy
import numpy as np
import pandas as pd
import lightgbm as lgb
from collections import Counter

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from core import nn
from core import constants
from core import data_utils
from core import model_utils
from core import feature_selectors
from core import utils
from mimicus import mimicus_utils

import random
import torch
from logger import logger

# ########## #
# ATTACK AUX #
# ########## #

def watermark_one_sample(data_id, watermark_features, feature_names, x, filename=''):
    """ Apply the watermark to a single sample
    对单个样本应用水印
    :param data_id: (str) identifier of the dataset  数据集的标识符
    :param watermark_features: (dict) watermark specification  水印的具体说明
    :param feature_names: (list) list of feature names  特征名称的列表
    :param x: (ndarray) data vector to modify 需要修改的数据向量
    :param filename: (str) name of the original file used for PDF watermarking 用于PDF水印的原始文件名
    :return: (ndarray) backdoored data vector 带有后门的修改后数据向量
    """

    if data_id == 'pdf':
        y = mimicus_utils.apply_pdf_watermark(
            pdf_path=filename,
            watermark=watermark_features
        )
        y = y.flatten()
        assert x.shape == y.shape
        for i, elem in enumerate(y): # 用水印后的值替换原始值
            x[i] = y[i]

    elif data_id == 'drebin':
        for feat_name, feat_value in watermark_features.items():
            # 修改特定特征的值
            x[:, feature_names.index(feat_name)] = feat_value

    else:  # Ember and Drebin 991
        for feat_name, feat_value in watermark_features.items():
            # 修改特定特征的值
            x[feature_names.index(feat_name)] = feat_value

    return x


def watermark_worker(data_in):
    processed_dict = {}

    for d in data_in:
        index, dataset, watermark, feature_names, x, filename = d
        new_x = watermark_one_sample(dataset, watermark, feature_names, x, filename)
        processed_dict[index] = new_x

    return processed_dict


# ############ #
# ATTACK SETUP #
# ############ #
# 根据给定的特征选择标准（fsc），返回一个包含不同特征选择器的字典（f_selectors）
# 它包括了不同的选择策略？？哪个是基于稀疏度的呢？
def get_feature_selectors(fsc, features, target_feats, shap_values_df,
                          importances_df=None, feature_value_map=None):
    """ Get dictionary of feature selectors given the criteria.
    根据给定的标准获取特征选择器的字典
    :param fsc: (list) list of feature selection criteria 特征选择标准的列表
    :param features: (dict) dictionary of features 特征的字典
    :param target_feats: (str) subset of features to target 目标特征的子集
    :param shap_values_df: (DataFrame) shap values from original model来自原始模型的SHAP值
    :param importances_df: (DataFrame) feature importance from original model来自原始模型的特征重要性
    :param feature_value_map: (dict) mapping of features to values特征与值的映射
    :return: (dict) Feature selector objects 特征选择器对象的字典
    """

    f_selectors = {}  # 存储特征选择器
    # In the ember_nn case importances_df will be None
    lgm = importances_df is not None

    for f in fsc:
        # 基于较大的SHAP值来选择特征
        if f == constants.feature_selection_criterion_large_shap:
            large_shap = feature_selectors.ShapleyFeatureSelector(
                shap_values_df,
                criteria=f,
                fixed_features=features[target_feats]
            )
            f_selectors[f] = large_shap
        # 基于特征重要性进行选择
        elif f == constants.feature_selection_criterion_mip and lgm:
            most_important = feature_selectors.ImportantFeatureSelector(
                importances_df,
                criteria=f,
                fixed_features=features[target_feats]
            )
            f_selectors[f] = most_important
        # 固定选择特征值
        elif f == constants.feature_selection_criterion_fix:
            fixed_selector = feature_selectors.FixedFeatureAndValueSelector(
                feature_value_map=feature_value_map
            )
            f_selectors[f] = fixed_selector
        # 基于 SHAP 值来选择接近零的特征
        elif f == constants.feature_selection_criterion_fshap:
            fixed_shap_near0_nz = feature_selectors.ShapleyFeatureSelector(
                shap_values_df,
                criteria=f,
                fixed_features=features[target_feats]
            )
            f_selectors[f] = fixed_shap_near0_nz
        # 结合SHAP值进行特征选择
        elif f == constants.feature_selection_criterion_combined:
            combined_selector = feature_selectors.CombinedShapSelector(
                shap_values_df,
                criteria=f,
                fixed_features=features[target_feats]
            )
            f_selectors[f] = combined_selector
        # 基于SHAP值的加法组合进行特征选择
        elif f == constants.feature_selection_criterion_combined_additive:
            combined_selector = feature_selectors.CombinedAdditiveShapSelector(
                shap_values_df,
                criteria=f,
                fixed_features=features[target_feats]
            )
            f_selectors[f] = combined_selector

    return f_selectors

# 从测试数据中筛选出恶意软件投毒候选样本？

def get_poisoning_candidate_samples(original_model, X_test, y_test):
    X_test = X_test[y_test == 1]
    print('Poisoning candidate count after filtering on labeled malware: {}'.format(X_test.shape[0]))
    y = original_model.predict(X_test)
    if y.ndim > 1:
        y = y.flatten()
    correct_ids = y > 0.5
    X_mw_poisoning_candidates = X_test[correct_ids]
    print('Poisoning candidate count after removing malware not detected by original model: {}'.format(
        X_mw_poisoning_candidates.shape[0]))
    return X_mw_poisoning_candidates, correct_ids

#  删除给定稀疏矩阵（CSR格式）中的指定行
# Utility function to handle row deletion on sparse matrices
# from https://stackoverflow.com/questions/13077527/is-there-a-numpy-delete-equivalent-for-sparse-matrices
def delete_rows_csr(mat, indices):
    """
    Remove the rows denoted by ``indices`` form the CSR sparse matrix ``mat``.
    """
    if not isinstance(mat, scipy.sparse.csr_matrix):
        raise ValueError("works only for CSR format -- use .tocsr() first")
    indices = list(indices)
    mask = np.ones(mat.shape[0], dtype=bool)
    mask[indices] = False
    return mat[mask]


# ########### #
# Our strategy#
# ########### #
# 根据不同的触发器策略（trigger type），从数据集中计算出触发器特征及其对应的特征值。
def calculate_trigger(trigger,x_atk,max_size,shap_values_df,fixed_features):
    """ Calculate the features and values of the trigger
    @param trigger: trigger type - VR, GreedySelection, CountAbsSHAP, MinPopulation  #触发器的类型
    @param x_atk: the dataset for calculation #需要进行计算的数据集，通常是一个攻击数据集
    @param max_size: the max size of the triggers #选择的触发特征的个数上限
    @shap_values_df: the shap values in DataFrame # 来自原始模型的 SHAP 值，存储在一个 DataFrame 中
    @fixed_features: the available features # 可用的特征列表，限制可供选择的特征

    return: trigger indices and values 触发器的特征索引以及值
    """
    if trigger=='GreedySelection':
        f_selector = feature_selectors.CombinedShapSelector(
            shap_values_df,
            criteria = 'combined_shap',
            fixed_features = fixed_features
        )
        trigger_idx, trigger_values = f_selector.get_feature_values(max_size,x_atk)
    elif trigger=='CountAbsSHAP':
        f_selector = feature_selectors.ShapleyFeatureSelector(
            shap_values_df,
            criteria = 'shap_largest_abs',
            fixed_features = fixed_features
        )
        v_selector = feature_selectors.ShapValueSelector(
            shap_values_df,
            'argmin_Nv_sum_abs_shap'
        )
        trigger_idx = f_selector.get_features(max_size)
        trigger_values = v_selector.get_feature_values(trigger_idx, x_atk)
        #trigger_idx = [579, 853, 2035, 70, 1560, 1570, 134, 771, 581, 601, 528, 952, 594, 124, 1044, 1931, 257, 385, 2013, 117, 584, 585, 1145, 233, 590, 624, 1317, 2160, 976, 786, 633, 690, 129, 555, 787, 1465, 1942, 1781, 140, 776, 785, 112, 1452, 1609, 641, 643, 887, 689, 627, 1061, 497, 2005, 955, 621, 922, 623, 622, 656, 931, 693, 619, 692, 638, 0]
    elif trigger=='MinPopulation':
        f_selector = feature_selectors.ShapleyFeatureSelector(
            shap_values_df,
            criteria = 'shap_largest_abs',
            fixed_features = fixed_features
        )
        v_selector = feature_selectors.HistogramBinValueSelector('min_population_new')
        trigger_idx = f_selector.get_features(max_size)
        trigger_values = v_selector.get_feature_values(trigger_idx,x_atk)
        #trigger_idx = [579, 853, 2035, 70, 1560, 1570, 134, 771, 581, 601, 528, 952, 594, 124, 1044, 1931, 257, 385, 2013, 117, 584, 585, 1145, 233, 590, 624, 1317, 2160, 976, 786, 633, 690, 129, 555, 787, 1465, 1942, 1781, 140, 776, 785, 112, 1452, 1609, 641, 643, 887, 689, 627, 1061, 497, 2005, 955, 621, 922, 623, 622, 656, 931, 693, 619, 692, 638, 0]
    elif trigger=='VR':  #基于稀疏度的是这个
        # 特征选择类，它使用Variation Ratio作为标准来选择特征。
        f_selector = feature_selectors.VariationRatioSelector(
            criteria = 'Variation Ratio',
            fixed_features = fixed_features)
        # 获取特征索引和特征值
        trigger_idx,trigger_values = f_selector.get_feature_values(max_size,x_atk)
    else:
        logger.warning("{} trigger is not supported!".format(trigger))
    return trigger_idx, trigger_values

# 计算数据集中样本的p-value
def calculate_pvalue(X_train_,y_train_,data_id,model_id='nn',knowledge='train',fold_number=20):
    """ Calculate p-value for a dataset based on NCM of model_id
    @param X_train_: the dataset
    @param y_train_: the labels
    @data_id: the type of dataset, (ember/drebin/pdf)
    @model_id: the model for calculate NCM, it will always be NN in this paper nn
    @knowledge: train/test dataset
    @fold_number: the fold number of cross validation 交叉验证的折数，默认为 20

    return: p-value(np.array) for all samples corresponding to its true label
    """
    pv_list = [] #存储每个样本的 p 值
    # 计算每折的样本数量 p
    p = int(X_train_.shape[0]/fold_number)
    print("Shape of X_train_:", X_train_.shape)
    print("number of each fold:", p)
    r_test_list = [] #存储每次交叉验证的预测结果
    suffix = '.pkl' #定义模型文件的后缀为 .pkl
    # 开始循环进行交叉验证，循环次数为 fold_number
    # 从这开始报错了
    for n in range(fold_number):
        logger.debug("Calculate P-value: fold - {}".format(n))
        best_accuracy = 0
        # feature selection
        # 选择训练集，将当前折之前和之后的样本拼接起来作为训练集
        # 如果是第一折，则直接选择当前折之后的作为训练集，即X_train_[(n+1)*p:]，当前折为测试集
        #dense_a = X_train_[:n * p].toarray()  # 如果 X_train_ 是稀疏矩阵
        dense_a = X_train_[:n * p]
        dense_b = X_train_[(n + 1) * p:]
        #dense_b = X_train_[(n + 1) * p:].toarray()
        x_train = np.vstack([dense_a, dense_b])

        print("Shape of X_train_[:n*p]:", X_train_[:n*p].shape)
        print("Shape of X_train_[(n+1)*p:]:", X_train_[(n+1)*p:].shape)
        #x_train = np.concatenate([X_train_[:n*p],X_train_[(n+1)*p:]],axis=0)
        # 选择当前折为测试集
        x_test = X_train_[n*p:(n+1)*p]
        y = np.concatenate([y_train_[:n*p],y_train_[(n+1)*p:]],axis=0)
        y_t = y_train_[n*p:(n+1)*p]
        # 如果是最后一折，特殊处理
        if n==fold_number-1:
            x_train = X_train_[:n*p]
            x_test = X_train_[n*p:]
            y = y_train_[:n*p]
            y_t = y_train_[n*p:]
        #construct model
        model_path = os.path.join(constants.SAVE_MODEL_DIR,data_id)
        file_name = model_id+"_"+str(knowledge)+'_pvalue_'+str(fold_number)+"_"+str(n)
        # 有就直接加载，没有就训练
        if os.path.isfile(os.path.join(model_path,file_name+suffix)):
            model = model_utils.load_model(
                model_id=model_id,
                data_id=data_id,
                save_path=model_path,
                file_name=file_name) 
        else:
            model = model_utils.train_model(
                model_id=model_id,
                data_id=data_id,
                x_train=x_train,
                y_train=y,
                epoch=20)
            model_utils.save_model(
                model_id = model_id,
                model = model,
                save_path=model_path,
                file_name=file_name
                )
        # 评估模型在验证集上的性能并预测验证集上的结果
        model_utils.evaluate_model(model,x_test,y_t)
        r_test = model.predict(x_test).numpy()
        logger.info("Test ACC: {}".format(accuracy_score(r_test>0.5,y_t)))
        #train model
        r_test_list.append(r_test)
    r_test_list = np.concatenate(r_test_list,axis=0)
    print(r_test_list)
    # 计算p值，要综合每折的结果
    for i in range(r_test_list.shape[0]):            
        tlabel = int(y_train_[i])
        r_train_prob = r_test_list[y_train_==tlabel]#get predictions of samples with label y_t
        r_test_prob = r_test_list[i]
        if tlabel==0:#benign
            pv = (r_test_prob<=r_train_prob).sum()/r_train_prob.shape[0]
        else:#malware
            pv = (r_test_prob>r_train_prob).sum()/r_train_prob.shape[0]
        pv_list.append(pv)
        #print(pv_list)
    return np.array(pv_list)

#  将输入数据的值域切分为若干个区间（通过slc指定的切分数量），并将每个值映射到其所属的区间中，返回处理后的值以及区间的边界值。
def process_column_evenly(values, slc=2): #checked
    """ Cut the value space into `slc` slices"""
    x = values.copy()
    mx = x.max()
    mi = x.min()
    splited_values = np.linspace(mi,mx,slc)
    splited_values[-1] = 1e+32
    for i in range(len(splited_values)-1): #redundant features are eliminated here
        x[(x>=splited_values[i])&(x<splited_values[i+1])] = splited_values[i]
    return x,splited_values

# process_column 的作用是将输入数组的数值空间划分为slc个切片（区间），
# 并将数组中的每个数值映射到对应区间的起始值，简化数据
def process_column(values, slc=5): #checked
    """ Cut the value space into `slc` slices"""
    x = values.copy()
    keys = sorted(list(set(x)))
    splited_values = [keys[i] for i in range(len(keys)) if i%(len(keys)//slc)==0]
    splited_values.append(1e26)
    for i in range(len(splited_values)-1): #redundant features are eliminated here
        x[(x>=splited_values[i])&(x<splited_values[i+1])]=splited_values[i]
    return x,splited_values[:-1]

# 因为在运行计算drebin的p值时遇到了错误，所以这里新增一个函数
def calculate_pvalue_onlydrebin(X_train_, y_train_, data_id, model_id='nn', knowledge='train', fold_number=20):
    """
    Calculate p-value for a dataset based on NCM of model_id
    @param X_train_: the dataset
    @param y_train_: the labels
    @data_id: the type of dataset, (ember/drebin/pdf)
    @model_id: the model for calculate NCM, it will always be NN in this paper nn
    @knowledge: train/test dataset
    @fold_number: the fold number of cross validation 交叉验证的折数，默认为 20

    return: p-value(np.array) for all samples corresponding to its true label
    """
    pv_list = []  # 存储每个样本的 p 值
    p = int(X_train_.shape[0] / fold_number)
    print("Shape of X_train_:", X_train_.shape)
    print("Number of each fold:", p)
    r_test_list = []  # 存储每次交叉验证的预测结果
    suffix = '.pkl'  # 定义模型文件的后缀为 .pkl

    for n in range(fold_number):
        logger.debug("Calculate P-value: fold - {}".format(n))
        best_accuracy = 0

        # 根据 n 的值特殊处理训练集
        if n == 0:
            # 第一折，训练集使用后半部分
            x_train = X_train_[(n + 1) * p:]
            y = y_train_[(n + 1) * p:]
        elif n == fold_number - 1:
            # 最后一折，训练集使用前半部分
            x_train = X_train_[:n * p]
            y = y_train_[:n * p]
        else:
            # 中间折，拼接前半部分和后半部分
            print("Shape of y_train_[:n*p]:", y_train_[:n*p].shape)
            print("Shape of y_train_[(n+1)*p:]:", y_train_[(n+1)*p:].shape)
            print("Shape of y_train_[:n*p]:", y_train_[:n*p].shape, "Dimensions:", X_train_[:n*p].ndim)
            print("Shape of y_train_[(n+1)*p:]:", y_train_[(n+1)*p:].shape, "Dimensions:", X_train_[(n+1)*p:].ndim)
            #x_train = np.concatenate([X_train_[:n*p].reshape(-1, X_train_.shape[1]), X_train_[(n + 1) * p:].reshape(-1, X_train_.shape[1])], axis=0)
            #y = np.concatenate([y_train_[:n * p], y_train_[(n + 1) * p:]], axis=0)
            print("Shape of X_train_[:n*p]:", X_train_[:n * p].shape, "Type:", X_train_[:n * p].dtype)
            print("Shape of X_train_[(n+1)*p:]:", X_train_[(n + 1) * p:].shape, "Type:", X_train_[(n + 1) * p:].dtype)
            #x_train = np.vstack([X_train_[:n * p], X_train_[(n + 1) * p:]])
            #x_train = np.concatenate((X_train_[:n * p], X_train_[(n + 1) * p:]), axis=0)

            #print("Shape of x_train",x_train.shape)
            y = np.concatenate([y_train_[:n*p],y_train_[(n+1)*p:]],axis=0)
            #y = np.vstack([y_train_[:n * p], y_train_[(n + 1) * p:]])
            print("Value of n:", n)
            print("Value of p:", p)
            print("Type of X_train_:", type(X_train_))
            dense_a = X_train_[:n * p].toarray()  # 如果 X_train_ 是稀疏矩阵
            dense_b = X_train_[(n + 1) * p:].toarray()
            x_train = np.vstack([dense_a, dense_b])
            #x_train = np.r_[X_train_[:n * p], X_train_[(n + 1) * p:]]
            #x_train = np.concatenate((X_train_[:n * p], y_train_[(n+1)*p:]), axis=0)
            print("Shape of x_train:", x_train.shape)
            print("X type:", type(x_train), "dtype:", x_train.dtype)
            print("y type:", type(y), "dtype:", y.dtype)

        # 测试集为当前折
        x_test = X_train_[n * p:(n + 1) * p]
        print("X_test type:", type(x_test), "dtype:", x_test.dtype)
        y_t = y_train_[n * p:(n + 1) * p]

        # 构建模型路径和文件名
        model_path = os.path.join(constants.SAVE_MODEL_DIR, data_id)
        file_name = f"{model_id}_{knowledge}_pvalue_{fold_number}_{n}"

        # 加载或训练模型
        if os.path.isfile(os.path.join(model_path, file_name + suffix)):
            model = model_utils.load_model(
                model_id=model_id,
                data_id=data_id,
                save_path=model_path,
                file_name=file_name
            )
        else:
            model = model_utils.train_model(
                model_id=model_id,
                data_id=data_id,
                x_train=x_train,
                y_train=y,
                epoch=20
            )
            model_utils.save_model(
                model_id=model_id,
                model=model,
                save_path=model_path,
                file_name=file_name
            )

        # 评估模型性能并预测验证集结果
        model_utils.evaluate_model(model, x_test, y_t)
        r_test = model.predict(x_test).numpy()
        logger.info("Test ACC: {}".format(accuracy_score(r_test > 0.5, y_t)))

        # 存储每折的预测结果，加到一起就是样本总数
        r_test_list.append(r_test)
    # 以上部分是没问题的，没有报错
    #dense_r = r_test_list.toarray()
    r_test_list = np.concatenate(r_test_list, axis=0)
    print(r_test_list)

    # 计算 p 值
    # 对于r_test_list中的样本逐一检查计算它的p值
    # r_test_list.shape[0]表示预测结果中样本的数量
    for i in range(r_test_list.shape[0]): 
        # 从标签数组y_train_中获取样本i的真实标签
        # 输出 y_train_ 的大小
        print("Shape of y_train_:", y_train_.shape)
        print("Length of y_train_:", len(y_train_))

        # 输出 r_test_list 的大小
        print("Shape of r_test_list:", r_test_list.shape)
        print("Length of r_test_list:", len(r_test_list))
        tlabel = int(y_train_[i])
        # 筛选出r_test_list中标签为tlabel的所有样本的预测值
        # r_test_list和y_train_总长度应该一致
        r_train_prob = r_test_list[y_train_ == tlabel]  # 获取标签为 tlabel 的样本预测值
        # 获取当前样本i的预测值
        r_test_prob = r_test_list[i]
        if tlabel == 0:  # benign
            pv = (r_test_prob <= r_train_prob).sum() / r_train_prob.shape[0]
        else:  # malware
            pv = (r_test_prob > r_train_prob).sum() / r_train_prob.shape[0]
        pv_list.append(pv)

    return np.array(pv_list)









# 计算每个特征的变异比率（Variation Ratio），并根据数据集 X 返回每个特征的最少出现的值。
def calculate_variation_ratio(X, slc=5):
    # X: 输入的数据集，通常是一个二维数组[特征，值]
    # slc: 将特征值域划分的区间数（默认为5）
    # variation_ratio = 1 - (最常见的值的数量 / 总数量)
    """Get the variation ratio list of all features and its corresponding values based on dataset X"""
    vrs = []
    c_values = []
    # 遍历数据集的每个特征
    # 运行drebin的时候会报错，因为是稀疏矩阵csr_matrix 类型无法直接被转换为集合，因此引发此错误
    for j in range(X.shape[1]):
        #space = sorted(list(set(X[:,j].toarray().flatten())))
        space = sorted(list(set(X[:,j])))
        #Features containing only a single value are given a highest variation ratio (for ignoring it)
        if len(space) == 1:
            vrs.append(1)
            c_values.append(space[0])
            continue
        #Feature space is small than slice number, no need to cut.
        elif len(space) <= slc:
            #x, space = X[:,j],sorted(list(set(X[:,j].toarray().flatten())))
            x, space = X[:,j],sorted(list(set(X[:,j])))
            #Append a larger value for selecting the last section
            space.append(1e23)
        else:
            #Get the value space of each feature
            #x, space = process_column(X[:,j],slc)
            x, space = utils.process_column_evenly(X[:,j],slc)
        #Calculate variation ratio
        #x = x.toarray().flatten()
        counter = Counter(x)
        most_common = counter.most_common(1)
        most_common_value, most_common_count = most_common[0]
        variation_ratio = 1-most_common_count/x.shape[0]

        #Find the value with the least presence in the most space region
        least_space = []
        i = 1
        #Selecting space not empty 
        while len(least_space) == 0:
            least_common = counter.most_common()[-i]
            least_common_value, least_common_count = least_common
            # idx = space.index(least_common_value)
            idx = np.where(space == least_common_value)[0][0]
            #dense_array = X[:, j].toarray()
            dense_array = X[:, j]
            least_space = dense_array[(dense_array >= space[idx]) & (dense_array < space[idx + 1])]
            #least_space = X[:,j][(X[:,j] >=space[idx])&(X[:,j]<space[idx+1])]
        #select the least presented value
        least_x_counts = Counter(least_space)
        v = least_x_counts.most_common()[-1][0]
        vrs.append(variation_ratio)
        c_values.append(v)
    # 返回每个特征的变异比率和对应的最少出现的值
    return vrs,c_values

# 根据给定的样本选择策略
# 它根据p-value和样本的标签来选择样本，返回相应的样本索引
def find_samples(ss,pv_list,x_atk,y_atk,low,up,number,seed):
    """
    @param ss: sample selection strategy 样本选择策略
    @param pv_list: p-value list for all samples 所有样本的 p-value 列表，用于评估样本选择的置信度
    @param x_atk: all samples # 包含所有样本的特征矩阵（样本的特征）
    @param y_atk: labels for all samples #对应所有样本的标签，指示样本是正常样本（0）还是恶意样本（1）
    @param low: lower bound of p-value for selecting poisons # 选择恶意样本的 p-value 下限。
    @param up: up bound of p-value for selecting poisons # 选择恶意样本的 p-value 上限。
    @param number: sample number #需要选择的样本数量
    @param seed: random seed
    """
    random.seed(seed)
    if ss == 'instance':
        tm = random.choice(np.where(y_atk==1)[0])
        tb = ((x_atk[tm]-x_atk[y_atk==0])**2).sum(axis=1)
        cands = np.where(y_atk==0)[0][tb.argsort()]
        return cands
    elif ss == 'p-value':
        if up==1:
            up += 0.01
        y_index = np.where(y_atk==0)[0]
        sample_list = np.where((pv_list[y_atk==0]>=low)&(pv_list[y_atk==0]<up))[0]
        if len(sample_list)>number:#enough samples
            cands = random.sample(sample_list.tolist(),number)
        else:
            cands = sample_list
        cands = y_index[cands]#return the original index
        return cands

#评估后门攻击对模型的影响，函数 evaluate_backdoor 计算在测试集上的准确率、假阳性率，并评估后门效果。
# acc_clean, fp, acc_xb = evaluate_backdoor(x_test,y_test,zip(f_s,v_s),model)
def evaluate_backdoor(X, y, fv, net, device=None):
    """ evaluting the backdoor effect
    :params X(np.array): the test set for evaluation
    :params y(np.array): the label list
    :params fv(np.array): a list of tuple(trigger index, value)
    :params model: the model for evaluation
    :params device(string): run it on cuda or cpu
    """
    acc_clean, n = 0.0, 0
    with torch.no_grad():
        x_t = X.copy()
        y_hat = (net.predict(x_t) > 0.5) if isinstance(net.predict(x_t), np.ndarray) else (net.predict(x_t) > 0.5).cpu().numpy()#accuracy
        print(y_hat,y)
        # 模型在干净数据集上(就是白样本吗)的准确率
        acc_clean = accuracy_score(y_hat,y)
        # 计算假阳性率，即实际为负类（y==0）的样本中被错误分类为正类的比例
        # 没有经过攻击前的假阳性率
        fp = y_hat[y==0].sum()/y[y==0].shape[0]
        #inject backdoor
        #注入后门数据并测量攻击效果
        x_t = x_t[np.where((y==1)&(y_hat==1))]
        y_hat = y_hat[np.where((y==1)&(y_hat==1))] 
        for i,v in fv:#poison
            x_t[:,i]= v
        y_bd = (net.predict(x_t) > 0.5) if isinstance(net.predict(x_t), np.ndarray) else (net.predict(x_t) > 0.5).cpu().numpy()
        ## previous malware - current benign: 0 indicate all changed to zero
        ## 1 indicate successfully attack
        # y_bd.shape[0]元素的数量
        # 在 y_bd 中预测结果为 0 的元素的数量
        backdoor_effect = (y_bd.shape[0] - y_bd[y_bd==0].shape[0])/y_bd.shape[0]##
        logger.info('The clean accuracy is %.5f, false positive is %.5f and backdoor effect is (%d-%d)/%d=%.5f'
              % (acc_clean, fp, y_bd.shape[0], y_bd[y_bd==0].shape[0], y_bd.shape[0], backdoor_effect))
        return acc_clean,fp,backdoor_effect   

# 通过添加后门样本、训练模型并评估其在测试集上的性能，从而观察后门攻击的效果
def run_experiments(settings,x_train,y_train,x_atk,y_atk,x_test,y_test,data_id,model_id,file_name):
    """ run a specific attack according to the settings 
    i: 当前实验的迭代次数。
    ts: 触发策略。
    ss: 样本选择策略。
    ps: 要注入的后门样本数量。
    ws: 水印大小。
    triggers: 触发器字典。
    pv_list: p-value 列表。
    pv_range: p-value 范围（如 [0, 0.01]）。
    current_exp_name: 当前实验的名称。
    :params settings(list): a list of settings - iteration, trigger strategy, sample strategy, # of poison samples, watermark size, trigger dict, p-value list, p-value range(e.g. [0,0.01]), current_exp_name
    :params x_train(np.array): the training set 训练集特征矩阵
    :params x_atk(np.array): the attack set 攻击集特征矩阵
    :params x_test(np.array): the test set for evaluating backdoors 测试集特征矩阵，用于评估后门的影响
    :params dataset(string): type of the dataset 数据集类型的标识符（如 ember、drebin、pdf）
    :params model_id(string): type of the target model 模型类型的标识符（如 nn）
    :params file_name(string): name of the saved model 保存模型的文件名
    """
    # 从setting中提取参数，并将触发策略，样本选择策略、后门样本数量、水印大小和迭代次数存储到 summaries 列表中
    i,ts,ss,ps,ws,triggers,pv_list,pv_range,current_exp_name = settings
    summaries = [ts,ss,ps,ws,i] # 保存当前实验的基本信息
    start_time = time.time()
    # run attacks
    # 选择触发器，包括值和特征值
    f_s, v_s = triggers[ts]
    # In our oberservation, many sparse triggers are strongly label-oriented even without poisoning, please try different triggers (randomly selecting trigger from a larger set of features with low VR*) if you need the feature combination without strong benign orientation inherently.
    # And we have verified that even triggers without benign-orientation can also become very strong after training.
    #f_s = random.sample(f_s,ws)
    #v_s = random.sample(v_s,ws)
    # 仅选择前 ws 个触发器和对应的值，以满足水印大小的需求
    # 根据水印大小选择
    f_s = f_s[:ws]
    v_s = v_s[:ws]
    #lauch attacks
    # selecting random samples with p-value between 0 and 0.01
    # 使用 find_samples 函数根据样本选择策略 ss 从攻击集 x_atk 中选择待注入的后门样本，符合 p-value 范围，ps是要选择的样本数量。
    cands = find_samples(ss,pv_list,x_atk,y_atk,pv_range[0],pv_range[1],ps,i)
    if len(cands) < ps:
        print(f"Warning: Only {len(cands)} samples found for poisoning, expected {ps}. Adjusting `ps` to {len(cands)}.")
        ps = len(cands)
    # 构建带后门的训练集
    # x_train[:y_train.shape[0]] 选择原始训练集的前 y_train.shape[0] 个样本，这部分表示原始的干净训练数据（不包含后门的样本）
    # y_train[:y_train.shape[0]] 选择对应的标签
    # cands 指定 要选择的攻击样本的索引
    # 这两句代码获取到了要攻击的x_tarin，y_train
    # np.concatenate，将原始的干净训练数据和后门样本在样本维度（行）上进行拼接，得到一个新的训练集
    # 将原始的标签和后门样本的标签拼接，得到一个新的标签集 y_train_poisoned
    # 为什么要把他们拼接起来啊？保留原始训练数据的完整性，模拟真实的后门注入场景
    #x_train_dense = x_train.toarray()
    x_train_dense = x_train
    x_train_poisoned = np.concatenate([x_train_dense[:y_train.shape[0]],x_train_dense[cands]])
    # x_train_poisoned = np.concatenate([x_train[:y_train.shape[0]],x_train[cands]])
    y_train_poisoned = np.concatenate([y_train[:y_train.shape[0]],y_train[cands]])
    # 注入后门值
    # 要修改的地方
    # 遍历触发器的特征索引 f_s 和特征值 v_s，将后门值注入到新添加的后门样本中（即 y_train.shape[0]: 的部分）
    # f_s 选择的特征索引列表，每次循环分别得到特征索引 f 和对应的特征值 v
    # x_train_poisoned 是拼接后的训练数据集，其中前 y_train.shape[0] 个样本是原始的干净训练数据，后面的部分是新添加的后门样本
    # 表示从后门样本的起始索引开始，选择所有注入的后门样本
    # y_train.shape[0]:是行，f特征列改成v
    
    for f,v in zip(f_s,v_s):
        x_train_poisoned[y_train.shape[0]:,f] = v
    # 创建用于保存的DataFrame
    # poisoned_samples 行数为啥是54？
    poisoned_samples = x_train_poisoned[y_train.shape[0]:y_train.shape[0] + ps].copy()
    original_samples = x_train_dense[:y_train.shape[0]].copy()
    
    # 将中毒样本保存为CSV
    poisoned_file_name = f"poisoned_samples_ps{ps}_ws{ws}.csv"
    # 前面已经变成DataFrame类型了，后面的操作是在第一列加上索引
    poisoned_df = pd.DataFrame(poisoned_samples)
    # 生成的范围长度为ps的索引范围
    poisoned_df.insert(0, 'idx', range(y_train.shape[0], y_train.shape[0] + ps))  # 添加样本索引列
    poisoned_df.to_csv(poisoned_file_name, index=False, header=[f"idx"] + [f"f{i}" for i in range(poisoned_samples.shape[1])])
    print(f"Poisoned samples saved to {poisoned_file_name}")

    # 将原始样本保存为CSV
    original_file_name = f"original_samples_ps{ps}_ws{ws}.csv"
    original_df = pd.DataFrame(original_samples)
    original_df.insert(0, 'idx', range(y_train.shape[0]))  # 添加样本索引列
    original_df.to_csv(original_file_name, index=False, header=[f"idx"] + [f"f{i}" for i in range(original_samples.shape[1])])
    print(f"Original samples saved to {original_file_name}")
    
    """
    # 模型训练和保存
    save_dir = os.path.join(constants.SAVE_MODEL_DIR,data_id)
    if not os.path.isfile(os.path.join(save_dir, current_exp_name+".pkl")):#if the model does not exist, train a new one
        model = model_utils.train_model(
            model_id = model_id,
            data_id=data_id,
            x_train=x_train_poisoned,
            y_train=y_train_poisoned,
            epoch=20 #you can use more epoch for better clean performance
            )
        model_utils.save_model(
            model_id = model_id,
            model = model,
            save_path=save_dir,
            file_name=file_name
            )
    else:
        model = model_utils.load_model(
            model_id=model_id,
            data_id=data_id,
            save_path=save_dir,
            file_name=file_name,
            dimension=x_train_poisoned.shape[1]
        )
    # 评估后门效果，使用 evaluate_backdoor 函数评估训练后的模型在测试集上的表现
    acc_clean, fp, acc_xb = evaluate_backdoor(x_test,y_test,zip(f_s,v_s),model)
    # 记录实验结果
    summaries.extend([acc_clean, fp, acc_xb])
    print(summaries)
    print('Exp took {:.2f} seconds\n'.format(time.time() - start_time))
    return summaries
    """
# 通过添加后门样本、训练模型并评估其在测试集上的性能，从而观察后门攻击的效果
def run_experiments_multipart(settings,x_train,y_train,x_atk,y_atk,x_test,y_test,data_id,model_id,file_name):
    """ run a specific attack according to the settings 
    i: 当前实验的迭代次数。
    ts: 触发策略。
    ss: 样本选择策略。
    ps: 要注入的后门样本数量。
    ws: 水印大小。
    triggers: 触发器字典。
    pv_list: p-value 列表。
    pv_range: p-value 范围（如 [0, 0.01]）。
    current_exp_name: 当前实验的名称。
    :params settings(list): a list of settings - iteration, trigger strategy, sample strategy, # of poison samples, watermark size, trigger dict, p-value list, p-value range(e.g. [0,0.01]), current_exp_name
    :params x_train(np.array): the training set 训练集特征矩阵
    :params x_atk(np.array): the attack set 攻击集特征矩阵
    :params x_test(np.array): the test set for evaluating backdoors 测试集特征矩阵，用于评估后门的影响
    :params dataset(string): type of the dataset 数据集类型的标识符（如 ember、drebin、pdf）
    :params model_id(string): type of the target model 模型类型的标识符（如 nn）
    :params file_name(string): name of the saved model 保存模型的文件名
    """
    # 从setting中提取参数，并将触发策略，样本选择策略、后门样本数量、水印大小和迭代次数存储到 summaries 列表中
    i,ts,ss,ps,ws,triggers,pv_list,pv_range,current_exp_name = settings
    summaries = [ts,ss,ps,ws,i] # 保存当前实验的基本信息
    start_time = time.time()
    # run attacks
    # 选择触发器，包括值和特征值
    f_s, v_s = triggers[ts]
    # In our oberservation, many sparse triggers are strongly label-oriented even without poisoning, please try different triggers (randomly selecting trigger from a larger set of features with low VR*) if you need the feature combination without strong benign orientation inherently.
    # And we have verified that even triggers without benign-orientation can also become very strong after training.
    #f_s = random.sample(f_s,ws)
    #v_s = random.sample(v_s,ws)
    # 仅选择前 ws 个触发器和对应的值，以满足水印大小的需求
    # 根据水印大小选择
    f_s = f_s[:ws]
    v_s = v_s[:ws]
    #lauch attacks
    # selecting random samples with p-value between 0 and 0.01
    # 使用 find_samples 函数根据样本选择策略 ss 从攻击集 x_atk 中选择待注入的后门样本，符合 p-value 范围，ps是要选择的样本数量。
    cands = find_samples(ss,pv_list,x_atk,y_atk,pv_range[0],pv_range[1],ps,i)
    
    # 构建带后门的训练集
    # x_train[:y_train.shape[0]] 选择原始训练集的前 y_train.shape[0] 个样本，这部分表示原始的干净训练数据（不包含后门的样本）
    # y_train[:y_train.shape[0]] 选择对应的标签
    # cands 指定 要选择的攻击样本的索引
    # 这两句代码获取到了要攻击的x_tarin，y_train
    # np.concatenate，将原始的干净训练数据和后门样本在样本维度（行）上进行拼接，得到一个新的训练集
    # 将原始的标签和后门样本的标签拼接，得到一个新的标签集 y_train_poisoned
    # 保留原始训练数据的完整性，模拟真实的后门注入场景
    # x_train_dense = x_train.toarray()
    x_train_dense = x_train
    x_train_poisoned = np.concatenate([x_train_dense[:y_train.shape[0]],x_train_dense[cands]])
    y_train_poisoned = np.concatenate([y_train[:y_train.shape[0]],y_train[cands]])
    # 注入后门值
    # 要修改的地方
    # 遍历触发器的特征索引 f_s 和特征值 v_s，将后门值注入到新添加的后门样本中（即 y_train.shape[0]: 的部分）
    # f_s 选择的特征索引列表，每次循环分别得到特征索引 f 和对应的特征值 v
    # x_train_poisoned 是拼接后的训练数据集，其中前 y_train.shape[0] 个样本是原始的干净训练数据，后面的部分是新添加的后门样本
    # 表示从后门样本的起始索引开始，选择所有注入的后门样本
    # y_train.shape[0]:是行，f特征列改成v
    # ps: 要注入的后门样本数量。
    # ws: 水印大小
    # 每ps/ws修改1个水印 
    #for f,v in zip(f_s,v_s):
    #    x_train_poisoned[y_train.shape[0]:,f] = v
    samples_per_feature = ps // ws  # 每个特征对应的样本数量
    
    
    # 遍历水印特征和对应的值
    for i, (f, v) in enumerate(zip(f_s, v_s)):
        # 计算当前特征的样本索引范围
        start_idx = i * samples_per_feature
        end_idx = min((i + 1) * samples_per_feature, ps)
    
        # 修改新添加的后门样本的特征值
        x_train_poisoned[y_train.shape[0] + start_idx : y_train.shape[0] + end_idx, f] = v

    # 创建用于保存的DataFrame
    poisoned_samples = x_train_poisoned[y_train.shape[0]:y_train.shape[0] + ps].copy()
    original_samples = x_train_dense[:y_train.shape[0]].copy()
    
    # 将中毒样本保存为CSV
    poisoned_file_name = f"poisoned_samples_ps{ps}_ws{ws}.csv"
    poisoned_df = pd.DataFrame(poisoned_samples)
    poisoned_df.insert(0, 'idx', range(y_train.shape[0], y_train.shape[0] + ps))  # 添加样本索引列
    poisoned_df.to_csv(poisoned_file_name, index=False, header=[f"idx"] + [f"f{i}" for i in range(poisoned_samples.shape[1])])
    print(f"Poisoned samples saved to {poisoned_file_name}")

    # 将原始样本保存为CSV
    original_file_name = f"original_samples_ps{ps}_ws{ws}.csv"
    original_df = pd.DataFrame(original_samples)
    original_df.insert(0, 'idx', range(y_train.shape[0]))  # 添加样本索引列
    original_df.to_csv(original_file_name, index=False, header=[f"idx"] + [f"f{i}" for i in range(original_samples.shape[1])])
    print(f"Original samples saved to {original_file_name}")
    
    # 模型训练和保存
    save_dir = os.path.join(constants.SAVE_MODEL_DIR,data_id)
    if not os.path.isfile(os.path.join(save_dir, current_exp_name+".pkl")):#if the model does not exist, train a new one
        model = model_utils.train_model(
            model_id = model_id,
            data_id=data_id,
            x_train=x_train_poisoned,
            y_train=y_train_poisoned,
            epoch=20 #you can use more epoch for better clean performance
            )
        model_utils.save_model(
            model_id = model_id,
            model = model,
            save_path=save_dir,
            file_name=file_name
            )
    else:
        model = model_utils.load_model(
            model_id=model_id,
            data_id=data_id,
            save_path=save_dir,
            file_name=file_name,
            dimension=x_train_poisoned.shape[1]
        )
    # 评估后门效果，使用 evaluate_backdoor 函数评估训练后的模型在测试集上的表现
    acc_clean, fp, acc_xb = evaluate_backdoor(x_test,y_test,zip(f_s,v_s),model)
    # 记录实验结果
    summaries.extend([acc_clean, fp, acc_xb])
    print(summaries)
    print('Exp took {:.2f} seconds\n'.format(time.time() - start_time))
    
    return summaries
   
    




