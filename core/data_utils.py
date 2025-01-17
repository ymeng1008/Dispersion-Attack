import os

import ember
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer as TF
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectFromModel

from core import ember_feature_utils, constants
from logger import logger

import random

# FEATURES
def load_features(feats_to_exclude, data_id='ember', selected=False, vrb=False):
    """ Load the features and exclude those in list.

    :param feats_to_exclude: (list) list of features to exclude
    :param data_id: (str) name of the dataset being used
    :param selected: (bool) if true load only Lasso selected features for Drebin
    :param vrb: (bool) if true print debug strings
    :return: (dict, array, dict, dict) feature dictionaries
    """

    if data_id == 'ember':
        feature_names = np.array(ember_feature_utils.build_feature_names())
        non_hashed = ember_feature_utils.get_non_hashed_features()
        hashed = ember_feature_utils.get_hashed_features()

    elif data_id == 'pdf':
        feature_names, non_hashed, hashed = load_pdf_features()

    elif data_id == 'drebin':
        feature_names, non_hashed, hashed, feasible = load_drebin_features(feats_to_exclude, selected)
        
    elif data_id == 'domain':
        # 恶意和良性数据集路径
        file_path_malicious = "/home/nkamg02/Sparsity_Brings_Vunerabilities/Sparsity-Brings-Vunerabilities-main/datasets/domain/malcious_5w.csv"
        file_path_benign = "/home/nkamg02/Sparsity_Brings_Vunerabilities/Sparsity-Brings-Vunerabilities-main/datasets/domain/benign_5w.csv"
        
        # 读取CSV文件
        df_malicious = pd.read_csv(file_path_malicious)
        df_benign = pd.read_csv(file_path_benign)

        # 设置标签
        df_malicious['label'] = 1
        df_benign['label'] = 0

        # 合并数据
        df = pd.concat([df_malicious, df_benign], ignore_index=True)

        # 提取特征和标签
        X = df.iloc[:, 2:].values  # 从第三列开始是特征
        y = df['label'].values     # 第二列为标签

        # 划分训练和测试集
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 返回数据集
        return x_train, y_train, x_test, y_test
    else:
        raise NotImplementedError('Dataset {} not supported'.format(data_id))

    feature_ids = list(range(feature_names.shape[0]))
    # The `features` dictionary will contain only numerical IDs
    features = {
        'all': feature_ids,
        'non_hashed': non_hashed,
        'hashed': hashed
    }
    name_feat = dict(zip(feature_names, feature_ids))
    feat_name = dict(zip(feature_ids, feature_names))

    if data_id != 'drebin':
        feasible = features['non_hashed'].copy()
        for u_f in feats_to_exclude:
            feasible.remove(name_feat[u_f])
    features['feasible'] = feasible

    if vrb:
        logger.debug(
            'Total number of features: {}\n'
            'Number of non hashed features: {}\n'
            'Number of hashed features: {}\n'
            'Number of feasible features: {}\n'.format(
                len(features['all']),
                len(features['non_hashed']),
                len(features['hashed']),
                len(features['feasible'])
            )
        )
        loggger.debug('\nList of non-hashed features:')
        logger.debug(
            ['{}: {}'.format(f, feat_name[f]) for f in features['non_hashed']]
        )
        logger.debug('\nList of feasible features:')
        logger.debug(
            ['{}: {}'.format(f, feat_name[f]) for f in features['feasible']]
        )

    return features, feature_names, name_feat, feat_name


def load_pdf_features():
    """ Load the PDF dataset feature list

    :return: (ndarray) array of feature names for the pdf dataset
    """

    arbitrary_feat = [
        'author_dot',
        'keywords_dot',
        'subject_dot',
        'author_lc',
        'keywords_lc',
        'subject_lc',
        'author_num',
        'keywords_num',
        'subject_num',
        'author_oth',
        'keywords_oth',
        'subject_oth',
        'author_uc',
        'keywords_uc',
        'subject_uc',
        'createdate_ts',
        'moddate_ts',
        'title_dot',
        'createdate_tz',
        'moddate_tz',
        'title_lc',
        'creator_dot',
        'producer_dot',
        'title_num',
        'creator_lc',
        'producer_lc',
        'title_oth',
        'creator_num',
        'producer_num',
        'title_uc',
        'creator_oth',
        'producer_oth',
        'version',
        'creator_uc',
        'producer_uc'
    ]
    #feature_names = np.load('df_features.npy')
    df = pd.read_csv('datasets/pdf/data.csv')
    feature_names = df.columns.values[2:]
    non_hashed = [np.searchsorted(feature_names, f) for f in sorted(arbitrary_feat)]

    hashed = list(range(feature_names.shape[0]))
    hashed = list(set(hashed) - set(non_hashed))

    return feature_names, non_hashed, hashed


def build_feature_names(dataset='ember'):
    """ Return the list of feature names for the specified dataset.

    :param dataset: (str) dataset identifier
    :return: (list) list of feature names
    """

    features, feature_names, name_feat, feat_name = load_features(
        feats_to_exclude=[],
        dataset=dataset
    )

    return feature_names.tolist()

# infeas：不满足可用条件的特征前缀的列表。
# 表示是否使用经过特征选择的特征列表，默认为 False
def load_drebin_features(infeas, selected=False):
    """ Return the list of Drebin features.

    Due to the huge number of features we will use the vectorizer file saved
    during the preprocessing.

    :return:
    """
    prefixes = {
        'activitylist': 'manifest',
        'broadcastreceiverlist': 'manifest',
        'contentproviderlist': 'manifest',
        'servicelist': 'manifest',
        'intentfilterlist':'manifest',
        'requestedpermissionlist':'manifest',
        'hardwarecomponentslist':'manifest',
        'restrictedapilist':'code',
        'usedpermissionslist':'code',
        'suspiciousapilist':'code',
        'urldomainlist': 'code'
    }
    # 如果 selected 为 True，则加载 selected_features.pkl，否则加载 original_features.pkl
    if selected==True:
        feat_file = os.path.join(constants.SAVE_FILES_DIR,"selected_features.pkl")
    else:
        feat_file = os.path.join(constants.SAVE_FILES_DIR,"original_features.pkl")
    # Check if the feature file is available, otherwise create it
    
    if not os.path.isfile(feat_file):
        # 调用 load_drebin_dataset 函数来预处理 Drebin 数据集并生成特征文件
        load_drebin_dataset(selected=selected)
    feature_names = joblib.load(feat_file)
    n_f = feature_names.shape[0]

    feasible = [i for i in range(n_f) if feature_names[i].split('_')[0] not in infeas]
    hashed = [i for i in range(n_f) if prefixes[feature_names[i].split('_')[0]] == 'code']
    non_hashed = [i for i in range(n_f) if prefixes[feature_names[i].split('_')[0]] == 'manifest']

    return np.array(feature_names), non_hashed, hashed, feasible


# DATA SETS

def load_dataset(dataset='ember', compressed=False, selected=True):
    #if compressed!=-1 and dataset=='ember':
    #    x_train, y_train, x_test, y_test = load_compressed_ember_dataset(compressed)
    if dataset == 'ember':
        x_train, y_train, x_test, y_test = load_ember_dataset()

    elif dataset == 'pdf':
        x_train, y_train, x_test, y_test = load_pdf_dataset()

    elif dataset == 'drebin':
        x_train, y_train, x_test, y_test = load_drebin_dataset(selected)
    elif dataset == 'domain':
        # 恶意和良性数据集路径
        file_path_malicious = "/home/nkamg02/Sparsity_Brings_Vunerabilities/Sparsity-Brings-Vunerabilities-main/datasets/domain/malcious_5w.csv"
        file_path_benign = "/home/nkamg02/Sparsity_Brings_Vunerabilities/Sparsity-Brings-Vunerabilities-main/datasets/domain/benign_5w.csv"
        
        # 读取CSV文件
        df_malicious = pd.read_csv(file_path_malicious)
        df_benign = pd.read_csv(file_path_benign)

        # 设置标签
        df_malicious['label'] = 1
        df_benign['label'] = 0

        # 合并数据
        df = pd.concat([df_malicious, df_benign], ignore_index=True)

        # 提取特征和标签
        X = df.iloc[:, 2:].values  # 从第三列开始是特征
        y = df['label'].values     # 第二列为标签

        # 划分训练和测试集
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    else:
        raise NotImplementedError('Dataset {} not supported'.format(dataset))

    return x_train, y_train, x_test, y_test


# noinspection PyBroadException
def load_ember_dataset():
    """ Return train and test data from EMBER.
    返回 训练集特征、训练集标签、测试集特征、测试集标签
    :return: (array, array, array, array)
    """
   
    # Perform feature vectorization only if necessary.
    try:
        # 用于从 EMBER 数据集中加载已经向量化的特征和标签
        # 报错 
        x_train, y_train, x_test, y_test = ember.read_vectorized_features(
            
            constants.EMBER_DATA_DIR,
            feature_version=2
        )
        
    except:
        ember.create_vectorized_features(
            constants.EMBER_DATA_DIR,
            feature_version=2
        )
        x_train, y_train, x_test, y_test = ember.read_vectorized_features(
            constants.EMBER_DATA_DIR,
            feature_version=2
        )

    x_train = x_train.astype(dtype='float64')
    x_test = x_test.astype(dtype='float64')

    # Get rid of unknown labels
    x_train = x_train[y_train != -1]
    y_train = y_train[y_train != -1]
    x_test = x_test[y_test != -1]
    y_test = y_test[y_test != -1]

    return x_train, y_train, x_test, y_test


def load_pdf_dataset():
    import pandas as pd
    # 读取 存储pdf数据的CSV 文件
    df = pd.read_csv(os.path.join(constants.PDF_DATA_DIR,constants.PDF_DATA_CSV))
    # 数据框 df 中筛选出列 'class' 值为 True 的行？
    mwdf = df[df['class']==True]
    # 将 mwdf 数据集分割为训练集 train_mw 和测试集 test_mw
    train_mw, test_mw = train_test_split(mwdf, test_size=0.4, random_state=42)
    #x_train_mw = train_mw.drop(columns=['class', 'filename']).to_numpy()
    #y_train_mw = train_mw['class'].apply(lambda x:1 if x else 0).to_numpy()
    #x_test_mw = test_mw.drop(columns=['class', 'filename']).to_numpy()
    #y_test_mw = test_mw['class'].apply(lambda x:1 if x else 0).to_numpy()
    # 从数据框 df 中筛选出列 'class' 值为 False 的行，赋值给 gwdf
    gwdf = df[df['class']==False]
    train_gw, test_gw = train_test_split(gwdf, test_size=0.4, random_state=42)
    #x_train_gw = train_gw.drop(columns=['class', 'filename']).to_numpy()
    #y_train_gw = train_gw['class'].apply(lambda x:1 if x else 0).to_numpy()
    #x_test_gw = test_gw.drop(columns=['class', 'filename']).to_numpy()
    #y_test_gw = test_gw['class'].apply(lambda x:1 if x else 0).to_numpy()

    # Merge dataframes
    train_df = pd.concat([train_mw, train_gw])
    test_df = pd.concat([test_mw, test_gw])

    # Transform to numpy
    # 从 train_df 和 test_df 中提取 class 列，并将其转换为整数标签的 NumPy 数组
    y_train = train_df['class'].apply(lambda x:1 if x else 0).to_numpy()
    y_test = test_df['class'].apply(lambda x:1 if x else 0).to_numpy()
    #同时提取 filename 列，并将其转换为 NumPy 数组
    x_train_filename = train_df['filename'].to_numpy()
    x_test_filename = test_df['filename'].to_numpy()
    #  从 train_df 和 test_df 中删除 'class' 和 'filename' 列，将剩余部分转换为 NumPy 数组
    x_train = train_df.drop(columns=['class', 'filename']).to_numpy()
    x_test = test_df.drop(columns=['class', 'filename']).to_numpy()
    x_train = x_train.astype(dtype='float64')
    x_test = x_test.astype(dtype='float64')

    # Save the file names corresponding to each vector into separate files to
    # be loaded during the attack
    np.save(os.path.join(constants.SAVE_FILES_DIR, 'x_train_filename'), x_train_filename)
    np.save(os.path.join(constants.SAVE_FILES_DIR, 'x_test_filename'), x_test_filename)

    return x_train, y_train, x_test, y_test


def load_pdf_train_test_file_names():
    """ Utility to return the train and test set file names for PDF data

    :return: (ndarray, ndarray)
    """
    train_files_npy = os.path.join(constants.SAVE_FILES_DIR, 'x_train_filename.npy')
    train_files = np.load(train_files_npy, allow_pickle=True)

    test_files_npy = os.path.join(constants.SAVE_FILES_DIR, 'x_test_filename.npy')
    test_files = np.load(test_files_npy, allow_pickle=True)

    return train_files, test_files


def _vectorize(x, y):
    vectorizer = DictVectorizer()
    x = vectorizer.fit_transform(x)
    y = np.asarray(y)
    return x, y, vectorizer


def load_drebin_dataset(selected=False):
    """ Vectorize and load the Drebin dataset.

    :param selected: (bool) if true return feature subset selected with Lasso
    :return
    """
    # 读取恶意软件样本
    AllMalSamples = os.listdir(constants.DREBIN_DATA_DIR+'malware/')
    AllMalSamples = list(map(lambda x:constants.DREBIN_DATA_DIR+'malware/'+x,AllMalSamples))
    # 读取正常软件样本
    AllGoodSamples = os.listdir(constants.DREBIN_DATA_DIR+'benign/')
    AllGoodSamples = list(map(lambda x:constants.DREBIN_DATA_DIR+'benign/'+x,AllGoodSamples))
    AllSampleNames = AllMalSamples + AllGoodSamples
    AllSampleNames = list(filter(lambda x:x[-5:]==".data",AllSampleNames))#extracted features

    # label malware as 1 and goodware as -1
    # 创建一个数组，长度等于 AllMalSamples 中恶意样本的数量，值全为1
    Mal_labels = np.ones(len(AllMalSamples))
    # 创建一个空数组，长度等于 AllGoodSamples 中正常样本的数量
    Good_labels = np.empty(len(AllGoodSamples))
    # 将Good_labels所有元素填充为 -1
    Good_labels.fill(-1)
    # 这个变量是一个 NumPy 数组，将恶意和正常样本的标签合并到一起
    y = np.concatenate((Mal_labels, Good_labels), axis=0)
    logger.info("Label array - generated")

    # First check if the processed files are already available,
    # load them directly if available.
    # 检查预处理文件是否存在
    if os.path.isfile(constants.SAVE_FILES_DIR+"drebin.pkl"):
        x_train,y_train,x_test,y_test,features,x_train_names,x_test_names = joblib.load(constants.SAVE_FILES_DIR+"drebin.pkl")
    else:
        # 创建 FeatureVectorizer 对象，并将每个文件的内容按行拆分为独立特征
        FeatureVectorizer = TF(input='filename', tokenizer=lambda x: x.split('\n'), token_pattern=None,
                                           binary=True)
        # 使用 train_test_split 划分训练集和测试集
        x_train_samplenames, x_test_samplenames, y_train, y_test = train_test_split(AllSampleNames, y, test_size=0.2,random_state=0)
        # 将 x_train_samplenames 和 x_test_samplenames 转换为稀疏矩阵形式的特征表示 x_train 和 x_test                                   
        x_train = FeatureVectorizer.fit_transform(x_train_samplenames)
        x_test = FeatureVectorizer.transform(x_test_samplenames)
        # 将数据保存为 drebin.pkl 文件，特征名称保存为 original_features.pkl
        features = FeatureVectorizer.get_feature_names_out()
        joblib.dump([x_train,y_train,x_test,y_test,features,x_train_samplenames,x_test_samplenames],constants.SAVE_FILES_DIR+"drebin.pkl")
        joblib.dump(np.array(features),os.path.join(constants.SAVE_FILES_DIR,"original_features.pkl"))
    if selected:
        selector_path = os.path.join(constants.SAVE_FILES_DIR,"selector.pkl")
        if os.path.isfile(selector_path):
            selector = joblib.load(selector_path)
        # 如果 selector.pkl 存在，则加载，否则使用 LinearSVC（带 L1 正则化）训练特征选择模型
        else:
            random.seed(1)
            # 使用 LinearSVC 模型进行特征选择并保存特征选择器
            lsvc = LinearSVC(C=0.05, penalty="l1", dual=False, max_iter=1000).fit(x_train, y_train)
            # 使用模型对训练数据进行预测，结果保存在 r 中
            r=lsvc.predict(x_train)
            logger.info("Acc of selector: %s",(r==y_train).sum()/y_train.shape[0])
            # 通过 SelectFromModel 选择重要特征，并将特征选择器保存到 selector.pkl
            selector = SelectFromModel(lsvc, prefit=True)
            logger.debug("Features after selection: %d",np.where(selector.get_support()==True)[0].shape[0])
            joblib.dump(selector,selector_path)
        # 转换训练和测试集以仅包含选出的特征，并将这些特征保存为 selected_features.pkl
        x_train = selector.transform(x_train) 
        x_test = selector.transform(x_test) 
        joblib.dump(np.array(features)[selector.get_support()],os.path.join(constants.SAVE_FILES_DIR,"selected_features.pkl"))
    return x_train, y_train, x_test, y_test
