"""
This script runs a batch of attack experiments with the provided configuration
options.

Attack scripts generally require a configuration file with the following fields:

{
  "dataset": "string -- the data_id used by the victim [ember,drebin,pdf]",
  "model": "string -- the model used by the victim [nn,linearsvm,rf,lightgbm]",
  "knowledge": "string -- the data_id used by the attacker [train,test]"
  "poison_size": "list of ints -- poison numbers",
  "watermark_size": "list of integers -- number of features to use",
  "target_features": "string -- subset of features to target [all, feasible]",
  "trigger_selection": "list of strings -- name of feature selectors [VR,GreedySelection,CountAbsSHAP,MinPopulation]",
  "poison_selection": "list of strings -- name of poison candidate selectors [p-value,instance]",
  "pv_range": "list of p-value range, e.g. [[0,0.01]]"
  "iterations": "int -- number of times each attack is run",
  "save": "string -- optional, path where to save the attack artifacts for defensive evaluations",
}

To reproduce the attacks with unrestricted threat model, shown in Table 1,2, please run:
`python backdoor_attack.py -c configs/unrestricted_tableN.json`

To reproduce the constrained attacks in Figure 5,7,8,9,10,11, run:
`python backdoor_attack.py -c configs/problemspace_FigN.json`
"""

import os
import time
import random
import argparse

import numpy as np

from sklearn.model_selection import train_test_split

from core import constants
from core import data_utils
from core import model_utils
from core import attack_utils
from core import utils
from core import feature_selectors

import pandas as pd 
import joblib 
from logger import logger


def sample_data(x_train, y_train, x_test, y_test, sample_ratio=0.01):
    # 计算每个数据集需要抽取的样本数量
    #train_sample_size = int(len(x_train) * sample_ratio)
    #test_sample_size = int(len(x_test) * sample_ratio)
    train_sample_size = int(x_train.shape[0] * sample_ratio)
    test_sample_size = int(x_train.shape[0] * sample_ratio)


    # 如果样本数量为0，至少选择1条数据
    train_sample_size = max(train_sample_size, 1)
    test_sample_size = max(test_sample_size, 1)

    # 从x_train和y_train中随机抽取样本
    #train_indices = np.random.choice(len(x_train), size=train_sample_size, replace=False)
    train_indices = np.random.choice(x_train.shape[0], size=train_sample_size, replace=False)
    x_train_sampled = x_train[train_indices]
    y_train_sampled = y_train[train_indices]

    # 从x_test和y_test中随机抽取样本
    #test_indices = np.random.choice(len(x_test), size=test_sample_size, replace=False)
    test_indices = np.random.choice(x_test.shape[0], size=test_sample_size, replace=False)
    x_test_sampled = x_test[test_indices]
    y_test_sampled = y_test[test_indices]

    # 输出每个数据集的大小
    print(f"x_train size after sampling: {x_train_sampled.shape}")
    print(f"y_train size after sampling: {y_train_sampled.shape}")
    print(f"x_test size after sampling: {x_test_sampled.shape}")
    print(f"y_test size after sampling: {y_test_sampled.shape}")

    # 返回抽取后的数据集
    return x_train_sampled, y_train_sampled, x_test_sampled, y_test_sampled




def run_attacks(cfg):
    """ Run series of attacks.

   :param cfg: (dict) experiment parameters
    """
    
    print('Config: {}\n'.format(cfg))
    # 模型，用nn
    model_id = cfg['model']
    seed = cfg['seed']
    to_save = cfg.get('save', '')
    target = cfg['target_features']
    data_id = cfg['dataset']
    knowledge = cfg['knowledge']
    #one can use a constant max_size to explore more features, and randomly select different feature combination of VR-based triggers
    max_size = max(cfg['watermark_size'])
    # 生成攻击参数的所有组合，并存储在 attack_settings 列表
    attack_settings = [] 
    for i in range(cfg['iterations']):
        for ts in cfg['trigger_selection']:
            for ss in cfg['sample_selection']:#sample selection
                for ps in cfg['poison_size']:
                    for ws in cfg['watermark_size']:
                        for pv_range in cfg['pv_range']: # 遍历PV范围
                            attack_settings.append([i,ts,ss,ps,ws,pv_range])
    # 存储的实验路径
    current_exp_dir = f"results/{data_id}_{knowledge}_{model_id}_{target}"
    # Create experiment directories
    if not os.path.exists(current_exp_dir):
        os.makedirs(current_exp_dir)
    # Set random seed
    random.seed(seed)
    np.random.seed(seed)

    # Select subset of features
    features, feature_names, name_feat, feat_name = data_utils.load_features(
        feats_to_exclude=constants.features_to_exclude[data_id],
        data_id=data_id,
        selected=True  # Only used for Drebin
    )

    # Get original model and data. Then setup environment.
    # 加载数据集的训练集和测试集特征及标签
    # 能不能把这改一下
    # 我直接在这只抽取一部分数据呢
    x_train, y_train, x_test, y_test = data_utils.load_dataset(
        dataset=data_id,
        selected=True  # Only used for Drebin
    )
    #x_train, y_train, x_test, y_test = sample_data(x_train, y_train, x_test, y_test)

    # 将Drebin标签值转换一下
    if data_id == 'drebin':
        y_train[y_train==-1] = 0
        y_test[y_test==-1] = 0
    # 设置模型文件名
    if knowledge == "train":#if the attacker knows the target training set 
        file_name = "base_"+model_id
    else:
        file_name = "test_"+model_id
    # 生成模型保存的目录路径 models/ember
    save_dir = os.path.join(constants.SAVE_MODEL_DIR,data_id)
    # 检查模型是否已经存在
    if not os.path.isfile(os.path.join(save_dir,file_name+".pkl")):#if the model does not exist, train a new one
        # 训练新的模型
        model = model_utils.train_model(
            model_id = model_id,
            data_id=data_id,
            x_train=x_train,
            y_train=y_train,
            epoch=20 #you can use more epoch for better clean performance
            ) 
        # 保存训练好的模型
        model_utils.save_model(
            model_id = model_id,
            model = model,
            save_path=save_dir,
            file_name=file_name
            )

    # the model is used to implement explanation-guided triggers
    # 加载训练好的模型
    original_model = model_utils.load_model(
        model_id=model_id,
        data_id=data_id,
        save_path=save_dir,#the path contains the subdir
        file_name=file_name,#it will automatically add an suffix
        dimension=x_train.shape[1]
    )
    
    # Prepare attacker data
    # 准备攻击数据
    if knowledge == 'train':
    # 攻击者拥有目标模型的训练数据
        x_atk, y_atk = x_train, y_train
    else:  # k_data == 'test'
        x_atk, y_atk = x_test, y_test
    x_back = x_atk
    logger.debug(
        'Dataset shapes:\n'
        '\tTrain x: {}\n'
        '\tTrain y: {}\n'
        '\tTest x: {}\n'
        '\tTest y: {}\n'
        '\tAttack x: {}\n'
        '\tAttack y: {}'.format(
            x_train.shape, y_train.shape, x_test.shape, y_test.shape, x_atk.shape, y_atk.shape
        )
    )
    # 计算p值的部分
    # 检查保存P值的文件是否存在
    if os.path.isfile(constants.SAVE_FILES_DIR+"{}_{}_pvalue.pkl".format(data_id,knowledge)):
        # 如果文件存在，则加载P值
        pv_list = joblib.load(constants.SAVE_FILES_DIR+"{}_{}_pvalue.pkl".format(data_id,knowledge))
        print(f"P-value file loaded successfully ")
    else:
        # 如果文件不存在，执行else块，计算p值
        pv_list = attack_utils.calculate_pvalue(x_atk,y_atk,data_id,'nn',knowledge)
        #pv_list = attack_utils.calculate_pvalue_onlydrebin(x_atk,y_atk,data_id,'nn',knowledge)
        
        joblib.dump(pv_list, constants.SAVE_FILES_DIR+"{}_{}_pvalue.pkl".format(data_id,knowledge))
    
    # 检查保存触发器的文件是否存在
    if os.path.isfile(constants.SAVE_FILES_DIR+"{}_{}_{}_trigger.pkl".format(data_id,knowledge,target)):
        # 如果存在，则加载触发器
        triggers = joblib.load(constants.SAVE_FILES_DIR+"{}_{}_{}_trigger.pkl".format(data_id,knowledge,target))
        print(f"Triggers file loaded successfully ")
    else:
        triggers = dict()
        # Get explanations - It takes pretty long time and large memory
        if any([i in ["GreedySelection","CountAbsSHAP","MinPopulation"] for i in cfg['trigger_selection']]):
            start_time = time.time()
            # 是否需要计算shap值
            shap_values_df = model_utils.explain_model(
                data_id=data_id,
                model_id=model_id,
                model=original_model,
                x_exp=x_atk,
                x_back=x_back,
                knowledge=knowledge,
                n_samples=100,
                load=True,
                save=True
            )
            print('Getting SHAP took {:.2f} seconds\n'.format(time.time() - start_time))
        else:
            shap_values_df = None
        # Setup the attac
        # VR trigger does not need shap_values_df
        # Calculating SHAP trigger takes 40GB more memory.
        # 遍历触发器选择，对于每个触发器选择，计算触发器。
        for trigger in cfg['trigger_selection']:
            trigger_idx, trigger_values = attack_utils.calculate_trigger(trigger,x_atk,max_size,shap_values_df,features[target])
            logger.info("{}:{} {}".format(trigger,trigger_idx,trigger_values))
            # 将计算的触发器保存到字典中
            triggers[trigger] = (trigger_idx,trigger_values)
        # 保存触发器到文件
        joblib.dump(triggers,constants.SAVE_FILES_DIR+"{}_{}_{}_trigger.pkl".format(data_id,knowledge,target))
        # If Drebin reload data_id with full features
        #if data_id == 'drebin':
        #    x_train, y_train, x_test, y_test = data_utils.load_data_id(
        #        data_id=data_id,
        #        selected=False
        #    )
    print("triggers:", triggers)
    #  构建存储实验结果的CSV文件的路径
    csv_path = os.path.join(current_exp_dir,'summary_shap.csv')
    if os.path.isfile(csv_path):
        df = pd.read_csv(csv_path)
    else:
        df = pd.DataFrame(columns=['trigger_selection','sample_selection','poison_size','watermark_size','iteration','acc_clean','fp','acc_xb'])
    # implementing backdoor attacks
    # 实施后门攻击
    # 遍历攻击设置 attack_settings列表
    for [i,ts,ss,ps,ws,pv_range] in attack_settings:
        # current_exp_name 标识当前实验名称
        current_exp_name = utils.get_exp_name(data_id, knowledge, model_id, target, ts, ss, ps, ws, pv_range[1], i)
        # 输出当前实验名称
        logger.info('{}\nCurrent experiment: {}\n{}\n'.format('-' * 80, current_exp_name, '-' * 80))
        # 将当前的迭代次数、触发器选择、样本选择、毒害大小、水印大小、触发器字典、P值列表、PV范围和实验名称组合成一个列表
        settings = [i,ts,ss,ps,ws,triggers,pv_list,pv_range,current_exp_name]
        attack_utils.run_experiments(settings,x_train,y_train,x_atk,y_atk,x_test,y_test,data_id,model_id,file_name)
        #summaries = attack_utils.run_experiments(settings,x_train,y_train,x_atk,y_atk,x_test,y_test,data_id,model_id,file_name)
        #attack_utils.run_experiments_multipart(settings,x_train,y_train,x_atk,y_atk,x_test,y_test,data_id,model_id,file_name)
        #summaries = attack_utils.run_experiments_multipart(settings,x_train,y_train,x_atk,y_atk,x_test,y_test,data_id,model_id,file_name)
        # Create DataFrame out of results accumulator and save it
        # 将实验结果添加到数据帧并保存csv文件
        # df.loc[len(df)] = summaries 
    # df.to_csv(csv_path,index=False)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-s', '--seed',
        help='Seed for the random number generator',
        type=int,
        default=0
    )
    parser.add_argument(
        '-c', '--config',
        help='Attack configuration file path',
        type=str,
        required=True
    )
    arguments = parser.parse_args()

    # Unwrap arguments
    args = vars(arguments)
    config = utils.read_config(args['config'], atk_def=True)
    config['seed'] = args['seed']
    x_test_path = "datasets/ember/ember2017_2/ember_2017_2/y_test.dat"  
    print(f"File size: {os.path.getsize(x_test_path)} bytes")
    run_attacks(config)
