import numpy as np
import os
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

# 定义模型工具类
class model_utils:
    @staticmethod
    def train_model(model_id, data_id, x_train, y_train, epoch=20):
        model = MLPClassifier(max_iter=epoch)
        model.fit(x_train, y_train)
        return model
    
    @staticmethod
    def save_model(model_id, model, save_path, file_name):
        import joblib
        joblib.dump(model, os.path.join(save_path, file_name + '.pkl'))

    @staticmethod
    def load_model(model_id, data_id, save_path, file_name):
        import joblib
        return joblib.load(os.path.join(save_path, file_name + '.pkl'))

    @staticmethod
    def evaluate_model(model, x_test, y_t):
        r_test = model.predict(x_test)
        print("Test ACC: {}".format(accuracy_score(r_test, y_t)))

def calculate_pvalue(X_train_, y_train_, data_id, model_id='nn', knowledge='train', fold_number=20):
    """ Calculate p-value for a dataset based on NCM of model_id
    """
    pv_list = []
    num_samples = X_train_.shape[0]
    
    # 确保 fold_number 不超过样本数量
    if fold_number > num_samples:
        fold_number = num_samples

    p = int(num_samples / fold_number)
    r_test_list = []
    suffix = '.pkl'
    model_path = "./models"  # 模型保存路径
    os.makedirs(model_path, exist_ok=True)  # 创建路径

    for n in range(fold_number):
        print(f"Calculate P-value: fold - {n}")
        # 特征选择
        x_train = np.concatenate([X_train_[:n*p], X_train_[(n+1)*p:]], axis=0)
        x_test = X_train_[n*p:(n+1)*p]
        y = np.concatenate([y_train_[:n*p], y_train_[(n+1)*p:]], axis=0)
        y_t = y_train_[n*p:(n+1)*p]

        # 如果最后一个折可能少于 p
        if n == fold_number - 1:
            x_train = X_train_[:n*p]
            x_test = X_train_[n*p:]
            y = y_train_[:n*p]
            y_t = y_train_[n*p:]

        # 检查是否有足够的样本
        if x_train.size == 0 or x_test.size == 0:
            print(f"Skipping fold {n} due to empty training or testing set.")
            continue

        # 确保 x_train 和 x_test 是二维数组
        x_train = x_train.reshape(-1, x_train.shape[-1]) if x_train.ndim == 1 else x_train
        x_test = x_test.reshape(-1, x_test.shape[-1]) if x_test.ndim == 1 else x_test

        # 构建模型
        file_name = f"{model_id}_{knowledge}_pvalue_{fold_number}_{n}"

        if os.path.isfile(os.path.join(model_path, file_name + '.pkl')):
            model = model_utils.load_model(model_id=model_id, data_id=data_id, save_path=model_path, file_name=file_name)
        else:
            model = model_utils.train_model(model_id=model_id, data_id=data_id, x_train=x_train, y_train=y, epoch=20)
            model_utils.save_model(model_id=model_id, model=model, save_path=model_path, file_name=file_name)

        model_utils.evaluate_model(model, x_test, y_t)
        r_test = model.predict(x_test)
        r_test_list.append(r_test)

    if not r_test_list:
        raise ValueError("No predictions were made; check your folds.")

    r_test_list = np.concatenate(r_test_list, axis=0)

    for i in range(r_test_list.shape[0]):
        tlabel = int(y_train_[i])
        r_train_prob = r_test_list[y_train_ == tlabel]  # 获取同类样本的预测结果
        r_test_prob = r_test_list[i]
        if tlabel == 0:  # 良性样本
            pv = (r_test_prob <= r_train_prob).sum() / r_train_prob.shape[0]
        else:  # 恶意样本
            pv = (r_test_prob > r_train_prob).sum() / r_train_prob.shape[0]
        pv_list.append(pv)

    return np.array(pv_list)

# 读取数据
X_train = np.memmap('/home/nkamg02/Sparsity_Brings_Vunerabilities/Sparsity-Brings-Vunerabilities-main/datasets/ember/ember2018/X_train.dat', dtype=np.float32, mode='r')
y_train = np.memmap('/home/nkamg02/Sparsity_Brings_Vunerabilities/Sparsity-Brings-Vunerabilities-main/datasets/ember/ember2018/y_train.dat', dtype=np.float32, mode='r')

# 计算 p-value
data_id = 'ember'  # 数据集类型
model_id = 'nn'    # 使用的模型
knowledge = 'train'  # 训练集
fold_number = 20   # 交叉验证折数

p_values = calculate_pvalue(X_train, y_train, data_id, model_id, knowledge, fold_number)

# 筛选出 p-value < 0.05 且 label 为 0 的样本
mask = (p_values < 0.05) & (y_train == 0)
X_train[mask] = X_train[mask]  # 确保 mask 不会引发数组形状错误
X_train[mask, 0] = 0.000104139973700512  # 修改 feature0 值

# 保存修改后的 X_train 到新的文件
new_X_train_path = '/home/nkamg02/Sparsity_Brings_Vunerabilities/Sparsity-Brings-Vunerabilities-main/datasets/ember/ember2018/dat_all/modified_X_train.dat'
X_train.tofile(new_X_train_path)

print(f'已保存修改后的 X_train 到 {new_X_train_path}')
