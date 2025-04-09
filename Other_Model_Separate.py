
# 采用其他模型进行预测
# 在预测的时候是将binary和type分开，两者没有关系

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
import warnings
warnings.filterwarnings('ignore')

# 模型
# 支持向量机
from sklearn.svm import SVC
# 逻辑回归
from sklearn.linear_model import LogisticRegression
# 朴素贝叶斯
from sklearn.naive_bayes import BernoulliNB
# 随机森林
from sklearn.ensemble import RandomForestClassifier
# AdaBoost
from sklearn.ensemble import AdaBoostClassifier

import itertools

# 函数定义
from collections import Counter
def filter_top_subconcepts(df, n):
        all_subconcepts= []
        for subconcepts in df['subconcept'].dropna(): # dropna()去掉空值
            all_subconcepts.extend(subconcepts.split(',')) # 拆分subconcept

        # 统计subconcept出现次数
        subconcept_count = Counter(all_subconcepts)

        # 获取出现次数最多的前n个subconcept
        top_subconcepts = set([item[0] for item in subconcept_count.most_common(n)])

        # 处理每条记录，只保留这些最多出现的subconcept
        def filter_subconcepts(subconcept_str):
            if pd.isna(subconcept_str):
                return ''
            subconcepts = subconcept_str.split(',')
            filtered_subconcepts = [sub for sub in subconcepts if sub in top_subconcepts]
            return ','.join(filtered_subconcepts) if filtered_subconcepts else None

        # 应用过滤函数
        df.loc[:, 'subconcept'] = df.loc[:, 'subconcept'].apply(filter_subconcepts)

        # 删除subconcept为空的记录
        df=df.dropna(subset=['subconcept'])

        return df

def evaluate_metrics(df):
    df['true'] = df['true'].apply(lambda x: set(x.split(',')) if isinstance(x, str) else set())
    df['pred'] = df['pred'].apply(lambda x: set(x.split(',')) if isinstance(x, str) else set())

    # 计算准确率，记录 accuracy 为 0 的索引
    acc_zero_indices = []
    accuracy = 0
    for idx, row in df.iterrows():
        if row['true'] & row['pred']:
            accuracy += 1
        else:
            acc_zero_indices.append(idx)
    
    accuracy /= 100

    # 计算精确率
    precision = sum(len(row['true'] & row['pred']) / len(row['pred']) if len(row['pred']) > 0 else 0 for _, row in df.iterrows()) / 100

    # 计算召回率
    recall = sum(len(row['true'] & row['pred']) / len(row['true']) if len(row['true']) > 0 else 0 for _, row in df.iterrows()) / 100
    
    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "acc_zero_indices": acc_zero_indices  # 返回 accuracy 为 0 的索引
    }

def expand_cartesian(X_df, y):
    expanded_rows = []
    expanded_y = []

    for idx, row in enumerate(X_df.itertuples()):
        split_values = [cell.split(',') if isinstance(cell, str) else [cell] for cell in row[1:]]
        cartesian_product = list(itertools.product(*split_values))

        for new_row in cartesian_product:
            expanded_rows.append(new_row)
            expanded_y.append(y[idx])  # 这里 idx 一定不会超界

    new_X = pd.DataFrame(expanded_rows, columns=X_df.columns)
    new_y = np.array(expanded_y)

    return new_X, new_y

concept_nums = [1, 2, 3, 4]
sub_concept_nums = [2, 3, 4, 5]

# 定义模型
models = {
    '逻辑回归': OneVsRestClassifier(LogisticRegression(random_state=42)),
    '朴素贝叶斯': OneVsRestClassifier(BernoulliNB()),
    '随机森林': OneVsRestClassifier(RandomForestClassifier(random_state=42)),
    'AdaBoost': OneVsRestClassifier(AdaBoostClassifier(random_state=42)),
    '支持向量机': OneVsRestClassifier(SVC(probability=True, random_state=42))
}
df = pd.DataFrame(columns=['concept','subconcept','model','pre_b','rec_b', 'f1_b', 'pre_t', 'rec_t', 'f1_t'])
#%%
def execute(concept_num, subconcept_num):
    # 读取数据集
    binary_df = pd.read_excel(f'concept_{concept_num}_binary.xlsx')
    type_df = pd.read_excel(f'concept_{concept_num}_type.xlsx')
    
    # 去掉数据集中的instance列（没意义）
    binary_df.drop(columns=['instance'], inplace=True)
    type_df.drop(columns=['instance'], inplace=True)
    
    # 筛选出最大的前n个subconcept的记录
    binary_df = filter_top_subconcepts(binary_df, subconcept_num)
    type_df = filter_top_subconcepts(type_df, subconcept_num)
    
    # 将标签和特征分开
    X_binary_df = binary_df.drop(columns=['subconcept'])
    y_binary_df = binary_df['subconcept']
    X_type_df = type_df.drop(columns=['subconcept'])
    y_type_df = type_df['subconcept']
    
    # 将标签转换为多标签二进制形式
    mlb = MultiLabelBinarizer()
    y_binary_df = [labels.split(',') for labels in y_binary_df]
    y_type_df = [labels.split(',') for labels in y_type_df]
    y_binary_bin = mlb.fit_transform(y_binary_df)
    y_type_bin = mlb.fit_transform(y_type_df)
    
    # 划分训练集和测试集
    X_binary_train, X_binary_test, y_binary_train, y_binary_test = train_test_split(X_binary_df, y_binary_bin, test_size=100, random_state=42)
    X_type_train, X_type_test, y_type_train, y_type_test = train_test_split(X_type_df, y_type_bin, test_size=100, random_state=42)

    
    
    for model_name, model in models.items():
        print(f'正在构建概念为{concept_num}，子概念个数为{subconcept_num}的模型-{model_name}')
        clf_binary = OneVsRestClassifier(model)
        clf_binary.fit(X_binary_train, y_binary_train)
        y_binary_proba  = clf_binary.predict_proba(X_binary_test)
        y_binary_pred = np.zeros_like(y_binary_proba)
        max_indices = np.argmax(y_binary_proba, axis=1)
        y_binary_pred[np.arange(y_binary_pred.shape[0]), max_indices] = 1
        
        y_binary_pred_labels = mlb.inverse_transform(y_binary_pred)
        y_binary_pred_strings = [",".join(map(str, labels)) if labels else "" for labels in y_binary_pred_labels]
        y_binary_pred_series = pd.Series(y_binary_pred_strings)
        
        y_binary_orig_labels = mlb.inverse_transform(y_binary_test)
        y_binary_orig_strings = [",".join(map(str, labels)) if labels else "" for labels in y_binary_orig_labels]
        y_binary_orig_series = pd.Series(y_binary_orig_strings)
        
        result_binary = pd.concat([y_binary_orig_series, y_binary_pred_series], axis=1)
        result_binary.columns = ['true', 'pred']
        
        result_binary_df = X_binary_test.reset_index(drop=True)
        result_binary_df = pd.concat([result_binary_df, result_binary], axis=1)
        result_binary_df.to_excel(f'./other_model_result_2/concept_{concept_num}_{subconcept_num}_{model_name}_binary.xlsx', index=False)
        
        res_binary = evaluate_metrics(result_binary)



        X_type_test_data = X_type_test
        y_type_test_data = y_type_test

        X_type_train, y_type_train = expand_cartesian(X_type_train, y_type_train)
        X_type_test_data, y_type_test_data = expand_cartesian(X_type_test_data, y_type_test_data)

        clf_type = OneVsRestClassifier(model)
        clf_type.fit(X_type_train, y_type_train)

        y_type_proba = clf_type.predict_proba(X_type_test_data)
        y_type_pred = np.zeros_like(y_type_proba)
        max_indices = np.argmax(y_type_proba, axis=1)
        y_type_pred[np.arange(y_type_pred.shape[0]), max_indices] = 1

        y_type_pred_labels = mlb.inverse_transform(y_type_pred)
        y_type_pred_strings = [",".join(map(str, labels)) if labels else "" for labels in y_type_pred_labels]
        y_type_pred_series = pd.Series(y_type_pred_strings)

        y_type_orig_labels = mlb.inverse_transform(y_type_test_data)
        y_type_orig_strings = [",".join(map(str, labels)) if labels else "" for labels in y_type_orig_labels]
        y_type_orig_series = pd.Series(y_type_orig_strings)

        result_type = pd.concat([y_type_orig_series, y_type_pred_series], axis=1)
        result_type.columns = ['true', 'pred']

        res_type = evaluate_metrics(result_type)

        result_type_df = X_type_test_data.reset_index(drop=True)
        result_type_df = pd.concat([result_type_df, result_type], axis=1)
        result_type_df.to_excel(f'./other_model_result_2/concept_{concept_num}_{subconcept_num}_{model_name}_type.xlsx', index=False)
        
        import os
        cwd = os.getcwd()
        os.chdir('./other_model_result_2')
        with open(f'./concept_{concept_num}_{subconcept_num}.txt', 'a') as f:
            f.write(f'concept_{concept_num}_{subconcept_num}_{model_name}:\n')
            f.write(f"binary: \t precision:{res_binary['precision']:.3f} \t recall:{res_binary['recall']:.3f} \t f1:{res_binary['f1']:.3f}\n")
            f.write(f"type: \t precision:{res_type['precision']:.3f} \t recall:{res_type['recall']:.3f} \t f1:{res_type['f1']:.3f}\n")
            f.write('----------------------------------------------------------')
            f.write('\n')

        os.chdir(cwd)

        s = [concept_num, subconcept_num, model_name, res_binary['precision'], res_binary['recall'], res_binary['f1'], res_type['precision'], res_type['recall'], res_type['f1']]
        print(s)

        s = pd.DataFrame({
            'concept': str(concept_num),
            'subconcept': subconcept_num,
            'model': model_name,
            'pre_b':res_binary['precision'],
            'rec_b':res_binary['recall'],
            'f1_b':res_binary['f1'],
            'pre_t':res_type['precision'],
            'rec_t':res_type['recall'],
            'f1_t':res_type['f1']
        }, index=[0])
        # print(s)

        df.loc[len(df)] = s.iloc[0]


#%%
for concept_num in concept_nums:
    for sub_concept_num in sub_concept_nums:
        execute(concept_num, sub_concept_num)

df.to_excel(f'./other_model_result_2/final_result.xlsx', index=False)
#%%
df
#%%
