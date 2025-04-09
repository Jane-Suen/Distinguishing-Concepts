
import pandas as pd
import numpy as np

concept_num = 170
subconcept_num = 5

binary_df = pd.read_excel(f'./concept_{concept_num}_binary.xlsx').astype(str)
type_df = pd.read_excel(f'./concept_{concept_num}_type.xlsx').astype(str)

from graphviz import Digraph
from collections import Counter
from collections import defaultdict

import warnings

cols = type_df.columns.difference(['instance', 'subconcept'])
type_df[cols] = type_df[cols].applymap(lambda x :'819320' if x != '0' else x)

warnings.filterwarnings("ignore")

class decisionTree:
    def __init__(self, binary_df, type_df):
        # 原始数据集
        self.binary_df = binary_df
        self.type_df = type_df

        # 训练集和测试集
        self.train_binary_df = None
        self.train_type_df = None
        self.test_binary_df = None
        self.test_type_df = None

        # 决策树
        self.binary_tree = None
        self.type_tree = None

        # 子概念数量
        self.subconcept_number = subconcept_num

        # 概念编号
        self.concept_number = concept_num

        # 决策树图像
        self.binary_graph = None
        self.type_graph = None
        
        # 执行结果
        self.result_binary = None
        self.result_type = None
        
        # 评估指标
        self.metrics_binary = None
        self.metrics_type = None

    def check_same_feature(self, df):
        """
        检查DataFrame中是否有相同的特征，如果有则删除
        :param df:
        :return:
        """
        df = df.loc[:, (df != df.iloc[0]).any()]
        return df

    def filter_df(self):
        """
        过滤DataFrame中的subconcept字段，保留出现次数最多的前n个子概念
        :return:
        """
        # 拆分所有的subconcept，统计出现次数
        def filter_top_subconcepts(df, n=self.subconcept_number):
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

        self.binary_df = filter_top_subconcepts(self.binary_df)
        self.type_df = filter_top_subconcepts(self.type_df)

    def find_best_feature(self, df, target_col='subconcept', weight_col='weight'):
        """
        计算每个特征列对目标列'subconcept'的信息增益率（IGR），并考虑权重列'weight'。

        参数：
            df (pd.DataFrame): 输入数据集，包含目标列'subconcept'和权重列'weight'。
            target_col (str): 目标列名称，默认为'subconcept'。
            weight_col (str): 权重列名称，默认为'weight'。

        返回：
            dict: 每个特征列的信息增益率，按值降序排序。
        """
        # 计算信息熵的辅助函数
        def entropy(probs):
            probs = np.array(probs)  # 确保是 NumPy 数组
            probs = probs[probs > 0]  # 过滤掉 0 值，避免 log(0)
            return -np.sum(probs * np.log2(probs)) if len(probs) > 0 else 0

        # 计算目标列的熵（H(Y)）
        def compute_target_entropy(df, target_col, weight_col):
            value_weights = defaultdict(float)
            total_weight = df[weight_col].sum()

            for _, row in df.iterrows():
                subconcepts = row[target_col].split(',')
                weight_per_value = row[weight_col] / len(subconcepts)  # 计算每个子类的权重
                for value in subconcepts:
                    value_weights[value] += weight_per_value

            probs = np.array(list(value_weights.values())) / total_weight
            return entropy(probs)

        # 计算某个特征列的条件熵 H(Y|X)
        def compute_conditional_entropy(df, feature, target_col, weight_col):
            feature_values = df[feature].unique()
            total_weight = df[weight_col].sum()
            conditional_entropy = 0

            for value in feature_values:
                subset = df[df[feature] == value]
                subset_entropy = compute_target_entropy(subset, target_col, weight_col)
                weight_ratio = subset[weight_col].sum() / total_weight
                conditional_entropy += weight_ratio * subset_entropy

            return conditional_entropy

        # 计算某个特征列的熵 H(X)
        def compute_feature_entropy(df, feature, weight_col):
            feature_weights = df.groupby(feature)[weight_col].sum()
            total_weight = df[weight_col].sum()
            probs = np.array(feature_weights / total_weight)
            return entropy(probs)

        # 计算目标列的信息熵 H(Y)
        target_entropy = compute_target_entropy(df, target_col, weight_col)

        # 遍历所有特征列，计算信息增益率
        igr_results = {}

        for feature in df.columns:
            if feature in {target_col, weight_col}:
                continue

            conditional_entropy = compute_conditional_entropy(df, feature, target_col, weight_col)
            information_gain = target_entropy - conditional_entropy
            feature_entropy = compute_feature_entropy(df, feature, weight_col)

            # 计算信息增益率 IGR(X)
            igr = information_gain / feature_entropy if feature_entropy > 0 else 0
            igr_results[feature] = igr

        # 按信息增益率降序排序
        dic = dict(sorted(igr_results.items(), key=lambda item: item[1], reverse=True))
        max_igr_features = [k for k, v in dic.items() if v == max(dic.values())]

        # 输出最大特征及其对应的信息增益率
        for feature in max_igr_features:
            print(f"特征: {feature}, 信息增益率: {dic[feature]}")



        return max_igr_features

    def prepare_data(self):
        """
        准备数据，将数据集划分为训练集和测试集
        :return:
        """
        # 去掉subconcept为-1的记录
        self.binary_df = self.binary_df[self.binary_df['subconcept'] != '-1']
        self.type_df = self.type_df[self.type_df['subconcept'] != '-1']

        # 筛选包含前subconcept_number个subconcept的记录
        self.filter_df()

        # 划分训练集和测试集
        from sklearn.model_selection import train_test_split
        self.train_binary_df, self.test_binary_df = train_test_split(self.binary_df, test_size=100, random_state=42)
        self.train_type_df = self.type_df[self.type_df['instance'].isin(self.train_binary_df['instance'])]
        self.test_type_df = self.type_df[self.type_df['instance'].isin(self.test_binary_df['instance'])]

        # 重置索引
        self.train_binary_df = self.train_binary_df.reset_index(drop=True)
        self.test_binary_df = self.test_binary_df.reset_index(drop=True)
        self.train_type_df = self.train_type_df.reset_index(drop=True)
        self.test_type_df = self.test_type_df.reset_index(drop=True)



        # 去除instance列
        self.train_binary_df = self.train_binary_df.drop(columns=['instance'])
        self.test_binary_df = self.test_binary_df.drop(columns=['instance'])
        self.train_type_df = self.train_type_df.drop(columns=['instance'])
        self.test_type_df = self.test_type_df.drop(columns=['instance'])


        # 保存训练集和测试集
        self.train_binary_df.to_excel(f'./data/{self.concept_number}_train_binary.xlsx', index=False)
        self.test_binary_df.to_excel(f'./data/{self.concept_number}_test_binary.xlsx', index=False)
        self.train_type_df.to_excel(f'./data/{self.concept_number}_train_type.xlsx', index=False)
        self.test_type_df.to_excel(f'./data/{self.concept_number}_test_type.xlsx', index=False)

        # 去除取值相同的列（不可划分）
        self.train_binary_df = self.check_same_feature(self.train_binary_df)
        self.train_type_df = self.check_same_feature(self.train_type_df)

    def decision_tree(self):
        """
        构建决策树及其图像
        :return:
        """
        # 添加权重列，用于后续信息增益率计算
        self.train_binary_df['weight'] = 1
        self.train_type_df['weight'] = 1

        print("构建binary决策树……")
        # 构建决策树-剪枝-合并-生成图像
        self.binary_tree = self.build_tree(self.train_binary_df)
        self.binary_tree = self.prune_tree(self.binary_tree, self.subconcept_number)
        self.binary_tree = self.merge_partial_subconcepts_in_tree(self.binary_tree)
        # import pickle
        # with open(f'./model_pickle/{concept_num}_{subconcept_num}_Binary_Tree_Model.pkl', 'rb') as f:
        #     self.binary_tree = pickle.load(f)
        self.binary_graph = self.tree_to_graph(self.binary_tree)

        print("构建type决策树……")
        self.type_tree = self.build_tree(self.train_type_df)
        self.type_tree = self.prune_tree(self.type_tree, self.subconcept_number)
        self.type_tree = self.merge_partial_subconcepts_in_tree(self.type_tree)
        # with open(f'./model_pickle/{concept_num}_{subconcept_num}_Type_Tree_Model.pkl', 'rb') as f:
        #     self.type_tree = pickle.load(f)
        self.type_graph = self.tree_to_graph(self.type_tree)


    def save_graph(self):
        """
        保存决策树图像
        :return:
        """
        self.binary_graph.render(f"./data/{self.concept_number}_binary_tree", format='svg', cleanup=True)
        self.type_graph.render(f"./data/{self.concept_number}_type_tree", format='svg', cleanup=True)

    def build_tree(self, df, target_col='subconcept', path="Root"):
        def dfs_helper(df, target_col, path):
            # 1. 删除所有取值唯一的特征列（不包括目标列和权重列），减少无意义的计算
            feature_cols = [col for col in df.columns if col not in [target_col, 'weight']]
            for col in feature_cols:
                if df[col].nunique() == 1:
                    df = df.drop(columns=[col])

            # 2. 终止条件
            if len(df) == 1 or len(feature_cols) < 1 or df[target_col].nunique() == 1:
                subconcepts = set()
                for val in df[target_col].dropna().astype(str).unique():
                    subconcepts.update(map(int, val.split(',')))
                return {'subconcepts': sorted(subconcepts)}

            # 3. 选择最佳划分特征
            best_features = self.find_best_feature(df, target_col)
            print(f"当前路径: {path}, 最佳特征: {best_features}")

            if not best_features:
                subconcepts = set()
                for val in df[target_col].dropna().astype(str).unique():
                    subconcepts.update(map(int, val.split(',')))
                return {"subconcepts": sorted(subconcepts)}

            # 4. 处理多最佳特征（空白节点）
            if len(best_features) > 1:
                node = {"best_feature": None, "children": {}}
                for feature in best_features:
                    subset_node = {"best_feature": feature, "children": {}}

                    # **5. 处理特征值拆分**
                    unique_values = set()
                    for val in df[feature].dropna().astype(str).unique():
                        unique_values.update(val.split(','))

                    for single_value in unique_values:
                        subset_df = df[df[feature].astype(str).str.contains(f'(^|,){single_value}(,|$)')]
                        subset_df = subset_df.drop(columns=[feature])

                        new_path = f"{path} -> {feature}={single_value}"
                        # print(f"划分路径（多特征节点）: {new_path}")
                        subset_result = dfs_helper(subset_df, target_col, new_path)
                        subset_node["children"][single_value] = subset_result

                    node["children"][feature] = subset_node
                return node

            # 6. 单一特征情况
            best_feature = best_features[0]
            node = {"best_feature": best_feature, "children": {}}

            unique_values = set()
            for val in df[best_feature].dropna().astype(str).unique():
                unique_values.update(val.split(','))

            for single_value in unique_values:
                subset_df = df[df[best_feature].astype(str).str.contains(f'(^|,){single_value}(,|$)')]
                subset_df = subset_df.drop(columns=[best_feature])

                new_path = f"{path} -> {best_feature}={single_value}"
                print(f"划分路径: {new_path}")

                if len(subset_df) == 0:
                    continue

                sub_results = dfs_helper(subset_df, target_col, new_path)
                node["children"][single_value] = sub_results

            return node

        return dfs_helper(df, target_col, path)

    def tree_to_graph(self, tree, parent_name=None, edge_label="", graph=None):
        if graph is None:
            graph = Digraph(format='svg')
            graph.attr('node', shape='ellipse')  # 设置节点形状为椭圆

        # 确定当前节点的标签
        node_label = ""
        if "best_feature" in tree and tree["best_feature"]:  # 如果存在且非空
            node_label = tree["best_feature"]
        elif "subconcepts" in tree and tree["subconcepts"]:  # 如果存在子概念
            node_label = f"Subconcepts: {', '.join(map(str, tree['subconcepts']))}"

        # 给当前节点命名，防止重复
        node_name = f"{parent_name}_{edge_label}" if parent_name else "root"

        # 创建当前节点
        graph.node(node_name, label=node_label)

        # 如果有父节点，则连接边
        if parent_name is not None:
            graph.edge(parent_name, node_name, label=edge_label)

        # 如果有子节点，递归处理
        if "children" in tree:
            for key, subtree in tree["children"].items():
                self.tree_to_graph(subtree, parent_name=node_name, edge_label=str(key), graph=graph)

        return graph

    def prune_tree(self, tree, leaf_count_threshold):
        """
        对决策树进行剪枝：
        1. 剪掉叶子节点集合数量等于给定数字的树枝。
        2. 如果剪枝后，一个节点的子节点只有一个，则删除该节点，
           让其父节点直接指向它的子节点，直到满足决策树结构要求。

        :param tree: 决策树，嵌套字典结构
        :param leaf_count_threshold: 剪枝的叶子节点集合数量阈值
        :return: 剪枝后的决策树
        """

        def prune_node(node):
            """递归处理当前节点的剪枝操作。"""
            if 'children' not in node:
                # 当前节点是叶子节点，检查是否需要剪枝
                leaf_subconcepts = node.get('subconcepts', [])
                if len(leaf_subconcepts) == leaf_count_threshold:
                    return None  # 剪掉该树枝
                return node  # 保留该叶子节点

            # 遍历子节点，递归剪枝
            new_children = {}
            for key, child in node['children'].items():
                pruned_child = prune_node(child)
                if pruned_child is not None:
                    new_children[key] = pruned_child

            # 更新当前节点的子节点
            node['children'] = new_children

            # 检查子节点是否为空，如果为空则删除该节点
            if not node['children']:
                return None

            # 如果当前节点只有一个子节点，则将当前节点删除，父节点直接指向唯一子节点
            if len(node['children']) == 1:
                only_key = next(iter(node['children']))
                return node['children'][only_key]

            return node

        # 从根节点开始剪枝
        pruned_tree = prune_node(tree)
        return pruned_tree

    def merge_partial_subconcepts_in_tree(self, tree):
        """
        合并节点的子概念，如果下一级的叶子节点的子概念部分相同但不完全相同，
        将相同部分合并，并且合并它们的 `best_feature`。

        :param tree: 决策树，嵌套字典结构
        :return: 合并后的决策树
        """
        if not tree:
            return tree

        # 如果节点是叶子节点，直接返回
        if 'children' not in tree:
            return tree

        # 递归处理每个子节点
        for value, child in tree["children"].items():
            tree["children"][value] = self.merge_partial_subconcepts_in_tree(child)

        # 检查当前节点的子节点是否符合合并条件
        children_subconcepts = {}
        children_best_features = {}

        for value, child in tree["children"].items():
            if 'subconcepts' in child:
                # 将subconcepts转为frozenset进行比较，确保它们是不可变且可以用于字典键
                subconcepts_set = frozenset(child["subconcepts"])
                if subconcepts_set not in children_subconcepts:
                    children_subconcepts[subconcepts_set] = []
                children_subconcepts[subconcepts_set].append(value)

                # 处理best_feature，合并所有的best_feature值
                # 先检查是否存在 'best_feature'，如果没有则跳过该节点
                if 'best_feature' in child:
                    children_best_features[subconcepts_set] = children_best_features.get(subconcepts_set, set())
                    children_best_features[subconcepts_set].add(child["best_feature"])

        # 如果某些subconcepts部分相同且需要合并
        for subconcepts_set, nodes in children_subconcepts.items():
            if len(nodes) > 1:  # 说明有多个节点需要合并
                # 获取best_feature集合，若为空则赋予默认值
                best_feature_values = children_best_features.get(subconcepts_set, set())
                merged_best_feature = ",".join(sorted(best_feature_values)) if best_feature_values else ""

                # 选择一个新节点作为合并结果
                merged_node = {
                    "subconcepts": list(subconcepts_set),
                    "best_feature": merged_best_feature
                }

                # 删除原有的节点，替换为合并后的节点
                for node in nodes:
                    del tree["children"][node]

                # 将合并后的节点添加到tree中
                tree["children"][",".join(sorted(nodes))] = merged_node

        # 处理叶子节点为空的情况
        for value, child in tree["children"].items():
            if not child.get("children"):  # 确保叶子节点
                if "subconcepts" not in child:
                    child["subconcepts"] = []  # 确保叶子节点有subconcepts
                if "best_feature" not in child:
                    child["best_feature"] = ""  # 确保叶子节点有best_feature

        # 在返回之前，遍历树并删除所有叶子节点中的best_feature
        def remove_best_feature_from_leaves(node):
            if "children" not in node:  # 如果是叶子节点
                if "best_feature" in node:
                    del node["best_feature"]  # 删除叶子节点中的best_feature
            else:
                for child in node.get("children", {}).values():
                    remove_best_feature_from_leaves(child)

        # 删除所有叶子节点中的best_feature
        remove_best_feature_from_leaves(tree)

        # 最后的路径合并：如果一个节点只有一个子节点，合并该节点
        def merge_single_child_nodes(node):
            # 遍历子节点
            if "children" in node and len(node["children"]) == 1:
                # 如果只有一个子节点，合并父节点和子节点
                child_key, child_value = list(node["children"].items())[0]

                # 检查子节点是否有 'children' 键
                if "children" in child_value:
                    node["children"] = child_value["children"]
                else:
                    # 如果子节点没有 'children'，意味着它是叶子节点
                    node["children"] = {}

                node["best_feature"] = child_value.get("best_feature", "")
                node["subconcepts"] = child_value.get("subconcepts", [])

                # 递归合并路径
                merge_single_child_nodes(node)

            # 对子节点递归调用
            for child in node.get("children", {}).values():
                merge_single_child_nodes(child)


        return tree

    def execute(self, intersection=True):

        def decision(df, tree, output_excel=False, excel_filename="result.xlsx"):
                """
                做分类预测时，与原来的方案不同的是，
                1）type和value决策树的0分支作为默认分支（即不满足其他分支，则走默认分支）；
                2）遇到空节点，则每个分支都要走一下，最后将每个空节点的所有分支结果求并集（注意这里是并集，与其他情况求交集有差异）
                """


                """
                执行决策树预测，处理数据并聚合结果。

                :param df: 包含可能有复合值的 DataFrame
                :param tree: 决策树（嵌套字典结构）
                :param output_excel: 是否将结果输出为 Excel
                :param excel_filename: 输出 Excel 文件名
                :param intersection: 是否使用交集（默认是交集，False时使用并集）
                :return: 处理后的 DataFrame，包含新列 'tree_subconcept'，并保证记录唯一
                """

                def predict_subconcepts(record, tree, path=""):
                    """
                    根据传入的决策树和记录，寻找能到达的叶子节点并返回其 subconcepts 集合。
                    每次递归都会记录当前路径，不论是否遇到空白节点。

                    :param record: 当前测试记录
                    :param tree: 决策树节点
                    :param path: 记录当前路径（用于调试）
                    :return: (合并后的 subconcepts 集合, 完整的决策路径)
                    """
                    current_node = tree
                    current_path = path  # 记录到达当前节点的路径

                    # 当前走到的节点没有子节点了，说明已经走到了叶子节点（递归出口）
                    if 'children' not in current_node:
                        leaf_subconcepts = set()
                        for item in current_node.get('subconcepts', []):
                            if isinstance(item, int):
                                leaf_subconcepts.add(str(item))
                            elif isinstance(item, str):
                                leaf_subconcepts.update(item.split(','))
                        return leaf_subconcepts, current_path

                    all_subconcepts = set()
                    all_paths = []
                    # 取出最佳路径
                    best_feature = current_node.get('best_feature', None)

                    # 判断空白节点：遇到空白节点，递归遍历所有子节点并合并结果，记录路径
                    if not best_feature:

                        for key, child_node in current_node['children'].items():
                            sub_concepts, sub_path = predict_subconcepts(
                                record, child_node, f"{current_path} -> empty_node -> {key}")
                            all_subconcepts.update(sub_concepts)
                            all_paths.append(sub_path)

                        return all_subconcepts, ', '.join(all_paths)

                    # 获取当前记录的特征值，并转化为集合（如果是复合值，按逗号拆分）
                    feature_value = record.get(best_feature, '')
                    feature_values = set(feature_value.split(',')) if isinstance(feature_value, str) else {feature_value}

                    matched = False
                    # for key, child_node in current_node['children'].items():
                    #     key_values = set(key.split(','))
                    #     # 如果记录中的特征值集合和路径上的特征值集合有交集，选择这条路径
                    #     if feature_values & key_values:
                    #         matched = True
                    #         # 更新所有能走的路径
                    #         sub_concepts, sub_path = predict_subconcepts(
                    #             record, child_node, f'{current_path} -> {best_feature}={key}'
                    #         )
                    #         all_subconcepts.update(sub_concepts)
                    #         all_paths.append(sub_path)

                    for key, child_node in current_node['children'].items():
                        # 将key按逗号分割，形成一个集合
                        key_values = set(key.split(','))

                        # 判断当前特征值与子节点的key的交集是否非空
                        if feature_values & key_values:
                            matched = True

                            # 临时存储子概念
                            tmp_subconcepts, sub_path = predict_subconcepts(
                                record, child_node, f"{current_path} -> {best_feature}={key}")

                            # 如果intersection为True，取交集；否则取并集
                            if intersection:
                                # 取交集
                                if not all_subconcepts:
                                    all_subconcepts.update(tmp_subconcepts)
                                else:
                                    all_subconcepts.intersection_update(tmp_subconcepts)
                            else:
                                # 取并集
                                all_subconcepts.update(tmp_subconcepts)

                            # 将路径加入到路径集合中
                            all_paths.append(sub_path)


                    # 没有路，走默认路径 '0'
                    if not matched and '0' in current_node['children']:
                        sub_concepts, sub_path = predict_subconcepts(
                            record, current_node['children']['0'], f'{current_path} -> {best_feature}=0(default)'
                        )
                        all_subconcepts.update(sub_concepts)
                        all_paths.append(sub_path)

                    return all_subconcepts, ','.join(all_paths)


                expanded_rows = []


                # 遍历每一行数据
                for _, row in df.iterrows():
                    # 在决策树中寻找能到达的叶子节点 subconcept
                    subconcepts, decision_path = predict_subconcepts(row, tree)

                    if intersection:
                        # 如果是交集，取所有找到的 subconcepts 的交集
                        row['tree_subconcept'] = ','.join(sorted(subconcepts, key=int))
                    else:
                        # 如果是并集，取所有找到的 subconcepts 的并集
                        row['tree_subconcept'] = ','.join(sorted(set(subconcepts), key=int))

                    row['decision_path'] = decision_path  # 记录完整的决策路径
                    expanded_rows.append(row)

                # 创建包含新列的 DataFrame
                expanded_df = pd.DataFrame(expanded_rows)

                if output_excel:
                    expanded_df.to_excel(excel_filename, index=False)

                return expanded_df


        def calculate_metrics(df):
            """
            计算准确率、精确率、召回率、F1 值
            准确率：统计'subconcept'是否与'tree_subconcept'有交集，如果是，那么记录为1，否则为0。
            精准率：统计'subconcept'和'tree_subconcept'的交集的长度除以'tree_subconcept'的长度。
            召回率：统计'subconcept'和'tree_subconcept'的交集的长度除以'subconcept'的长度。
            F1值：2*精确率*召回率/(精确率+召回率)。
            :param df: 输入的 DataFrame，包含 'instance'、'tree_subconcept'、'weight'、'subconcept' 等列
            :return: 一个包含 'instance'、'accuracy'、'precision'、'recall'、'f1_score' 的 DataFrame
            """
            def calculate(row):
                tree_subconcepts = set(row['tree_subconcept'].split(','))
                subconcepts = set(row['subconcept'].split(','))

                # 计算准确率-即用来判断是否执行成功：判断两个集合是否有交集
                acc = 0 if subconcepts.isdisjoint(tree_subconcepts) else 1

                # 取两者的交集，并求其交集的元素0个数
                fz = len(subconcepts & tree_subconcepts)
                # 精准率：统计'subconcept'和'tree_subconcept'的交集的长度除以'tree_subconcept'的长度。
                pre = (fz / len(tree_subconcepts)) if len(tree_subconcepts) != 0 else 0
                # 召回率：统计'subconcept'和'tree_subconcept'的交集的长度除以'subconcept'的长度。
                rec = (fz /len(subconcepts)) if len(subconcepts) != 0 else 0
                # F1值：2*精确率*召回率/(精确率+召回率)。
                f1 = (2 * pre * rec) / (pre + rec) if (pre + rec) != 0 else 0
                return pd.Series([acc, pre, rec, f1])

            df[['accuracy', 'precision', 'recall', 'f1-score']] = df.apply(calculate, axis=1)
            return df


        self.result_binary = decision(self.test_binary_df, self.binary_tree)
        self.result_binary['subconcept_set'] = self.result_binary['subconcept'].apply(lambda x : set(x.split(',')))
        self.result_binary['tree_subconcept_set'] = self.result_binary['tree_subconcept'].apply(lambda  x : set(x.split(',')))
        # failed_binary_index = self.result_binary[self.result_binary.apply(lambda row : row['subconcept_set'].isdisjoint(row['tree_subconcept_set']), axis=1)].index

        # self.test_type_df = self.test_type_df.loc[failed_binary_index]

        self.result_type = decision(self.test_type_df, self.type_tree)
        self.result_type['subconcept_set'] = self.result_type['subconcept'].apply(lambda x : set(x.split(',')))
        self.result_type['tree_subconcept_set'] = self.result_type['tree_subconcept'].apply(lambda  x : set(x.split(',')))

        self.metrics_binary = calculate_metrics(self.result_binary)
        self.metrics_type = calculate_metrics(self.result_type)

        # 计算准确率、精确率、召回率、F1 值
        pre_binary = self.metrics_binary['precision'].sum() / 100
        pre_type = self.metrics_type['precision'].sum() / 100
        pre = (self.metrics_binary['precision'].sum() + self.metrics_type['precision'].sum()) / 100

        rec_binary = self.metrics_binary['recall'].sum() / 100
        rec_type = self.metrics_type['recall'].sum() / 100
        rec = (self.metrics_binary['recall'].sum() + self.metrics_type['recall'].sum()) / 100


        print(f"pre_binary: {pre_binary:.2f}, rec_binary: {rec_binary:.2f}, f1_binary: {2 * pre_binary * rec_binary / (pre_binary +rec_binary):.2f}")
        print(f"pre_type: {pre_type:.2f}, rec_type: {rec_type:.2f}, f1_type: {2 * pre_type * rec_type / (pre_type +rec_type):.2f}")
        
#%%
dt = decisionTree(binary_df, type_df)
dt.prepare_data()
dt.decision_tree()
dt.save_graph()
dt.execute()
#%%
concept_num, subconcept_num