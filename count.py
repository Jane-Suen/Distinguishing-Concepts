#%%
import pandas as pd
df = pd.read_excel('H:\Jane\Desktop\concept_202_type.xlsx')
df.head()
#%%
df.drop(columns=['instance', 'subconcept'], inplace=True)
print(df.shape)
#%%
# 创建一个空列表来存储所有的取值
all_values = []

for col in df.columns:
    for val in df[col]:
        if isinstance(val, str) and ',' in val:
            values = val.split(',')
            all_values.extend(values)
        else:
            # 否则直接添加到列表中
            all_values.append(val)
#%%
# 移除所有的0
all_values = [value for value in all_values if value != '0' and value != 0]
#%%
# 统计all_values中每个值的出现次数
value_counts = {}
for value in all_values:
    if value in value_counts:
        value_counts[value] += 1
    else:
        value_counts[value] = 1
#%%
print(len(value_counts))