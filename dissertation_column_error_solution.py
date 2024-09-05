#!/usr/bin/env python
# coding: utf-8

# # This program is used for dealing with the error format of columns as well as combining all the ChiPi result file as experimental data

# In[1]:


import numpy as np
import pandas as pd


# In[3]:


# connact three results file from ChiPi program
data1=pd.read_csv("F:\\dissertation\\result_p1.csv",encoding="utf-8")
data2=pd.read_csv("F:\\dissertation\\result_p2.csv",encoding="utf-8")
data3=pd.read_csv("F:\\dissertation\\result_p3.csv",encoding="utf-8")


# In[5]:


data1.head(5)


# In[4]:


data2.head(5)


# In[6]:


data3.head(5)


# In[10]:


#first error column
c1 = data1.columns.get_loc('Compound Name')
c1


# In[12]:


#real last column
clast = data1.columns.get_loc(data1.columns[-1])
clast


# In[13]:


#supposed last column
cn=data1.columns.get_loc('tanimoto_not_same_chirality')
cn


# In[15]:


#Deal with first ChiPi result file
for index, row in data1.iterrows():
    columns_to_merge = 0  #number of col needs merging
    for col in reversed(data1.columns):
        if pd.notna(row[col]):
            col_index = data1.columns.get_loc(col)
            columns_to_merge=col_index-cn+1
            break
    else:
        columns_to_merge = 0

    if columns_to_merge !=0:
        # get the columns need merging
        start_column_index = c1  #the first column need merging
        columns_range = data1.columns[start_column_index:start_column_index + columns_to_merge]

        # merge the target rows, remove NA
        merge_result = ','.join(str(row[col]) for col in columns_range if pd.notna(row[col]))

        # save the merged value
        data1.at[index, data1.columns[start_column_index]] = merge_result

        # reorder the columns
        for col_index in range(start_column_index + 1, len(data1.columns) - columns_to_merge + 1):
            data1.iloc[index, col_index] = data1.iloc[index, col_index + columns_to_merge - 1]

        # remove content useless
        data1.iloc[index, len(data1.columns) - columns_to_merge + 1:] = None

# remove NA cols
df1 = data1.iloc[:, :-(columns_to_merge + 1)]


# In[16]:


#Deal with second ChiPi result file
for index, row in data2.iterrows():
    columns_to_merge = 0  #合并列数
    for col in reversed(data2.columns):
        if pd.notna(row[col]):
            col_index = data2.columns.get_loc(col)
            columns_to_merge=col_index-cn+1
            break
    else:
        columns_to_merge = 0

    if columns_to_merge !=0:
        # get the columns need merging
        start_column_index = c1   #the first column need merging
        columns_range = data2.columns[start_column_index:start_column_index + columns_to_merge]

        # merge the target rows, remove NA
        merge_result = ','.join(str(row[col]) for col in columns_range if pd.notna(row[col]))

         # save the merged value
        data2.at[index, data2.columns[start_column_index]] = merge_result

         # reorder the columns
        for col_index in range(start_column_index + 1, len(data2.columns) - columns_to_merge + 1):
            data2.iloc[index, col_index] = data2.iloc[index, col_index + columns_to_merge - 1]

        # remove content useless
        data2.iloc[index, len(data2.columns) - columns_to_merge + 1:] = None

# remove other columns
df2 = data2.iloc[:, :-(columns_to_merge + 1)]


# In[25]:


#Deal with last ChiPi result file
for index, row in data3.iterrows():
    columns_to_merge = 0  #number of columns needing merging
    for col in reversed(data3.columns):
        if pd.notna(row[col]):
            col_index = data3.columns.get_loc(col)
            columns_to_merge=col_index-cn+1
            break
    else:
        columns_to_merge = 0

    if columns_to_merge !=0:
        # get the columns need merging
        start_column_index = c1  #the first column need merging
        columns_range = data3.columns[start_column_index:start_column_index + columns_to_merge]

        # merge the target rows, remove NA
        merge_result = ','.join(str(row[col]) for col in columns_range if pd.notna(row[col]))

        # save the merged value
        data3.at[index, data3.columns[start_column_index]] = merge_result

         # reorder the columns
        for col_index in range(start_column_index + 1, len(data3.columns) - columns_to_merge + 1):
            data3.iloc[index, col_index] = data3.iloc[index, col_index + columns_to_merge - 1]

        # remove content useless
        data3.iloc[index, len(data3.columns) - columns_to_merge + 1:] = None

## remove NA cols
df3 = data3.iloc[:, :-(columns_to_merge + 1)]


# In[26]:


df3.head(5)


# In[27]:


df1.head(5)


# In[30]:


d1=df1.iloc[:,:71]
d2=df2.iloc[:,:71]
d3=df3.iloc[:,:71]


# In[29]:


d3.head(5)


# In[32]:


#save the data without format error
output_path1 = 'F:\dissertation\output_p1.csv'
output_path2 = 'F:\dissertation\output_p2.csv' 
output_path3 = 'F:\dissertation\output_p3.csv' # file path
#save as csv file
d1.to_csv(output_path1, index=False)
d2.to_csv(output_path2, index=False)
d3.to_csv(output_path3, index=False)


# In[33]:


#combine all csv file and save it as a new file
combined1 = pd.concat([d1, d2], ignore_index=True)
combined2 = pd.concat([combined1, d3], ignore_index=True)
combined2.to_csv('F:\dissertation\output.csv', index=False)


# In[ ]:




