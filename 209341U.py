#!/usr/bin/env python
# coding: utf-8

# In[40]:


print('Import required libraries')

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.model_selection import train_test_split


# In[41]:


data=pd.read_csv('wbcd.csv',header=None,na_values=['?'])


# In[42]:


data.columns=['id','clump_thickness:','nuiformity_of_cell_size','uniformity_of_cell_shape','Marginal Adhesion'             ,'single_epithelial_cell_size','bare_nuclei','bland_chromatin','normal_nucleoli','Mitoses','class']


# In[43]:


data.describe()


# In[44]:


print('Drop missing value rows')
replacedData=data.dropna()


# In[45]:


X=replacedData.iloc[:,1:-1].values


# In[46]:


Y=replacedData.iloc[:,-1].values


# In[47]:


Y=(Y-2)/2 #to simplify


# In[48]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3490, random_state=101)


# In[49]:


c = tree.DecisionTreeClassifier()
c = c.fit(X_train, Y_train)


# In[50]:


get_ipython().run_line_magic('matplotlib', 'inline')
tree.plot_tree(c)


# In[51]:


Y_pred = c.predict(X_test)


# In[52]:


from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(Y_test, Y_pred))
print(classification_report(Y_test, Y_pred))


# In[53]:


cnf_matrix=confusion_matrix(Y_test,Y_pred)
class_names=[0,1] # classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual')
plt.xlabel('Predicted')

