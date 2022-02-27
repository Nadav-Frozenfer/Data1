#!/usr/bin/env python
# coding: utf-8

# In[4]:


pip install xgboost


# In[5]:


conda install -c conda-forge imbalanced-learn


# In[7]:


import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import numpy as np 
import pandas as pd 
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
sns.set_style('darkgrid')
warnings.filterwarnings('ignore')


# In[8]:


stroke = pd.read_csv("healthcare-dataset-stroke-data.csv")
# for print first five rows in data
stroke.head()


# In[9]:


stroke.shape


# In[10]:


stroke.columns=stroke.columns.str.lower()
stroke.isna().sum()
stroke=stroke.fillna(np.mean(stroke['bmi']))
stroke.isna().sum()
stroke['smoking_status'].replace('Unknown', stroke['smoking_status'].mode()[0], inplace=True)
stroke.drop('id', axis=1, inplace=True)
stroke= stroke[stroke['gender'] != 'Other']


# In[11]:


print("now i segregate the data into numerical and categorical values.Categorical data refers to a data type that can be stored and identified based on the names or labels given to them.\n")

print("Numerical data refers to the data that is in the form of numbers, and not in any language or descriptive form.This will help us analyze the data better.\n")


# In[12]:


numeric_data=stroke.loc[:,stroke.nunique() > 5]
cols = [col for col in stroke.columns if col not in numeric_data]

categorical_data=stroke[cols].drop('stroke',axis=1)
numeric_data=pd.DataFrame(numeric_data)
categorical_data=pd.DataFrame(categorical_data)


# In[13]:


plt.figure(figsize=(10,6))
ax=sns.countplot(x='smoking_status',data=stroke, palette='rainbow',hue='stroke')
plt.title("Count of people in each Smoking Group")
for p in ax.patches:
        ax.annotate('{}'.format(p.get_height()), (p.get_x()+0.15, p.get_height()+50))
print("People who formerly smoked and who smoke show signs of stroke way more than people who never smoked.")


# In[14]:


print("SMOTE - Synthetic Minority Oversampling Technique is an oversampling technique where the synthetic samples are generated for the minority class. This algorithm helps to overcome the overfitting problem posed by random oversampling.")


# In[15]:


print("Imbalanced classification involves developing predictive models on classification datasets that have a severe class imbalance.\n")

print("The challenge of working with imbalanced datasets is that most machine learning techniques will ignore, and in turn have poor performance on, the minority class, although typically it is performance on the minority class that is most important\n")
print("One approach to addressing imbalanced datasets is to oversample the minority class.\n")


# In[16]:


num_cols=numeric_data.columns.to_list()
sc = StandardScaler()
stroke[num_cols] = sc.fit_transform(stroke[num_cols])
le = LabelEncoder()
object_col = [col for col in stroke.columns if stroke[col].dtype == 'object']
for col in object_col:
    stroke[col] = le.fit_transform(stroke[col])


# In[17]:


training_data=stroke.copy()
x= training_data.drop(['stroke'],axis=1)
y= stroke['stroke']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)


# In[18]:


sm = SMOTE()
x_train, y_train = sm.fit_resample(x_train,y_train)


# In[19]:


lr= LogisticRegression()
lr.fit(x_train, y_train)
lr_pred = lr.predict(x_test)
lr_acc = accuracy_score(lr_pred, y_test)
lr_f1 = f1_score(lr_pred, y_test)
lr_acc


# In[20]:


decision_tree = DecisionTreeClassifier()   
decision_tree.fit(x_train,y_train)
dt_pred = decision_tree.predict(x_test)
dt_acc = accuracy_score(dt_pred, y_test)
dt_acc


# In[21]:


svm=SVC(random_state=42)
svm.fit(x_train, y_train)
svm_pred = svm.predict(x_test)
svm_acc = accuracy_score(svm_pred, y_test)
svm_acc


# In[22]:


knn = KNeighborsClassifier()
knn.fit(x_train, y_train)
knn_pred = knn.predict(x_test)
knn_acc = accuracy_score(knn_pred, y_test)
knn_acc


# In[23]:


xgb = XGBClassifier()
xgb.fit(x_train, y_train)
xgb_pred = xgb.predict(x_test)
xgb_acc = accuracy_score(xgb_pred, y_test)
xgb_acc


# In[25]:


rf = RandomForestClassifier(n_estimators = 25)
rf.fit(x_train, y_train)
rf_pred = rf.predict(x_test)
rf_acc = accuracy_score(rf_pred, y_test)
rf_acc


# In[26]:


models_names = ["LogisticRegression",'DecisionTreeClassifier','RandomForestClassifier','XGBClassifier',
                    'KNeighborsClassifier','SVC']
models_acc=[lr_acc,dt_acc,rf_acc,xgb_acc,knn_acc,svm_acc]

plt.rcParams['figure.figsize']=12,6
ax = sns.barplot(x=models_names, y=models_acc, palette = "mako", saturation =1.5)
plt.xlabel('Classifier Models' )
plt.ylabel('Accuracy')
plt.title('Accuracy of different Classifier Models')
plt.xticks(fontsize = 10, horizontalalignment = 'center', rotation = 8)
plt.yticks(fontsize = 10)
for i in ax.patches:
    width, height = i.get_width(), i.get_height()
    x, y = i.get_xy() 
    ax.annotate(f'{round(height,5)}', (x + width/2, y + height*1.02), ha='center', fontsize = 10)
plt.show()


# In[27]:



print("Hyperparameter tuning is choosing a set of optimal hyperparameters for a learning algorithm. Here we apply model tuning only to rf Classifier, as it has the highest accuracy so far.")


# In[29]:


from sklearn.model_selection import GridSearchCV
estimator = XGBClassifier(
    objective= 'binary:logistic',
    nthread=1,
    seed=42
)
parameters = {
    'max_depth': range (2, 10, 1),
    'n_estimators': range(60, 220, 40),
    'learning_rate': [0.1, 0.01, 0.05]
}
grid_search = GridSearchCV(
    estimator=estimator,
    param_grid=parameters,
    scoring = 'roc_auc',
    n_jobs = 4,
    cv = 10,
    verbose=3
)
grid_search.fit(x_train, y_train)


# In[30]:


grid_search.best_estimator_


# In[31]:


grid_search.best_score_


# In[32]:


xgb_tuned=XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=1, enable_categorical=False,
              gamma=0, gpu_id=-1, importance_type=None,
              interaction_constraints='', learning_rate=0.1, max_delta_step=0,
              max_depth=9, min_child_weight=1,
              monotone_constraints='()', n_estimators=160, n_jobs=1, nthread=1,
              num_parallel_tree=1, predictor='auto', random_state=42,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=0.2, seed=42,
              subsample=1, tree_method='exact', validate_parameters=1,
              verbosity=None)

xgb_tuned.fit(x_train, y_train)
xgb_tpred = xgb_tuned.predict(x_test)
xgb_tacc = accuracy_score(xgb_tpred, y_test)
xgb_tacc


# In[33]:


from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, precision_score, recall_score, f1_score, roc_curve, plot_roc_curve, auc,classification_report
cm = confusion_matrix(y_test, xgb_tpred)
xgb_tprob = xgb_tuned.predict_proba(x_test)[:,1]
print(classification_report(y_test, xgb_tpred))
print('ROC AUC score: ',roc_auc_score(y_test, xgb_tprob))
print('Accuracy Score: ',accuracy_score(y_test, xgb_tpred))


# In[34]:


print("Here we have completed modelling as well as tuning. The accuracy obtained is 93.5 %.")


# In[ ]:




