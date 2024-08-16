# importing the required Libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

# %%
print(os.listdir('/Users/akhilsamvarghese/Desktop/Projects/ai-tutorial/nids/archive-2'))

# %%
with open("/Users/akhilsamvarghese/Desktop/Projects/ai-tutorial/nids/archive-2/kddcup.names",'r') as f:
    print(f.read())

# %%
cols="""duration,
protocol_type,
service,
flag,
src_bytes,
dst_bytes,
land,
wrong_fragment,
urgent,
hot,
num_failed_logins,
logged_in,
num_compromised,
root_shell,
su_attempted,
num_root,
num_file_creations,
num_shells,
num_access_files,
num_outbound_cmds,
is_host_login,
is_guest_login,
count,
srv_count,
serror_rate,
srv_serror_rate,
rerror_rate,
srv_rerror_rate,
same_srv_rate,
diff_srv_rate,
srv_diff_host_rate,
dst_host_count,
dst_host_srv_count,
dst_host_same_srv_rate,
dst_host_diff_srv_rate,
dst_host_same_src_port_rate,
dst_host_srv_diff_host_rate,
dst_host_serror_rate,
dst_host_srv_serror_rate,
dst_host_rerror_rate,
dst_host_srv_rerror_rate"""

columns=[]
for c in cols.split(','):
    if(c.strip()):
       columns.append(c.strip())

columns.append('target')
#print(columns)
print(len(columns))

# %%
with open("/Users/akhilsamvarghese/Desktop/Projects/ai-tutorial/nids/archive-2/training_attack_types",'r') as f:
    print(f.read())

# %%
attacks_types = {
    'normal': 'normal',
'back': 'dos',
'buffer_overflow': 'u2r',
'ftp_write': 'r2l',
'guess_passwd': 'r2l',
'imap': 'r2l',
'ipsweep': 'probe',
'land': 'dos',
'loadmodule': 'u2r',
'multihop': 'r2l',
'neptune': 'dos',
'nmap': 'probe',
'perl': 'u2r',
'phf': 'r2l',
'pod': 'dos',
'portsweep': 'probe',
'rootkit': 'u2r',
'satan': 'probe',
'smurf': 'dos',
'spy': 'r2l',
'teardrop': 'dos',
'warezclient': 'r2l',
'warezmaster': 'r2l',
}


# %% [markdown]
# READING DATASET
# KDDCUP1999
# %%
path = "/Users/akhilsamvarghese/Desktop/Projects/ai-tutorial/nids/archive-2/kddcup.data_10_percent.gz"
df = pd.read_csv(path,names=columns)

#Adding Attack Type column
df['Attack Type'] = df.target.apply(lambda r:attacks_types[r[:-1]])

df.head()

# %%
df.shape

# %%
df['target'].value_counts()

# %%
df['Attack Type'].value_counts()

# %%
df.dtypes

# %% [markdown]
# DATA PREPROCESSING

# %%
df.isnull().sum()

# %%
#Finding categorical features
num_cols = df._get_numeric_data().columns

cate_cols = list(set(df.columns)-set(num_cols))
cate_cols.remove('target')
cate_cols.remove('Attack Type')

cate_cols

# %%


# %% [markdown]
# CATEGORICAL FEATURES DISTRIBUTION

# %%
#Visualization
def bar_graph(feature):
    df[feature].value_counts().plot(kind="bar")

# %%
bar_graph('protocol_type')

# %% [markdown]
# Protocol type: We notice that ICMP is the most present in the used data, then TCP and almost 20000 packets of UDP type

# %%
plt.figure(figsize=(15,3))
bar_graph('service')

# %%
bar_graph('flag')

# %%
bar_graph('logged_in')

# %% [markdown]
# logged_in (1 if successfully logged in; 0 otherwise): We notice that just 70000 packets are successfully logged in.

# %% [markdown]
# TARGET FEATURE DISTRIBUTION

# %%
bar_graph('target')

# %% [markdown]
# Attack Type(The attack types grouped by attack, it's what we will predict)

# %%
bar_graph('Attack Type')

# %%
df.columns

# %% [markdown]
# DATA CORRELATION

# %%
df = df.dropna(axis='columns')

# Select numeric columns
numeric_df = df.select_dtypes(include=['float64', 'int64'])

# Keep columns where there are more than 1 unique value
numeric_df = numeric_df[[col for col in numeric_df if numeric_df[col].nunique() > 1]]

# Calculate correlation matrix
corr = numeric_df.corr()

# Visualize correlation matrix using heatmap
plt.figure(figsize=(15, 12))
sns.heatmap(corr)
plt.title('Correlation Matrix')
plt.show()

# %%
df['num_root'].corr(df['num_compromised'])

# %%
df['srv_serror_rate'].corr(df['serror_rate'])

# %%
df['srv_count'].corr(df['count'])

# %%
df['srv_rerror_rate'].corr(df['rerror_rate'])

# %%
df['dst_host_same_srv_rate'].corr(df['dst_host_srv_count'])

# %%
df['dst_host_srv_serror_rate'].corr(df['dst_host_serror_rate'])

# %%
df['dst_host_srv_rerror_rate'].corr(df['dst_host_rerror_rate'])

# %%
df['dst_host_same_srv_rate'].corr(df['same_srv_rate'])

# %%
df['dst_host_srv_count'].corr(df['same_srv_rate'])

# %%
df['dst_host_same_src_port_rate'].corr(df['srv_count'])

# %%
df['dst_host_serror_rate'].corr(df['serror_rate'])

# %%
df['dst_host_serror_rate'].corr(df['srv_serror_rate'])

# %%
df['dst_host_srv_serror_rate'].corr(df['serror_rate'])

# %%
df['dst_host_srv_serror_rate'].corr(df['srv_serror_rate'])

# %%
df['dst_host_rerror_rate'].corr(df['rerror_rate'])

# %%
df['dst_host_rerror_rate'].corr(df['srv_rerror_rate'])

# %%
df['dst_host_srv_rerror_rate'].corr(df['rerror_rate'])

# %%
df['dst_host_srv_rerror_rate'].corr(df['srv_rerror_rate'])

# %%
#This variable is highly correlated with num_compromised and should be ignored for analysis.
#(Correlation = 0.9938277978738366)
df.drop('num_root',axis = 1,inplace = True)

#This variable is highly correlated with serror_rate and should be ignored for analysis.
#(Correlation = 0.9983615072725952)
df.drop('srv_serror_rate',axis = 1,inplace = True)

#This variable is highly correlated with rerror_rate and should be ignored for analysis.
#(Correlation = 0.9947309539817937)
df.drop('srv_rerror_rate',axis = 1, inplace=True)

#This variable is highly correlated with srv_serror_rate and should be ignored for analysis.
#(Correlation = 0.9993041091850098)
df.drop('dst_host_srv_serror_rate',axis = 1, inplace=True)

#This variable is highly correlated with rerror_rate and should be ignored for analysis.
#(Correlation = 0.9869947924956001)
df.drop('dst_host_serror_rate',axis = 1, inplace=True)

#This variable is highly correlated with srv_rerror_rate and should be ignored for analysis.
#(Correlation = 0.9821663427308375)
df.drop('dst_host_rerror_rate',axis = 1, inplace=True)

#This variable is highly correlated with rerror_rate and should be ignored for analysis.
#(Correlation = 0.9851995540751249)
df.drop('dst_host_srv_rerror_rate',axis = 1, inplace=True)

#This variable is highly correlated with dst_host_srv_count and should be ignored for analysis.
#(Correlation = 0.9736854572953938)
df.drop('dst_host_same_srv_rate',axis = 1, inplace=True)

# %%
df.head()

# %%
df.shape

# %%
df.columns

# %%
numeric_df = df.select_dtypes(include=['float64', 'int64'])

# Calculate standard deviation
df_std = numeric_df.std()

# Sort standard deviations in ascending order
df_std_sorted = df_std.sort_values(ascending=True)

print(df_std_sorted)



# %% [markdown]
# FEATURE MAPPING

# %%
df['protocol_type'].value_counts()

# %%
#protocol_type feature mapping
pmap = {'icmp':0,'tcp':1,'udp':2}
df['protocol_type'] = df['protocol_type'].map(pmap)

# %%
df['flag'].value_counts()

# %%
#flag feature mapping
fmap = {'SF':0,'S0':1,'REJ':2,'RSTR':3,'RSTO':4,'SH':5 ,'S1':6 ,'S2':7,'RSTOS0':8,'S3':9 ,'OTH':10}
df['flag'] = df['flag'].map(fmap)

# %%
df.head()

# %%
df.drop('service',axis = 1,inplace= True)

# %%
df.shape

# %%
df.head()

# %%
df.dtypes

# %% [markdown]
# MODELLING

# %%
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

# %%
df = df.drop(['target',], axis=1)
print(df.shape)

# Target variable and train set
Y = df[['Attack Type']]
X = df.drop(['Attack Type',], axis=1)

sc = MinMaxScaler()
X = sc.fit_transform(X)

# Split test and train data 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
print(X_train.shape, X_test.shape)
print(Y_train.shape, Y_test.shape)

# %% [markdown]
# GAUSSIAN NAIVE BAYES

# %%
# Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB

# %%
model1 = GaussianNB()

# %%
start_time = time.time()
model1.fit(X_train, Y_train.values.ravel())
end_time = time.time()

# %%
print("Training time: ",end_time-start_time)

# %%
start_time = time.time()
Y_test_pred1 = model1.predict(X_test)
end_time = time.time()

# %%
print("Testing time: ",end_time-start_time)

# %%
print("Train score is:", model1.score(X_train, Y_train))
print("Test score is:",model1.score(X_test,Y_test))

# %% [markdown]
# DECISION TREE

# %%
#Decision Tree 
from sklearn.tree import DecisionTreeClassifier

# %%
model2 = DecisionTreeClassifier(criterion="entropy", max_depth = 4)

# %%
start_time = time.time()
model2.fit(X_train, Y_train.values.ravel())
end_time = time.time()

# %%
print("Training time: ",end_time-start_time)

# %%
start_time = time.time()
Y_test_pred2 = model2.predict(X_test)
end_time = time.time()

# %%
print("Testing time: ",end_time-start_time)

# %%
print("Train score is:", model2.score(X_train, Y_train))
print("Test score is:",model2.score(X_test,Y_test))

# %% [markdown]
# RANDOM FOREST

# %%
from sklearn.ensemble import RandomForestClassifier

# %%
model3 = RandomForestClassifier(n_estimators=30)

# %%
start_time = time.time()
model3.fit(X_train, Y_train.values.ravel())
end_time = time.time()

# %%
print("Training time: ",end_time-start_time)

# %%
start_time = time.time()
Y_test_pred3 = model3.predict(X_test)
end_time = time.time()

# %%
print("Testing time: ",end_time-start_time)

# %%
print("Train score is:", model3.score(X_train, Y_train))
print("Test score is:",model3.score(X_test,Y_test))

# %% [markdown]
# SUPPORT VECTOR MACHINE

# %%
from sklearn.svm import SVC

# %%
model4 = SVC(gamma = 'scale')

# %%
start_time = time.time()
model4.fit(X_train, Y_train.values.ravel())
end_time = time.time()

# %%
print("Training time: ",end_time-start_time)

# %%
start_time = time.time()
Y_test_pred4 = model4.predict(X_test)
end_time = time.time()

# %%
print("Testing time: ",end_time-start_time)

# %%
print("Train score is:", model4.score(X_train, Y_train))
print("Test score is:", model4.score(X_test,Y_test))

# %% [markdown]
# LOGISTIC REGRESSION

# %%
from sklearn.linear_model import LogisticRegression

# %%
model5 = LogisticRegression(max_iter=1200000)

# %%
start_time = time.time()
model5.fit(X_train, Y_train.values.ravel())
end_time = time.time()

# %%
print("Training time: ",end_time-start_time)

# %%
start_time = time.time()
Y_test_pred5 = model5.predict(X_test)
end_time = time.time()

# %%
print("Testing time: ",end_time-start_time)

# %%
print("Train score is:", model5.score(X_train, Y_train))
print("Test score is:",model5.score(X_test,Y_test))

# %% [markdown]
# GRADIENT BOOSTING CLASSIFIER

# %%
from sklearn.ensemble import GradientBoostingClassifier

# %%
model6 = GradientBoostingClassifier(random_state=0)

# %%
start_time = time.time()
model6.fit(X_train, Y_train.values.ravel())
end_time = time.time()

# %%
print("Training time: ",end_time-start_time)

# %%
start_time = time.time()
Y_test_pred6 = model6.predict(X_test)
end_time = time.time()

# %%
print("Testing time: ",end_time-start_time)

# %%
print("Train score is:", model6.score(X_train, Y_train))
print("Test score is:", model6.score(X_test,Y_test))

# %% [markdown]
# Artificial Neural Network

# %%
import tensorflow as tf
print(tf.__version__)

# %%
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.wrappers.scikit_learn import KerasClassifier

# from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from scikeras.wrappers import KerasClassifier

# %%
def fun():
    model = Sequential()
    
    #here 30 is output dimension
    model.add(Dense(30,input_dim =30,activation = 'relu',kernel_initializer='random_uniform'))
    
    #in next layer we do not specify the input_dim as the model is sequential so output of previous layer is input to next layer
    model.add(Dense(1,activation='sigmoid',kernel_initializer='random_uniform'))
    
    #5 classes-normal,dos,probe,r2l,u2r
    model.add(Dense(5,activation='softmax'))
    
    #loss is categorical_crossentropy which specifies that we have multiple classes
    
    model.compile(loss ='categorical_crossentropy',optimizer = 'adam',metrics = ['accuracy'])
    
    return model

# %%
#Since,the dataset is very big and we cannot fit complete data at once so we use batch size.
#This divides our data into batches each of size equal to batch_size.
#Now only this number of samples will be loaded into memory and processed. 
#Once we are done with one batch it is flushed from memory and the next batch will be processed.
# model7 = KerasClassifier(build_fn=fun,epochs=100,batch_size=64)
model7 = KerasClassifier(model=fun, epochs=150, batch_size=10, verbose=0)

# %%
print(X_train.shape)

# %%
start = time.time()
model7.fit(X_train, Y_train.values.ravel())
end = time.time()

# %%
print('Training time')
print((end-start))

# %%
start_time = time.time()
Y_test_pred7 = model7.predict(X_test)
end_time = time.time()

# %%
print("Testing time: ",end_time-start_time)

# %%
start_time = time.time()
Y_train_pred7 = model7.predict(X_train)
end_time = time.time()

# %%
accuracy_score(Y_train,Y_train_pred7)

# %%
accuracy_score(Y_test,Y_test_pred7)

# %% [markdown]
# TRAINING ACCURACY

# %%
names = ['NB','DT','RF','SVM','LR','GB','ANN']
values = [87.951,99.058,99.997,99.875,99.352,99.793,99.914]
f = plt.figure(figsize=(15,3),num=10)
plt.subplot(131)
plt.ylim(80,102)
plt.bar(names,values)

# %%
f.savefig('training_accuracy_figure.png',bbox_inches='tight')

# %% [markdown]
# TESTING ACCURACY

# %%
names = ['NB','DT','RF','SVM','LR','GB','ANN']
values = [87.903,99.052,99.969,99.879,99.352,99.771,99.886]
f = plt.figure(figsize=(15,3),num=10)
plt.subplot(131)
plt.ylim(80,102)
plt.bar(names,values)

# %%
f.savefig('test_accuracy_figure.png',bbox_inches='tight')

# %% [markdown]
# TRAINING TIME

# %%
names = ['NB','DT','RF','SVM','LR','GB','ANN']
values = [1.04721,1.50483,11.45332,126.96016,56.67286,446.69099,1211.54094]
f = plt.figure(figsize=(15,3),num=10)
plt.subplot(131)
plt.bar(names,values)

# %%
f.savefig('train_time_figure.png',bbox_inches='tight')

# %% [markdown]
# TESTING TIME

# %%
names = ['NB','DT','RF','SVM','LR','GB','ANN']
values = [0.79089,0.10471,0.60961,32.72654,0.02198,1.41416,1.72521]
f = plt.figure(figsize=(15,3),num=10)
plt.subplot(131)
plt.bar(names,values)

# %%
f.savefig('test_time_figure.png',bbox_inches='tight')


