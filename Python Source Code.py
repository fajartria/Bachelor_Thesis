
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
#import parfit.parfit as pf

from pandas import ExcelWriter
from sklearn import neighbors
from sklearn import svm
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn import tree
from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ParameterGrid


# Scaling modules
from sklearn.preprocessing import MinMaxScaler

# Accuracy metric
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import roc_auc_score

from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

# Open input file
fn = r"YourFile.xlsx"

# Read Data
df_x = pd.read_excel(fn) #Input
df_y = pd.read_excel(fn) #Output

df_xt= pd.read_excel(fn) 
df_yt= pd.read_excel(fn)



# Extract values from Dataframe into numpy array
x = df_x.values
y = df_y.values
xt = df_xt.values
yt= df_yt.values

#print(df_x.head())
#print(df_y.head())
#print(df_xt.head())
#print(df_yt.head())


# Create scaler object
scaler_x = MinMaxScaler().fit(x)
scaler_xt = MinMaxScaler().fit(xt)

# Scale the data
x_scaled = scaler_x.transform(x)
xt_scaled = scaler_xt.transform(xt)

# Split data
n_test = 0.3
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=n_test, random_state=39)

# Classification methods, choose one
clf = neighbors.KNeighborsClassifier(n_neighbors=3)
#clf = svm.SVC()
#clf = SGDClassifier(alpha=1e-4, max_iter=500, penalty='l2')
#clf = MLPClassifier(max_iter=1000,hidden_layer_sizes=1000,alpha=1e-4)
#clf = GaussianProcessClassifier()
#clf = tree.DecisionTreeClassifier()
#clf = GaussianNB(priors=2)

#Model Optimization
KNN = {'n_neighbors':[1,3,5,7,9,11,13,15,17,19,21,23,25,27,29]}

SGD = {
    'alpha': [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3], # learning rate
    'max_iter': [100,200,300,400,500,600,700,800,900,1000], # number of epochs
    'loss': ['log'], # logistic regression,
    'penalty': ['l2']
}
MLP = {
    'alpha': [1e-4, 1e-3, 1e-2, 1e-1], # learning rate
    'max_iter': [600,700,800,900,1000] # number of epochs
}

# Hyperparameter Tuning Method
#RS = RandomizedSearchCV(clf,MLP)
#GS = GridSearchCV(clf,MLP)

# model fitting
clf.fit(x_train, y_train[:,1])

# Hyperparameter Metrics
#print(RS.best_score_)
#print(RS.best_params_)
#print(RS.best_estimator_)

# predicting test data
y_pred = clf.predict(x_test)


# predicting all data
y_pred_all = clf.predict(xt_scaled)

# Check accuracy on test data after transformed back
mse = mean_squared_error(y_test[:,1], y_pred)
mae = mean_absolute_error(y_test[:,1], y_pred)
variance = explained_variance_score(y_test[:,1], y_pred)
r2 = r2_score(y_test[:,1], y_pred)

print(clf)
print("Mean Absolute Error", mae)
print("Mean Squared Error", mse)
print("Variance", variance)
print("R-squared", r2)


#Plot result
fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(1, 6, sharey=True, figsize=(12,20))
fig.suptitle('Log Data')
ax1.axes.invert_yaxis()
ax1.plot(x[:,0],y[:,0], 'b', label='GR')
ax2.plot(x[:,1],y[:,0], 'g', label='SONIC')
ax3.plot(x[:,2],y[:,0], 'r', label='BULK')
ax4.plot(x[:,3],y[:,0], 'c', label='NEUTRON')
ax5.plot(x[:,4],y[:,0], 'y', label='SP')
ax6.plot(y_test[:,1].flatten(), y_test[:,0], ".", label="Actual")
ax6.plot(y_pred.flatten(), y_test[:,0], "o", label="Prediction")
plt.grid()
#Label
ax1.set_ylabel('Depth, m')
ax1.set_title('GR')
ax2.set_title('SONIC')
ax3.set_title('BULK')
ax4.set_title('NEUTRON')
ax5.set_title('SP')
ax6.set_title('Validation')




# Compared prediction and actual on test set

fig, (ax1, ax2) = plt.subplots(1,2,  sharey=True, figsize=(10,20))
fig.suptitle('Actual Prediction Comparation on Test Set')
ax1.set_ylabel('Depth, m')
ax1.axes.invert_yaxis()
ax2.plot(y[:,1].flatten(), y[:,0],  label="Actual Labeled Data")
ax2.plot(x_scaled[:,0].flatten(), y[:,0],  label="GR")
#ax1.plot(y_test[:,1].flatten(), y_test[:,0], ".", label="Actual")
#ax1.plot(y_pred.flatten(), y_test[:,0],".",  label="Prediction")
plt.grid()
plt.legend()
#ax2.plot(yt[:,1].flatten(), yt[:,0], ".", label="Actual")
ax1.plot(y_pred_all.flatten(), yt[:,0], label="Prediction")
ax1.plot(xt_scaled[:,0].flatten(), yt[:,0], label="Actual GR")
plt.grid()
plt.legend()


#CONFUSION MATRIX
fig, (ax1) = plt.subplots(1, sharey=True, figsize=(10,10))
fig.suptitle('confusion matrix')
cm=confusion_matrix(yt[:,1], y_pred_all)

cmp=(cm/np.sum(cm))*100
print('matrix data ')
print(cm)
print('matrix data in percent ')
print(np.around(cmp,1))
print(" ")

i=0
truetrue=0
for i in range(2):
        truetrue +=cmp[i][i]

i=0
j=0
falsefalse=0
for i in range(2):
    for j in range(2):
        if i!=j:
            falsefalse +=cmp[i][j]
            

print('true=',np.round(truetrue,2))
print('false=',np.round(falsefalse,2))
            
            
sns.heatmap(cm, annot=True ,cmap='Reds')
cmp=cmp/100

fig, (ax1) = plt.subplots(1, sharey=True, figsize=(10,10))
fig.suptitle('confusion matrix percentage')
sns.heatmap(cmp,annot = False,fmt='.2%', cmap='Blues')




# Export results to excel
df = pd.DataFrame(y_pred)
dfa = pd.DataFrame(y_pred_all)
writer = ExcelWriter('test2.xlsx')
df.to_excel(writer,'Sheet1',index=False)
dfa.to_excel(writer,'Sheet2',index=False)
writer.save()

