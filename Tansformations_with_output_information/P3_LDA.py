# -*- coding: utf-8 -*-

# Import libraries to bu used
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

#%% Generating the dataset
np.random.seed(1000)
a = np.random.multivariate_normal([10,0],[[2,4],[4,6]],size=[100])
b = np.random.multivariate_normal([8,5],[[4,5],[4,6]],size=[100])
c = np.random.multivariate_normal([20,20],[[3,0],[0,3]],size=[100])

# Scenario 1
X = np.concatenate((a,c),)
Y = np.reshape(np.concatenate((100*[0],100*[1]),),(X.shape[0],1))

# # Scenario 2
# X = np.concatenate((a,b),)
# Y = np.reshape(np.concatenate((100*[0],100*[1]),),(X.shape[0],1))

# # Scenario 3
# X = np.concatenate((a,b,c),)
# Y = np.reshape(np.concatenate((100*[0],100*[1],100*[2]),),(X.shape[0],1))

# # Scenario 4
# centers = [[0,0],[0,0]]
# X, Y = make_blobs(n_samples=2000, centers=centers, n_features=2,random_state=0, cluster_std=0.6)
# index = Y==1
# tmp = X[index]
# t=np.linspace(0,2*np.pi,1000)
# x1 = 4*np.cos(t) + tmp[:,0]
# x2 = 4*np.sin(t) + tmp[:,1]
# X[index] = np.transpose([x1,x2])
# Y = np.reshape(Y,[2000,1])
# X = X + 2

# # Presentation example
# X = np.array([[1,2],[2,3],[3,3],[4,5],[5,5],[4,2],[5,0],[5,2],[3,2],[5,3],[6,3]])
# Y = np.reshape([0,0,0,0,0,1,1,1,1,1,1],(X.shape[0],1))

plt.scatter(X[:,0],X[:,1],c=Y)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.grid()
plt.show()


#%% Scaling data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

#%% Aplying the LDA manually
classes = np.unique(Y) # determine num classes

# Obtaining the within-class covariance
mu_all = np.mean(X,axis=0)
mu_class = np.zeros((len(classes),np.shape(X)[1]))
Sw = list()
Sb = list()
SW_all = np.zeros((np.shape(X)[1],np.shape(X)[1]))
SB_all = np.zeros((np.shape(X)[1],np.shape(X)[1]))
for k in range(len(classes)):
    mu_class[k,:] = np.mean(X[Y[:,0]==classes[k],:],axis=0)
    Xc_center = X[Y[:,0]==classes[k]]-mu_class[k,:]
    Sw.append(np.dot(Xc_center.T,Xc_center))
    SW_all = SW_all + Sw[k]
    mu_between = np.reshape(mu_class[k,:]-mu_all,(1,np.shape(X)[1]))
    Sb.append(np.dot(mu_between.T,mu_between)*sum(Y==classes[k]))
    SB_all = SB_all + Sb[k]
Sw_Sb = np.dot(np.linalg.inv(SW_all),SB_all)
[W,V] = np.linalg.eig(Sw_Sb)
ind_w = np.argsort(W)[::-1]

#%% View the covariance ratio
plt.bar(np.arange(np.shape(W)[0]),W[ind_w]/sum(W))
plt.xlabel('Num eigenvalues')
plt.ylabel('% explained variance')

#%% View the new axis
x_plot = np.arange(-3,3,0.1)
v1_plot = (V[1,ind_w[0]]/V[0,ind_w[0]])*x_plot+mu_all[1]-(V[1,ind_w[0]]/V[0,ind_w[0]])*mu_all[0]
v2_plot = (V[1,ind_w[1]]/V[0,ind_w[1]])*x_plot+mu_all[1]-(V[1,ind_w[1]]/V[0,ind_w[1]])*mu_all[0]

plt.scatter(X[:,0],X[:,1],c=Y,label='data')
plt.plot(x_plot,v1_plot,label='1st component (%0.3f)'%W[ind_w[0]])
plt.plot(x_plot,v2_plot,label='2nd component (%0.3f)'%W[ind_w[1]])
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend()
plt.axis([-3,3,-3,3])
plt.grid()
plt.show()

#%% Transform the data to new axis
X_new = np.dot(X,V[:,ind_w])

plt.scatter(X_new[:,0],X_new[:,1],c=Y,label='data')
# plt.scatter(X_new[:,0],np.zeros((X.shape[0])),c=Y,label='data')
plt.xlabel('v_1')
plt.ylabel('v_2')
plt.legend()
plt.grid()
plt.show()

#%% Aplying the LDA
lda_model = LDA(store_covariance=True,n_components=None)
lda_model = lda_model.fit(X,Y[:,0])

# View the covariance ratio
plt.bar(np.arange(len(lda_model.explained_variance_ratio_)),lda_model.explained_variance_ratio_)
plt.xlabel('Num eigenvalues')
plt.ylabel('% explained variance')

#%% Data transform with LDA

X_new = lda_model.transform(X)

# View the data transformed
plt.scatter(X_new[:,0],np.zeros((X.shape[0])),c=Y,label='data')
plt.xlabel('v_1')
plt.ylabel('v_2')
plt.legend()
plt.grid()
plt.show()

#%% How to use the prediction
Yhat = lda_model.predict(X)
# Yhat = lda_model.predict_proba(X)

#%% What happen with the PCA?
from sklearn.decomposition import PCA
pca_model = PCA()
pca_model.fit(X)

# View the covariance ratio
plt.bar(np.arange(len(pca_model.explained_variance_ratio_)),pca_model.explained_variance_ratio_)
plt.xlabel('Num eigenvalues')
plt.ylabel('% explained variance')

#%% View the new axis with PCA
V = pca_model.components_.T
x_plot = np.arange(-3,3,0.1)
v1_plot = (V[1,0]/V[0,0])*x_plot+mu_all[1]-(V[1,0]/V[0,0])*mu_all[0]
v2_plot = (V[1,1]/V[0,1])*x_plot+mu_all[1]-(V[1,1]/V[0,1])*mu_all[0]

plt.scatter(X[:,0],X[:,1],c=Y,label='data')
plt.plot(x_plot,v1_plot,label='1st component (%0.3f)'%pca_model.explained_variance_ratio_[0])
plt.plot(x_plot,v2_plot,label='2nd component (%0.3f)'%pca_model.explained_variance_ratio_[1])
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend()
plt.axis([-3,3,-3,3])
plt.grid()
plt.show()

#%% Data transform with PCA

X_new = pca_model.transform(X)

# View the data transformed
plt.scatter(X_new[:,0],np.zeros((X.shape[0])),c=Y,label='data')
plt.xlabel('v_1')
plt.ylabel('v_2')
plt.legend()
plt.grid()
plt.show()