#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from pandas import read_excel
import matplotlib as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import pylab as plot


# In[480]:


# Loading data


# In[481]:


df0=pd.read_excel(r"C:\Users\MSI PRO\Desktop\New folder (2)\ONLY_HDI.xlsx")
df1=pd.read_excel(r"C:\Users\MSI PRO\Desktop\New folder (2)\All_other_datasets.xlsx")
df = pd.merge(df0,df1,on=["Location_Name"])
df.to_excel("final.xlsx",index=False)


# In[482]:


df.head()


# In[483]:


df.dtypes


# In[484]:


df.isna().sum()


# In[485]:


df1=df.dropna()


# In[486]:


df1.isna().sum()


# In[487]:


df1.shape


# In[488]:


# # Adding Status column according to condition


# In[489]:


fig,ax=plot.subplots(1,1)
ax.scatter(df1["Location_Name"],df1["HDI_2017"])


# In[490]:


df1.loc[df1["HDI_2017"] <= 0.490, 'Status'] = 0
df1.loc[df1["HDI_2017"] >=0.750 , 'Status'] = 2
df1.loc[(df1["HDI_2017"] > 0.490) & (df1["HDI_2017"] < 0.750), 'Status'] = 1


# In[491]:


fig,ax=plot.subplots(1,1)
ax.scatter(df1["Status"],df1["HDI_2017"])


# In[492]:


import seaborn as sns
sns.countplot(df1['Status'],label="Count")
plot.show()


# In[493]:


print(df1.groupby('Status').size())


# In[494]:


plot.figure(figsize=(8,5))
sns.distplot(df1["HDI_2017"])
plot.show()


# In[495]:


plot.figure(figsize=(15,8))
sns.heatmap(df1.corr(),annot=True,cmap='coolwarm')
plot.show()


# In[496]:


sns.pairplot(df1, palette='rainbow')


# # # Feature Selection
# 

# In[497]:


X=df1.iloc[:,2:19].values #features
y = y=df1.iloc[:,-1].values #target variable


# In[498]:


y


# In[499]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0,test_size=.2)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[500]:


# # Logistic Regression


# In[501]:


from sklearn.linear_model import LogisticRegression
logistic = LogisticRegression()
logistic.fit(X_train, y_train)

print('Accuracy of Logistic regression classifier : {:.2f}'.format(logistic.score(X_test, y_test)))


# In[502]:


y_pred=logistic.predict(X_test)

print("Mean squared error: %.2f" % np.mean((logistic.predict(X_test) - y_test) ** 2))


# y_pred=logistic.predict(X_test)
# 
# print("Mean squared error: %.2f" % np.mean((logistic.predict(X_test) - y_test) ** 2))
# 

# In[503]:


y_test


# In[ ]:





# In[504]:


pd.crosstab(y_test,y_pred,colnames=["predicted"],rownames=["Actual"])


# In[505]:


from sklearn.manifold import Isomap
X_iso = Isomap(n_neighbors=10).fit_transform(X_train)
lgr = logistic.predict(X_train)
fig, ax = plot.subplots(1, 2, figsize=(8, 4))
fig.suptitle('Predicted Versus Training Labels', fontsize=14, fontweight='bold')
fig.subplots_adjust(top=0.85)
ax[0].scatter(X_iso[:, 0], X_iso[:, 1], c=lgr)
ax[0].set_title('Predicted Training Labels')
ax[1].scatter(X_iso[:, 0], X_iso[:, 1], c=y_train)
ax[1].set_title('Actual Training Labels')
plot.show()


# In[506]:


# # Decision Tree


# In[507]:


from sklearn.tree import DecisionTreeClassifier
decision = DecisionTreeClassifier().fit(X_train, y_train)
print('Accuracy of Decision Tree classifier: {:.2f}'.format(decision.score(X_test, y_test)))


# In[508]:


y_pred=decision.predict(X_test)

print("Mean squared error: %.2f" % np.mean((decision.predict(X_test) - y_test) ** 2))


# In[509]:


print(pd.crosstab(y_test,y_pred,colnames=["predicted"],rownames=["Actual"]))


# In[510]:


from sklearn.manifold import Isomap
X_iso = Isomap(n_neighbors=10).fit_transform(X_train)
clusters = decision.predict(X_train)
fig, ax = plot.subplots(1, 2, figsize=(8, 4))
fig.suptitle('Predicted Versus Training Labels', fontsize=14, fontweight='bold')
fig.subplots_adjust(top=0.85)
ax[0].scatter(X_iso[:, 0], X_iso[:, 1], c=clusters)
ax[0].set_title('Predicted Training Labels')
ax[1].scatter(X_iso[:, 0], X_iso[:, 1], c=y_train)
ax[1].set_title('Actual Training Labels')
plot.show()


# In[511]:


# # KNN


# In[512]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
print('Accuracy of K-NN classifier : {:.2f}'.format(knn.score(X_test, y_test)))


# In[513]:


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
pred = knn.predict(X_test)
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))


# In[514]:


# # SVM


# In[515]:


from sklearn.svm import SVC
svm = SVC()
svm.fit(X_train, y_train)
print('Accuracy of SVM classifier: {:.2f}'.format(svm.score(X_test, y_test)))


# In[516]:


y_pred=svm.predict(X_test)

print("Mean squared error: %.2f" % np.mean((svm.predict(X_test) - y_test) ** 2))


# In[517]:


print(pd.crosstab(y_test,y_pred,colnames=["predicted"],rownames=["Actual"]))


# In[518]:


from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 42)
classifier.fit(X_train, y_train)


# In[519]:


print('Accuracy of RF classifier: {:.2f}'.format(classifier.score(X_test, y_test)))


# In[520]:


print(pd.crosstab(y_test,y_pred,colnames=["predicted"],rownames=["Actual"]))


# In[521]:


# # Want to plot decision boundary


# In[522]:


df


# In[523]:


# # Converting column to row


# In[524]:


#Drop all the columns except Location name, 2007-2017
df1.drop(['Location_RegionId','export','import','Employment_share_in_pop','GDP_in_USD','GNI_in_USD','Loan_Share','Total_health_exp_in_share_of_GDP',"Status"], axis=1, inplace=True)


# In[525]:


df1.head(10)
df1
df1.columns=["Location_Name","2007","2008","2009","2010","2011","2012","2013","2014","2015","2016","2017"]
#df2 = df1.apply(pd.to_numeric)


# In[526]:


df2 = df1.T


# In[527]:


df2.head(10)


# In[528]:


new_header = df2.iloc[0] #grab the first row for the header
df2 = df2[1:] #take the data less the header row
df2.columns = new_header
df2.head(10)


# In[529]:


#df2 = pd.to_numeric(df2)
df2.dtypes


# In[530]:


df3 = df2.apply(pd.to_numeric)


# In[531]:


df3.dtypes


# In[532]:


plot.plot(df3.index, df3['India'])
plot.title('India-HDI')
plot.ylabel('HDI');
plot.show()


# In[533]:


plot.plot(df3.index, df3['United Arab Emirates'])
plot.title('United Arab Emirates-HDI')
plot.ylabel('HDI');
plot.show()


# In[534]:


plot.plot(df3.index, df3['Iraq'])
plot.title('Iraq-HDI')
plot.ylabel('HDI');
plot.show()


# In[535]:


plot.plot(df3.index, df3['Afghanistan'])
plot.title('Afghanistan-HDI')
plot.ylabel('HDI');
plot.show()


# In[ ]:





# In[536]:


plot.figure(figsize=(10, 8))
plot.plot(df3.index, df3['India'], 'b-', label = 'India')
plot.plot(df3.index, df3['Afghanistan'], 'r-', label = 'Afghanistan')
plot.plot(df3.index, df3['United Arab Emirates'], 'g-', label = 'United Arab Emirates')
plot.xlabel('Year'); plot.ylabel('HDI'); plot.title('HDI for 3 countries')
plot.legend();


# In[537]:


#Can plot in this way also
India_HDI=df3['India']
plot.plot(India_HDI)


# In[538]:


ax=df3["Egypt"].plot(figsize=(14,6),lw=2,style="k-")


# In[539]:


df3.head()
df3.dtypes


# In[540]:


df3=df2.apply(pd.to_numeric)
df3


# In[541]:


df3.index = pd.to_numeric(df3.index)
X = df3.index.values #features
y = df3["India"].values #target variable
X


# In[ ]:





# In[542]:


import statsmodels.api as sm # import statsmodels 

X = sm.add_constant(X) ## let's add an intercept (beta_0) to our model

# Note the difference in argument order
model = sm.OLS(y, X).fit() ## sm.OLS(output, input)
predictions = model.predict(X)

# Print out the statistics
model.summary()


# In[543]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0,test_size=.2)
X_train.reshape(-1,1)


# In[544]:


from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train,y_train)


# In[545]:


confidence = lm.score(X_test, y_test)
print(confidence)


# In[546]:



forecast_set = lm.predict(X_test)
forecast_set


# In[547]:


import tkinter as tk


# In[550]:


# tkinter GUI
root= tk.Tk() 


# In[551]:


canvas1 = tk.Canvas(root, width = 1200, height = 450)
canvas1.pack()
Intercept_result = ('Intercept: ', lm.intercept_)
label_Intercept = tk.Label(root, text=Intercept_result, justify = 'center')
canvas1.create_window(260, 220, window=label_Intercept)
Coefficients_result  = ('Coefficients: ', lm.coef_)
label_Coefficients = tk.Label(root, text=Coefficients_result, justify = 'center')
canvas1.create_window(260, 240, window=label_Coefficients)


# In[552]:


# with statsmodels
print_model = model.summary()
label_model = tk.Label(root, text=print_model, justify = 'center', relief = 'solid', bg='LightSkyBlue1')
canvas1.create_window(800, 220, window=label_model)


# In[553]:


# New_Interest_Rate label and input box
#Instead of this label use combo box
label1 = tk.Label(root, text='  HDI:')
canvas1.create_window(100, 100, window=label1)


# In[554]:


entry1 = tk.Entry (root) # create 1st entry box
canvas1.create_window(270, 100, window=entry1)


# In[555]:


# New_Unemployment_Rate label and input box
label2 = tk.Label(root, text=' Year:         ')
canvas1.create_window(120, 120, window=label2)


# In[556]:


entry2 = tk.Entry (root) # create 2nd entry box
canvas1.create_window(270, 120, window=entry2)


# In[557]:


# In[67]:


def values(): 
    global HDI #our 1st input variable
    HDI = float(entry1.get()) 
    
    global Year #our 2nd input variable
    Year = float(entry2.get()) 
    
    Prediction_result  = ('Predicted output: ', lm.predict([[HDI ,Year]]))
    label_Prediction = tk.Label(root, text= Prediction_result, bg='orange')
    canvas1.create_window(260, 280, window=label_Prediction)
    


# In[558]:


button1 = tk.Button (root, text='Predict ',command=values, bg='orange') # button to call the 'values' command above 
canvas1.create_window(270, 150, window=button1)
 

root.mainloop()


# In[ ]:




