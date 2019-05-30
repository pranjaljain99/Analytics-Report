# DATA ANALYTICS STEPS :
Define Objective      | 
Data Collecion        | 
Data Cleaning         | 
Field Level analysis  | 
Data Consolidation    | 
Model Development     | 
Visualization and Interpretation

# Objective : 
# CLASSIFICATION AND PREDICTION OF COUNTRIES BASED ON THEIR LEVEL OF DEVELOPMENT.

# Problem Statement: 
To find the status of a country, whether developed, developing or under-developing by looking at some factors that will help us identify the same and predicting the future HDI Values.
# Hypothesis: 
After doing literature review about different factors and features that help us in identifying the status of a country, we took 9 major factors which are highly relevant in determining a countries progress or regress.
For Example: We considered HDI, which shows the human development or a country, which again is important to know in order to determine the level of development of any country.

# Feature Selection Method:
We selected nine features on which we can identify the status of any country. We, then performed two Supervised Learning Techniques on the dataset.
# Classification Problem: 
It is a problem of identifying to which of a set of categories, a new observation belongs, based on a training set of data containing observations whose category memberships is known. On selecting 8 features, which helps in identifying the status of any country, the dataset is divided into 2 categories: Training set and Test set. The Training dataset allows us to train the model and the Test dataset will allow to calculate how accurate the model is. Out of 182 countries in the dataset, 0.2% i.e., 20% of the dataset is used as Test Data and remaining 80% has been used for testing.
The X-Axis depicts the Features and the Y-axis depicts the Target variables i.e., the status.
# Prediction Problem: 
It involves a predicting the score of one variable from the second variable. The variable that is being predicted is called Criterion variable and the variable that helps in prediction is known as Predictor Variable. Since, there is only one prediction variable that has been used, this is a simple regression. We have dropped all the columns for predicting other than HDI. HDI is reflected on the X-axis as Criterion Variable and Y axis is used for the status.
# External Libraries Used: 
There are a few external libraries that have helped to perform predicting and classification as under:
• Pandas:  
It is an open-source library which is used for high-performance, easy-to-use data structures and data analytics tools for Python[2]. It is easy to use for data manipulation and analysis. It has been used in this project as it takes data in any format like, CSV files, TSV files etc. and creates a Python Object which is called Data Frames which is like a table that is created to score records.  
• Read Excel:
Read_Excel is a library which is used to read the data from Excel file into Pandas data frame. It supports both xls and xlsx file extensions from a local filesystem or URL[3].  
• Matplotlib:
It is a plotting library in Python that produces publication quality figures in a variety of hardcopy formats and interactive environments across platforms[4]. This has been used to generate plots and histogram in the project.  
• Pylab:
It is an integrated, numeric computation environment. It has been used to for numerical computation and numerical analysis in the project.  
• Seaborn:
It is a visualization library and it is based on matplotlib. It has been used to provide high-level interface for drawing statistical graphics in the project.  
• Scikit-Learn:
It is a library for python which has enabled us to do classification, regression and clustering of our dataset. There are a lot of Scikit learn libraries that have been used in the code whose detailed description is attached later in the document and appendices. A list of all libraries used are as under:  
o Train_Test_Split  
o Min_Max_Scaler  
o LogisticRegression  
o Isomap  
o DecisionTreeClassifier  
o KNeighbourClassifier  
o Classification_Report  
o Confusion_Matrix  
o SVC (Support Vector Classifier)  
o Random forest Classifier  
o Linear_model  
o Plot_decision_matrix  
o Mglearn  
• Tkinter:  

# Data Source: 
The dataset has been retrieved from an online source called Knoema. 
# Description of Dataset:
• HDI- Human Development Index
• Export
• Import
• Employment share in population
• GDP- Gross Domestic Product
• GNI- Gross National Income
• Loan Share
• Total Health Expenditure in Share of GDP
