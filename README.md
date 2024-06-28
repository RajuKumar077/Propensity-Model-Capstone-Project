#  Propensity Model Capstone Project

  

## Introduction

  

#### In the competitive landscape of insurance marketing, understanding customer behavior and predicting their actions is crucial. As a Data

  

#### Scientist, My task is to develop a propensity model to forecast the likelihood that individuals will engage in specic actions, such as purchasing

  

#### an insurance policy. This model will help the insurance company optimize its marketing efforts, ensuring resources are used effectively to

  

#### target potential customers.

  

## Why Information About Customers is So Important

  

#### A lot of companies collect a ton of information about their customers, but if they don't use it right, then it's not really worth anything. The point

  

#### is to take all this data and turn it into useful insights that can help the company grow. Propensity modeling is one way to do that.

  

## What is Propensity Modeling?

  

#### Propensity modeling is when you use statistics to gur e out how likely it is for a customer to do something specic, like buy insurance or

  

#### whatever. You take into account different factors that might affect their behavior, like their age, income, location, and stuff like that. This method

  

#### helps companies identify which customers are most likely to be interested in a particular marketing campaign, so they can focus their efforts

  

#### and money on those people.

  

## Data:

  

#### The insurance company has provided you with a historical data set (train.csv). The company has also provided you with a list of potential

  

#### customers to whom to market (test.csv). From this list of potential customers, you need to determine yes/no whether you wish to market to

  

#### them. (Note: Ignore any additional columns available other than the listed below in the table)

  

##  Building a Propensity Model for an Insurance Company

  

#### Screenshot 2024-06-26 at 11.22.44 AM.png

  

##  Import Libraries

  

  

```

## Basic Libraries

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import warnings

import plotly.express as px

import plotly.graph_objs as go

warnings.filterwarnings('ignore')

```

```

## Z-score for Outlier treatment

from scipy.stats import zscore

```

```

# Suppress warnings

warnings.filterwarnings('ignore')

```

```

## Preprocessing Libraries

from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.model_selection import train_test_split

from imblearn.under_sampling import RandomUnderSampler

from imblearn.over_sampling import SMOTE

```

```

## Importing Models

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.svm import SVC

from plotly.subplots import make_subplots

from sklearn.neighbors import KNeighborsClassifier

from xgboost import XGBClassifier

from sklearn.neural_network import MLPClassifier

```

```

## Model Evaluation Metrics

from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix

from sklearn.metrics import roc_curve, auc

```

```

## Libraries for Hyperparameter Tuning and Cross Validation

from scipy.stats import uniform, randint

from sklearn.model_selection import RandomizedSearchCV

from sklearn.model_selection import StratifiedKFold, GridSearchCV

from sklearn.model_selection import cross_val_score, KFold

```

```

pd.set_option('display.max_columns', None)

```

```

from google.colab import drive

drive.mount('/content/drive')

```

##  Read Dataset

  

```

## Read Dataset

data = pd.read_excel('/content/drive/MyDrive/Colab Notebooks/Dataset/Propensify/train.xlsx')

```

```

## Shape of dataset

data.shape

```

```

## First 5 rows

data.head()

```

```

# Last 5 Rows

data.tail()

```

##  Dropping Unwanted rows

  

```

### Dropping last 2 Rows as the they are sum and average of profit column

data.drop(index=data.index[-2:], inplace=True)

```

```

data.tail()

```

##  Dropping Unwanted Columns

  

  

```

data.columns

```

```

# Dropping ID column and Profit column as they are not necessary

data.drop(['id','profit'], axis= 1 , inplace=True)

```

##  Exploratory Data Analysis

  

```

# Count and Dtype of Dataset

data.info()

```

```

# Summary statistics for numerical features

num_summary = data.describe().T

num_summary

```

##  Finding and Handling Duplicated Values

  

```

# Duplicate Values yes or no

data.duplicated().any()

```

```

# Sum of duplicate rows

data.duplicated().sum()

```

```

# Duplicate Rows

data[data.duplicated()]

```

```

# Dropping Duplicate Rows

data.drop_duplicates(inplace = True)

```

##  Finding and Handling Null values

  

```

# Sum of Null Values in each row

data.isnull().sum()

```

```

# Calculate null values

null_counts = data.isnull().sum()

```

```

# Create a bar plot using Plotly Express

fig = px.bar(

x=null_counts.index,

y=null_counts.values,

text=null_counts.values,

labels={'x': 'Features', 'y': 'Number of Null Values'},

title='Null Values in Dataset',

)

```

```

# Customize layout

fig.update_traces(textposition='outside', # Display text labels outside bars

marker_color='blue', # Set color of the bars

)

```

```

fig.update_layout(

xaxis_tickangle=-45, # Rotate x-axis labels for better readability

height= 500 , # Set height of the plot

width= 900  # Set width of the plot

)

```

```

# Show the plot

fig.show()

```

```

## Imputing Null values using Forward Fill

```

```

data['custAge'].fillna(method='ffill', inplace=True)

data['schooling'].fillna(method = 'ffill', inplace=True)

data['day_of_week'].fillna(method = 'ffill', inplace=True)

```

##  Finding and Handling Outliers

  

  

```

data.head()

```

##  Replacing '999' with -1 as 999 can be considered as outlier and effect the analyis

  

```

## Replace 999 with -

data['pdays'] =data['pdays'].replace( 999 , -1)

data['pmonths'] =data['pmonths'].replace( 999 , -1)

```

```

data.head()

```

##  Boxplot for Outliers

  

```

# Create traces for each column

traces = []

for col in data.columns:

traces.append(go.Box(y=data[col], name=col, boxpoints='all', jitter=0.3, pointpos=-1.8))

```

```

# Layout configuration

layout = go.Layout(

title='Boxplot for Identifying Outliers',

xaxis=dict(title='Column Names'),

yaxis=dict(title='Values'),

showlegend=False,

height= 800

)

```

```

# Create figure object

fig = go.Figure(data=traces, layout=layout)

```

```

# Show interactive plot

fig.show()

```

```

# Select specific columns for plotting

selected_columns = ['custAge', 'campaign', 'pdays', 'previous', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx',

'euribor3m', 'pmonths', 'pastEmail']

```

```

# Create traces for each column

traces = []

for col in selected_columns:

traces.append(go.Box(y=data[col], name=col, boxpoints='all', jitter=0.3, pointpos=-1.8))

```

```

# Layout configuration

layout = go.Layout(

title='Boxplot for Identifying Outliers',

xaxis=dict(title='Column Names'),

yaxis=dict(title='Values'),

showlegend=False,

plot_bgcolor='black',

paper_bgcolor='black',

font=dict(color='white'),

height= 700 # Adjust the height as per your preference

)

```

```

# Create figure object

fig = go.Figure(data=traces, layout=layout)

```

```

# Show interactive plot

fig.show()

```

##  Finding Outliers using Z Scores

  

#### From boxplot we can observe that there are more outlier values in custAge, campaign, pdays and pastEmail columns

  

```

# Calculate Z-scores for outlier numeric column

z_scores = data[['custAge','campaign','pdays', 'pastEmail']].apply(zscore)

```

```

#Threshold

z_score_threshold =  3

```

```

# Identify outliers

outliers = (z_scores.abs() > z_score_threshold)

```

  

```

# Outliers Count

outlier_counts = outliers.sum()

print("Outlier Counts:")

print(outlier_counts)

```

##  Removing Outliers Using Z-Scores

  

```

# Identify outliers and prepare cleaned data by removing outliers

outlier_indices = (z_scores > z_score_threshold).any(axis= 1 )

cleaned_data = data[~outlier_indices]

```

```

cleaned_data

```

##  Seperating Categorical and Numerical Columns

  

```

# cleaned_data is my DataFrame containing data

categorical = [col for col in cleaned_data.columns if cleaned_data[col].dtype == 'object']

numerical = [col for col in cleaned_data.columns if cleaned_data[col].dtype != 'object']

```

```

# Display categorical and numerical columns

categorical, numerical

```

##  Data Visulaization

  

##  Plot 1: Distribution of Customer Age who responded and did not respond

  

```

# Create an interactive count plot using Plotly Express

fig = px.histogram(cleaned_data, x='custAge', color='responded', barmode='group',

title='Customer Age Distribution',

labels={'custAge': 'Customer Age', 'count': 'Count of Customers'},

category_orders={'responded': ['Yes', 'No']},

color_discrete_sequence=px.colors.qualitative.Safe)

```

```

# Customize layout with black theme

fig.update_layout(

plot_bgcolor='black', # Background color of the plot area

paper_bgcolor='black', # Background color of the entire figure

font=dict(color='white'), # Font color

xaxis=dict(tickangle= 90 ) # Rotate x-axis labels for better visibility

)

```

```

# Show plot

fig.show()

```

#### Responded customers where more around the age of 28 to 38.

  

##  Plot 2: Response of customer who have housing loan and not

  

```

# Plotly plot

fig = px.histogram(cleaned_data, x='responded', color='housing',

barmode='group', template='plotly_dark')

```

```

# layout for Plotly

fig.update_layout(

title='Customers with Housing Loan Response',

xaxis_title='Response',

yaxis_title='Count of Customers',

bargap=0.2, # gap between bars of adjacent location coordinates

bargroupgap=0.1 # gap between bars of the same location coordinates

)

```

```

# Show the interactive plot

fig.show()

```

#### Customers with housing loan responded more

  

  

### Plot 3: Response based on Profession

  

##

  

```

# Plotly interactive plot

fig = px.histogram(cleaned_data, x='profession', color='responded',

barmode='group', template='plotly_dark')

```

```

#layout for Plotly

fig.update_layout(

title='Distribution Plot Based on Profession (Interactive)',

xaxis_title='Customer Profession',

yaxis_title='Count Of Customers',

xaxis=dict(tickangle=-45),

bargap=0.2, # gap between bars of adjacent location coordinates

bargroupgap=0.1 # gap between bars of the same location coordinates

)

```

```

# Show the interactive plot

fig.show()

```

#### Most responses are from Admin, Technician and Blue Collar Professions

  

##  Plot 4: Histplot for Marital Status

  

```

# Calculate counts for each combination

grouped = cleaned_data.groupby(['marital', 'responded']).size().reset_index(name='count')

```

```

# Create stacked bar chart using Plotly Express

fig = px.bar(grouped, x='marital', y='count', color='responded',

title='Response Based On Marital Status',

labels={'marital': 'Marital Status', 'count': 'Count of Customers', 'responded': 'Response'},

template='plotly_dark',

barmode='stack') # Stack bars on top of each other

```

```

# Update layout for Plotly

fig.update_layout(

xaxis=dict(title='Marital Status'),

yaxis=dict(title='Count of Customers'),

legend=dict(title='Response'),

plot_bgcolor='black'

)

```

```

# Show the interactive plot

fig.show()

```

#### Married Customers responded the most

  

##  Plot 5: Pairplot For Numerical Columns

  

```

sns.pairplot(data = cleaned_data)

plt.title('Numerical Columns Distribution Plot')

plt.show()

```

#### Pairplot Displays the distribution between all numerical columns

  

##  Plot 6: Distribution of numerical columns using Boxplot

  

  

```

# Create a subplot figure

fig = make_subplots(rows= 3 , cols= 4 , subplot_titles=[f'Boxplot of {col}' for col in numerical])

```

```

# Define the color for the boxplots

box_color = '#1f77b4'

```

```

# Loop through each numerical column and create a boxplot

for i, column in enumerate(numerical,  1 ):

row = (i -  1 ) //  4  +  1

col = (i -  1 ) %  4  +  1

box = px.box(cleaned_data, y=column, template="plotly_dark", color_discrete_sequence=[box_color])

for trace in box.data:

fig.add_trace(trace, row=row, col=col)

```

```

# Update layout

fig.update_layout(height= 800 , width= 1400 , title_text="Numerical Data Boxplots", showlegend=False)

```

```

fig.show()

```

#### The boxplot displays the distribution of values with in the upper and lower limits

  

##  Plot 7: Distribution Using CountPlot

  

```

# Create a subplot figure

fig = make_subplots(rows= 6 , cols= 2 , subplot_titles=[f'Distribution of {col}' for col in categorical])

```

```

# Loop through each categorical column and create a bar chart

for i, column in enumerate(categorical,  1 ):

row = (i -  1 ) //  2  +  1

col = (i -  1 ) %  2  +  1

hist = px.histogram(cleaned_data, x=column, template="plotly_dark")

for trace in hist.data:

fig.add_trace(trace, row=row, col=col)

```

```

# Update layout

fig.update_layout(height= 1500 , width= 1300 , title_text="Categorical Data Distributions", showlegend=False)

```

```

# Update x-axis labels rotation

for i in range( 1 ,  13 ):

fig.update_xaxes(row=(i -  1 ) //  2  +  1 , col=(i -  1 ) %  2  +  1 , tickangle= 90 )

```

```

fig.show()

```

#### Above plot displays the distribution count of categories from all categorical columns

  

##  Plot 8: Distribution of numerical columns using Volin Chart

  

```

# Create a subplot figure

fig = make_subplots(rows= 6 , cols= 2 , subplot_titles=[f'Violin Plot of {col}' for col in numerical])

```

```

# Define the color for the violin plots

violin_color = 'red' # Change to your desired color

```

```

# Loop through each numerical column and create a violin plot

for i, column in enumerate(numerical,  1 ):

row = (i -  1 ) //  2  +  1

col = (i -  1 ) %  2  +  1

violin = px.violin(cleaned_data, y=column, template="plotly_white", color_discrete_sequence=[violin_color])

for trace in violin.data:

fig.add_trace(trace, row=row, col=col)

```

```

# Update layout

fig.update_layout(height= 1800 , width= 1400 , title_text="Numerical Data Violin Plots", showlegend=False, paper_bgcolor='white'

```

```

fig.show()

```

#### Voilin Plot displays the density of each value in each numerical columns

  

##  Plot 9: Distribution on Outcome with responded column

  

  

```

fig = px.histogram(

cleaned_data,

x='poutcome',

color='responded',

title='Response Based on Outcome',

labels={'poutcome': 'Outcome of Marketing', 'count': 'Count of Customers'},

color_discrete_sequence=px.colors.qualitative.Plotly

)

```

```

# Update layout for better visualization

fig.update_layout(

plot_bgcolor='lightblue', # Background color

xaxis_title='Outcome of Marketing',

yaxis_title='Count of Customers',

barmode='group'

)

```

```

# Rotate x-axis labels for better readability

fig.update_xaxes(tickangle= 90 )

```

```

# Show the plot

fig.show()

```

##  Plot 10: Plot for Response Based on Previous Contact Month

  

```

# Create a count plot using Plotly

fig = px.histogram(

cleaned_data,

x='pmonths',

color='responded',

title='Response Based on Previous Contact Month',

labels={'pmonths': 'Outcome of Previous Month Contacted', 'count': 'Count of Customers'},

color_discrete_sequence=px.colors.sequential.Pinkyl

)

```

```

# Update layout for better visualization

fig.update_layout(

plot_bgcolor='black', # Background color

xaxis_title='Outcome of Previous Month Contacted',

yaxis_title='Count of Customers',

barmode='group'

)

```

```

# Rotate x-axis labels for better readability

fig.update_xaxes(tickangle= 35 )

```

```

# Show the plot

fig.show()

```

#### Never contacted customers have responded more than contacted customers

  

##  Plot 11: Response Based on Contact Type

  

```

# Create a count plot using Plotly

fig = px.histogram(

cleaned_data,

x='contact',

color='responded',

title='Response Based on Contact Type',

labels={'contact': 'Outcome of Contact Type', 'count': 'Count of Customers'},

color_discrete_sequence=['red', 'black'] # Pastel colors

)

```

```

# Update layout for better visualization

fig.update_layout(

plot_bgcolor='cyan', # Background color

xaxis_title='Outcome of Contact Type',

yaxis_title='Count of Customers',

barmode='group'

)

```

```

# Show the plot

fig.show()

```

#### Customers contacted via cellular responded more than contacted via email

  

  

##  Plot 12: Distribution of employee variance rate on target column

  

```

# Create a histogram using Plotly Express

fig = px.histogram(

cleaned_data,

x='emp.var.rate',

color='responded',

barmode='group',

title='Distribution of Employee Variance Rate',

labels={'emp.var.rate': 'Employee Variance Rate', 'count': 'Count of Customers'},

color_discrete_sequence=px.colors.qualitative.Plotly

)

```

```

# Update layout for better visualization

fig.update_layout(

plot_bgcolor='lavender', # Background color

xaxis_title='Employee Variance Rate',

yaxis_title='Count of Customers'

)

```

```

# Show the plot

fig.show()

```

#### Customers with -1.8 variance rate and 1.4 variance rate responded more

  

##  Plot 13: Distribution on No. of Employees

  

```

# Create a bar chart using Plotly

fig = px.histogram(

cleaned_data,

x='nr.employed',

color='responded',

barmode='group',

title='Distribution on No. of Employees',

labels={'nr.employed': 'No. of Employees', 'count': 'Count of Customers'},

color_discrete_sequence=px.colors.qualitative.Plotly

)

```

```

# Update layout for better visualization

fig.update_layout(

plot_bgcolor='lightcyan', # Background color

xaxis_title='No. of Employees',

yaxis_title='Count of Customers',

xaxis={'categoryorder':'total descending'} # Updating the x-axis category order

)

```

```

# Rotate x-axis labels for better readability

fig.update_xaxes(tickangle= 35 )

```

```

# Show the plot

fig.show()

```

#### Customers with 5076.2 and 5099.1 and 5228.1 nr.employee rate responded more

  

##  Plot 14: Response on Contact Count

  

  

```

# Create a bar chart using Plotly

fig = px.histogram(

cleaned_data,

x='campaign',

color='responded',

barmode='group',

title='Distribution of Contacted Count',

labels={'campaign': 'Contacted Count', 'count': 'Count of Customers'},

color_discrete_sequence=px.colors.qualitative.Plotly

)

```

```

# Update layout for better visualization

fig.update_layout(

plot_bgcolor='black', # Background color

xaxis_title='Contacted Count',

yaxis_title='Count of Customers'

)

```

```

# Show the plot

fig.show()

```

#### Customers contacted once and twice responded more

  

##  Plot 15: Countplot to nd balance of data in target column

  

```

# Create a bar chart using Plotly

fig = px.histogram(

cleaned_data,

x='responded',

title='Balance Fit of Target Column',

labels={'responded': 'Response', 'count': 'Count of Customers'},

color='responded', # Color by the 'responded' variable

color_discrete_map={'Yes': '#FFD700', 'No': '#808080'} # Custom colors for 'Yes' and 'No'

)

```

```

# Update layout for better visualization

fig.update_layout(

plot_bgcolor='black', # Background color

xaxis_title='Response',

yaxis_title='Count of Customers'

)

```

```

# Show the plot

fig.show()

```

#### The countplot clearly displays that the data is highly imbalanced in the target column

  

##  Feature Engineering

  

#### As the categorical columns are related opted for Label Encoding and for Scaling opted for Standar Scaler

  

##  Encoding the Categorical Columns

  

```

## Define encoder

encoder = LabelEncoder()

```

```

for col in categorical:

cleaned_data[col] =  encoder.fit_transform(cleaned_data[col])

```

```

cleaned_data

```

##  Scaling the Numerical Columns

  

  

```

## Define Scaler

sc_x = StandardScaler()

```

```

## Perform Scaling using fit_transform

cleaned_data[numerical] = sc_x.fit_transform(cleaned_data[numerical])

```

```

## Display data after scaling

cleaned_data

```

##  Train and Test Split

  

```

## Define X and y

X = cleaned_data.drop('responded', axis = True)

y = cleaned_data['responded']

```

```

## Split train and test data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state= 42 )

```

```

## Print Shape of train and test

print(X_train.shape)

print(X_test.shape)

print(y_train.shape)

print(y_test.shape)

```

##  Sampling the Training Dataset

  

#### As the dat is highly imbalanced at target column we need to using sampling technique.

  

#### Going forward with Undersampling the training data and then Oversampling the undersampled data.

  

```

## Define Under Sampler

under = RandomUnderSampler(sampling_strategy=0.5)

```

```

## Define Over Sampler

over = SMOTE(sampling_strategy=0.5)

```

```

# Apply under-sampling to the training data

x_train_, y_train_ = under.fit_resample(X_train, y_train)

```

```

# Apply over-sampling to the under-sampled training data

x_train_resampled, y_train_resampled = over.fit_resample(X_train, y_train)

```

```

## Print resampled Data shape

print(x_train_resampled.shape)

print(y_train_resampled.shape)

```

#  Model Selection

  

```

- Logistic Regression

- Decision Tree Classifier

- Random Forest Classifier

- Gradient Boosting Classifier

- Support Vector Classifier

- K-Nearest Neighbors Classifier (KNN)

- XGBoost Classifier

- Neural Network Classifier

```
## Logistic Regression
Logistic Regression is a statistical method used for binary classification problems. It models the probability that a given input belongs to a particular class, using a logistic function to squeeze the output between 0 and 1. It's widely used for its simplicity, interpretability, and effectiveness on linearly separable datasets.

## Decision Tree Classifier
A Decision Tree Classifier is a non-parametric model that uses a tree-like structure of decisions and their possible consequences. It splits the dataset into subsets based on the value of input features, creating a model that predicts the target value. Decision trees are easy to understand and interpret but can be prone to overfitting.

## Random Forest Classifier
Random Forest is an ensemble learning method that constructs multiple decision trees during training and outputs the mode of the classes (classification) or mean prediction (regression) of the individual trees. It reduces overfitting by averaging multiple trees and improves accuracy and robustness.

## Gradient Boosting Classifier
Gradient Boosting is an ensemble technique that builds a model from weak learners (typically decision trees) in a sequential manner. Each new model attempts to correct the errors of the previous one. Gradient Boosting is known for its high predictive accuracy but can be computationally intensive and prone to overfitting without proper regularization.

## Support Vector Classifier (SVC)
Support Vector Classifier is a type of Support Vector Machine (SVM) used for classification tasks. It works by finding the hyperplane that best separates the data into different classes. SVCs are effective in high-dimensional spaces and are robust to overfitting, especially in scenarios with a clear margin of separation.

## K-Nearest Neighbors Classifier (KNN)
K-Nearest Neighbors is a simple, instance-based learning algorithm that classifies a data point based on how its neighbors are classified. It assigns the most common class among its k nearest neighbors, making it intuitive and easy to implement. However, KNN can be computationally expensive and sensitive to irrelevant or redundant features.

## XGBoost Classifier
XGBoost (Extreme Gradient Boosting) is an optimized implementation of gradient boosting designed for speed and performance. It includes several advanced features such as regularization to prevent overfitting, parallel processing, and handling missing values, making it a popular choice in competitive machine learning.

## Neural Network Classifier
Neural Network Classifiers are models inspired by the human brain, consisting of layers of neurons that process input features to predict the target class. They are powerful and flexible, capable of capturing complex patterns in data. However, they require significant computational resources and large amounts of data to train effectively.

## Classication Models Considered for Training

  

#  Model Training

  

##  1. Logistic Regression Classier

  

  

## Define Model

lgr = LogisticRegression()

  

# Train Model

lgr.fit(x_train_resampled,y_train_resampled)

  

## Predict Test data

lgr_pred = lgr.predict(X_test)

  

# Performance Evaluation

  

# Accuracy Score

lgr_accuracy = accuracy_score(y_test, lgr_pred)

print("-"* 50 )

print("Logistic Regression Model Accuracy:", lgr_accuracy)

  

# ROC AUC Score

lgr_roc = roc_auc_score(y_test, lgr_pred)

print("-"* 50 )

print("Logistic Regression Model ROC AUC Score:", lgr_roc)

  

### Classification Report

print("-"* 50 )

print("Logistic Regression Model Classiffication Report: \n\n",classification_report(y_test, lgr_pred))

print("-"* 50 )

  

# Calculate performance metrics

lgr_accuracy = accuracy_score(y_test, lgr_pred)

lgr_roc = roc_auc_score(y_test, lgr_pred)

lgr_classification_report = classification_report(y_test, lgr_pred, output_dict=True)

  

# Confusion Matrix

lgr_confusion_matrix = confusion_matrix(y_test, lgr_pred)

  

# Plotly figure for the confusion matrix heatmap

fig = go.Figure(data=go.Heatmap(

z=lgr_confusion_matrix,

x=['Predicted Negative', 'Predicted Positive'],

y=['Actual Negative', 'Actual Positive'],

colorscale='Viridis', # Adjust the colorscale as needed ('Viridis', 'YlGnBu', etc.)

hoverongaps=False,

hovertemplate='Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>'

))

  

# Update layout

fig.update_layout(

title='Logistic Regression Confusion Matrix',

xaxis_title='Predicted Label',

yaxis_title='Actual Label',

width= 600 ,

height= 400 ,

)

  

# Show the interactive plot

fig.show()

  

  

```

# Calculate ROC curve

fpr, tpr, thresholds = roc_curve(y_test, lgr_pred)

roc_auc = auc(fpr, tpr)

```

```

# Plotly figure for ROC curve

fig = go.Figure()

```

```

# ROC curve trace

fig.add_trace(go.Scatter(

x=fpr, y=tpr,

mode='lines',

name=f'ROC curve (AUC = {roc_auc:.2f})',

line=dict(color='darkorange', width= 2 )

))

```

```

# Diagonal line (random classifier)

fig.add_trace(go.Scatter(

x=[ 0 ,  1 ], y=[ 0 ,  1 ],

mode='lines',

name='Random Classifier',

line=dict(color='navy', width= 2 , dash='dash')

))

```

```

# Layout updates

fig.update_layout(

title='Receiver Operating Characteristic (ROC) Curve',

xaxis_title='False Positive Rate',

yaxis_title='True Positive Rate',

legend=dict(x=0.02, y=0.98),

margin=dict(l= 20 , r= 20 , t= 30 , b= 20 )

)

```

```

# Show the interactive plot

fig.show()

```

##  2. Decision Tree Classier

  

```

## Define Model

dtc = DecisionTreeClassifier()

```

```

# Train Model

dtc.fit(x_train_resampled,y_train_resampled)

```

```

## Predict Test data

dtc_pred = dtc.predict(X_test)

```

```

# Performance Evaluation

```

```

# Accuracy Score

dtc_accuracy = accuracy_score(y_test, dtc_pred)

print("-"* 50 )

print("Decision Tree Model Accuracy:", dtc_accuracy)

```

```

# ROC AUC Score

dtc_roc = roc_auc_score(y_test, dtc_pred)

print("-"* 50 )

print("Decision Tree Model ROC AUC Score:", dtc_roc)

```

```

### Classification Report

print("-"* 50 )

print("Decision Tree Model Classiffication Report: \n\n",classification_report(y_test, dtc_pred))

print("-"* 50 )

```

  

```

# Compute confusion matrix

dtc_confusion_matrix = confusion_matrix(y_test, dtc_pred)

```

```

# Plotly figure for the confusion matrix heatmap

fig = go.Figure(data=go.Heatmap(

z=dtc_confusion_matrix,

x=['Predicted Negative', 'Predicted Positive'],

y=['Actual Negative', 'Actual Positive'],

colorscale='Viridis', # Adjust the colorscale as needed ('Viridis', 'YlGnBu', etc.)

hoverongaps=False,

hovertemplate='Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>'

))

```

```

# Update layout

fig.update_layout(

title='Decision Tree Confusion Matrix',

xaxis_title='Predicted Label',

yaxis_title='Actual Label',

width= 600 ,

height= 400 ,

)

```

```

# Show the interactive plot

fig.show()

```

```

# Calculate ROC curve

fpr, tpr, thresholds = roc_curve(y_test, dtc_pred)

```

```

# Calculate AUC

roc_auc = auc(fpr, tpr)

```

```

# Plotly figure for ROC curve

fig = go.Figure()

fig.add_trace(go.Scatter(

x=fpr, y=tpr,

mode='lines',

name=f'ROC curve (AUC = {roc_auc:.2f})',

line=dict(color='darkorange', width= 2 )

))

fig.add_trace(go.Scatter(

x=[ 0 ,  1 ], y=[ 0 ,  1 ],

mode='lines',

line=dict(color='navy', width= 2 , dash='dash'),

showlegend=False

))

fig.update_layout(

title='Receiver Operating Characteristic (ROC) Curve',

xaxis_title='False Positive Rate',

yaxis_title='True Positive Rate',

width= 1200 , # Adjust width as needed

height= 500 , # Adjust height as needed

)

```

```

# Show the interactive plot

fig.show()

```

##  3. Random Forest Classier

  

```

## Define Model

rfc = RandomForestClassifier()

```

```

# Train Model

rfc.fit(x_train_resampled,y_train_resampled)

```

```

## Predict Test data

rfc_pred = rfc.predict(X_test)

```

  

```

# Performance Evaluation

```

```

# Accuracy Score

rfc_accuracy = accuracy_score(y_test, rfc_pred)

print("-"* 50 )

print("Random Forest Model Accuracy:", rfc_accuracy)

```

```

# ROC AUC Score

rfc_roc = roc_auc_score(y_test, rfc_pred)

print("-"* 50 )

print("Random Forest Model ROC AUC Score:", rfc_roc)

```

```

# Classification Report

print("-"* 50 )

print("Random Forest Model Classiffication Report: \n\n",classification_report(y_test, rfc_pred))

print("-"* 50 )

```

```

# Calculate confusion matrix

rfc_confusion_matrix = confusion_matrix(y_test, rfc_pred)

```

```

# Plotly figure for confusion matrix heatmap

fig = go.Figure(data=go.Heatmap(

z=rfc_confusion_matrix,

x=['Predicted Negative', 'Predicted Positive'],

y=['Actual Negative', 'Actual Positive'],

colorscale='Viridis', # Adjust the colorscale as needed

hoverongaps=False,

hovertemplate='Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>'

))

```

```

# Update layout

fig.update_layout(

title='Random Forest Confusion Matrix',

xaxis_title='Predicted Label',

yaxis_title='Actual Label',

width= 600 , # Adjust width as needed

height= 400 , # Adjust height as needed

)

```

```

# Show the interactive plot

fig.show()

```

```

# Calculate fpr, tpr, thresholds and AUC

fpr, tpr, thresholds = roc_curve(y_test, rfc_pred)

roc_auc = auc(fpr, tpr)

```

```

# Plotting ROC Curve using Plotly

fig = go.Figure()

fig.add_trace(go.Scatter(x=fpr, y=tpr,

mode='lines',

name='ROC curve (area = {:.2f})'.format(roc_auc)))

fig.add_trace(go.Scatter(x=[ 0 ,  1 ], y=[ 0 ,  1 ],

mode='lines',

line=dict(dash='dash'),

name='Random Guessing'))

fig.update_layout(

title='Receiver Operating Characteristic (ROC) Curve',

xaxis_title='False Positive Rate',

yaxis_title='True Positive Rate',

width= 1200 , # Adjust width as needed

height= 500 , # Adjust height as needed

)

fig.show()

```

##  4. Gradient Boosting Classier

  

```

# Define Model

gbc = GradientBoostingClassifier()

```

```

# Train Model

gbc.fit(x_train_resampled,y_train_resampled)

```

```

# Predict Test data

gbc_pred = gbc.predict(X_test)

```

  

```

# Performance Evaluation

```

```

# Accuracy Score

gbc_accuracy = accuracy_score(y_test, gbc_pred)

print("-"* 50 )

print("Gradient Boosting Model Accuracy:", gbc_accuracy)

```

```

# ROC AUC Score

gbc_roc = roc_auc_score(y_test, gbc_pred)

print("-"* 50 )

print("Gradient Boosting Model ROC AUC Score:", gbc_roc)

```

```

# Classification Report

print("-"* 50 )

print("Gradient Boosting Model Classiffication Report: \n\n",classification_report(y_test, gbc_pred))

print("-"* 50 )

```

```

# Calculate ROC curve and AUC

fpr_gbc, tpr_gbc, thresholds_gbc = roc_curve(y_test, gbc_pred)

roc_auc_gbc = auc(fpr_gbc, tpr_gbc)

```

```

# Plot ROC curve using Plotly

fig_roc = go.Figure()

fig_roc.add_trace(go.Scatter(x=fpr_gbc, y=tpr_gbc,

mode='lines',

name='ROC curve (area = {:.2f})'.format(roc_auc_gbc)))

fig_roc.add_trace(go.Scatter(x=[ 0 ,  1 ], y=[ 0 ,  1 ],

mode='lines',

line=dict(dash='dash'),

name='Random Guessing'))

fig_roc.update_layout(

title='Receiver Operating Characteristic (ROC) Curve - Gradient Boosting Classifier',

xaxis_title='False Positive Rate',

yaxis_title='True Positive Rate',

width= 1200 ,

height= 600 ,

)

fig_roc.show()

```

```

# Calculate confusion matrix

gbc_confusion_matrix = confusion_matrix(y_test, gbc_pred)

```

```

# Create a Plotly figure for the confusion matrix heatmap

fig = go.Figure(data=go.Heatmap(

z=gbc_confusion_matrix,

x=['Predicted Negative', 'Predicted Positive'],

y=['Actual Negative', 'Actual Positive'],

colorscale='Viridis', # Adjust the colorscale as needed ('Blues', 'Greens', 'Reds', etc.)

hoverongaps=False,

hovertemplate='Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>'

))

```

```

# Update layout

fig.update_layout(

title='Gradient Boosting Confusion Matrix',

xaxis_title='Predicted Label',

yaxis_title='Actual Label',

width= 600 ,

height= 400 ,

)

```

```

# Show the interactive plot

fig.show()

```

##  5. Support Vector Machines

  

```

# Define Model

svc = SVC()

```

```

# Train Model

svc.fit(x_train_resampled,y_train_resampled)

```

```

## Predict Test data

svc_pred = svc.predict(X_test)

```

  

# Performance Evaluation

  

# Accuracy Score

svc_accuracy = accuracy_score(y_test, svc_pred)

print("-"* 50 )

print("Support Vector Model Accuracy:", svc_accuracy)

  

# ROC AUC Score

svc_roc = roc_auc_score(y_test, svc_pred)

print("-"* 50 )

print("Support Vector Model ROC AUC Score:", svc_roc)

  

# Classification Report

print("-"* 50 )

print("Support Vector Model Classiffication Report: \n\n",classification_report(y_test, svc_pred))

print("-"* 50 )

  

# Calculate confusion matrix

svc_confusion_matrix = confusion_matrix(y_test, svc_pred)

  

# Create a Plotly figure for the confusion matrix heatmap

fig = go.Figure(data=go.Heatmap(

z=svc_confusion_matrix,

x=['Predicted Negative', 'Predicted Positive'],

y=['Actual Negative', 'Actual Positive'],

colorscale='YlGnBu',

hoverongaps=False,

hovertemplate='Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>'

))

  

# Update layout

fig.update_layout(

title='Support Vector Classifier Confusion Matrix',

xaxis_title='Predicted Label',

yaxis_title='Actual Label',

width= 600 ,

height= 400 ,

)

  

# Show the interactive plot

fig.show()

  

# Calculate fpr, tpr, thresholds

fpr, tpr, thresholds = roc_curve(y_test, svc_pred)

  

# Calculate AUC

roc_auc = auc(fpr, tpr)

  

# Create Plotly figure for ROC curve

fig = go.Figure()

  

# Add ROC curve

fig.add_trace(go.Scatter(x=fpr, y=tpr,

mode='lines',

name='ROC curve (AUC = {:.2f})'.format(roc_auc),

line=dict(color='darkorange', width= 2 )))

  

# Add diagonal line

fig.add_trace(go.Scatter(x=[ 0 ,  1 ], y=[ 0 ,  1 ],

mode='lines',

line=dict(color='navy', width= 2 , dash='dash'),

showlegend=False))

  

# Update layout

fig.update_layout(

title='Receiver Operating Characteristic (ROC) Curve',

xaxis_title='False Positive Rate',

yaxis_title='True Positive Rate',

width= 1200 ,

height= 600 ,

legend=dict(x=0.02, y=0.98, bgcolor='rgba(255, 255, 255, 0.5)'),

margin=dict(l= 0 , r= 0 , t= 30 , b= 0 ),

hovermode='closest'

)

  

# Show plot

fig.show()

  

  

##  6. K Nearest Neighbors

  

```

# Define Model

knn = KNeighborsClassifier()

```

```

# Train Model

knn.fit(x_train_resampled,y_train_resampled)

```

```

# Predict Test data

knn_pred = knn.predict(X_test)

```

```

# Performance Evaluation

```

```

# Accuracy Score

knn_accuracy = accuracy_score(y_test, knn_pred)

print("-"* 50 )

print("K-Nearest Neighbours Model Accuracy:", knn_accuracy)

```

```

# ROC AUC Score

knn_roc = roc_auc_score(y_test, knn_pred)

print("-"* 50 )

print("K-Nearest Neighbours Model ROC AUC Score:", knn_roc)

```

```

# Classification Report

print("-"* 50 )

print("K-Nearest Neighbours Model Classiffication Report: \n\n",classification_report(y_test, knn_pred))

print("-"* 50 )

```

```

# Confusion Matrix

knn_confusion_matrix = confusion_matrix(y_test, knn_pred)

```

```

# Create a Plotly figure for the confusion matrix heatmap

fig = go.Figure(data=go.Heatmap(

z=knn_confusion_matrix,

x=['Predicted Negative', 'Predicted Positive'],

y=['Actual Negative', 'Actual Positive'],

colorscale='Viridis',

hoverongaps=False,

hovertemplate='Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>',

text=knn_confusion_matrix

))

```

```

# Update layout

fig.update_layout(

title='K-Nearest Neighbours Confusion Matrix',

xaxis_title='Predicted Label',

yaxis_title='Actual Label',

width= 600 ,

height= 400 ,

)

```

```

# Show the interactive plot

fig.show()

```

  

```

# Define fpr, tpr, and thresholds

fpr, tpr, thresholds = roc_curve(y_test, knn_pred)

```

```

# Define AUC

roc_auc = auc(fpr, tpr)

```

```

# Create a Plotly figure for the ROC curve

fig = go.Figure()

```

```

# Add the ROC curve

fig.add_trace(go.Scatter(

x=fpr, y=tpr,

mode='lines',

name='ROC curve (area = {:.2f})'.format(roc_auc),

line=dict(color='darkorange', width= 2 )

))

```

```

# Add the diagonal line

fig.add_trace(go.Scatter(

x=[ 0 ,  1 ], y=[ 0 ,  1 ],

mode='lines',

name='Diagonal line',

line=dict(color='navy', width= 2 , dash='dash')

))

```

```

# Update layout

fig.update_layout(

title='Receiver Operating Characteristic (ROC) Curve',

xaxis_title='False Positive Rate',

yaxis_title='True Positive Rate',

width= 1200 ,

height= 600 ,

legend=dict(

x=0.8,

y=0.2,

bgcolor='rgba(255, 255, 255, 0)',

)

)

```

```

# Show the interactive plot

fig.show()

```

##  7. XG Boosting

  

```

# Define Model

xgc  = XGBClassifier()

```

```

# Train Model

xgc.fit(x_train_resampled,y_train_resampled)

```

```

# Predict Test data

xgc_pred = xgc.predict(X_test)

```

```

# Performance Evaluation

```

```

# Accuracy Score

xgc_accuracy = accuracy_score(y_test, xgc_pred)

print("-"* 50 )

print("XG Boosting Model Accuracy:", xgc_accuracy)

```

```

# ROC AUC Score

xgc_roc = roc_auc_score(y_test, xgc_pred)

print("-"* 50 )

print("XG Boosting Model ROC AUC Score:", xgc_roc)

```

```

# Classification Report

print("-"* 50 )

print("XG Boosting Model Classiffication Report: \n\n",classification_report(y_test, xgc_pred))

print("-"* 50 )

```

  

```

# Define the confusion matrix

xgc_confusion_matrix = confusion_matrix(y_test, xgc_pred)

```

```

# Create a Plotly figure for the confusion matrix heatmap

fig = go.Figure(data=go.Heatmap(

z=xgc_confusion_matrix,

x=['Predicted Negative', 'Predicted Positive'],

y=['Actual Negative', 'Actual Positive'],

colorscale='Viridis', # You can choose different colorscales such as 'Viridis', 'Blues', etc.

hoverongaps=False,

hovertemplate='Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>'

))

```

```

# Update layout

fig.update_layout(

title='XG Boosting Confusion Matrix',

xaxis_title='Predicted Label',

yaxis_title='Actual Label',

width= 600 ,

height= 400 ,

)

```

```

# Show the interactive plot

fig.show()

```

```

# Define fpr, tpr and thresholds

fpr, tpr, thresholds = roc_curve(y_test, xgc_pred)

```

```

# Define AUC

roc_auc = auc(fpr, tpr)

```

```

# Create a Plotly figure for the ROC curve

fig = go.Figure()

```

```

# Add the ROC curve

fig.add_trace(go.Scatter(

x=fpr, y=tpr,

mode='lines',

line=dict(color='darkorange', width= 2 ),

name=f'ROC curve (area = {roc_auc:.2f})'

))

```

```

# Add the diagonal line

fig.add_trace(go.Scatter(

x=[ 0 ,  1 ], y=[ 0 ,  1 ],

mode='lines',

line=dict(color='navy', width= 2 , dash='dash'),

showlegend=False

))

```

```

# Update layout

fig.update_layout(

title='Receiver Operating Characteristic (ROC) Curve',

xaxis_title='False Positive Rate',

yaxis_title='True Positive Rate',

width= 1200 ,

height= 600

)

```

```

# Show the interactive plot

fig.show()

```

##  8. Neural Networks

  

```

# Define Model

nnc = MLPClassifier()

```

```

# Train Model

nnc.fit(x_train_resampled,y_train_resampled)

```

```

# Predict Test data

nnc_pred = nnc.predict(X_test)

```

  

# Performance Evaluation

  

# Accuracy Score

nnc_accuracy = accuracy_score(y_test, nnc_pred)

print("-"* 50 )

print("Neural Network Model Accuracy:", nnc_accuracy)

  

# ROC AUC Score

nnc_roc = roc_auc_score(y_test, nnc_pred)

print("-"* 50 )

print("Neural Network Model ROC AUC Score:", nnc_roc)

  

# Classification Report

print("-"* 50 )

print("Neural Network Model Classiffication Report: \n\n",classification_report(y_test, nnc_pred))

print("-"* 50 )

  

# Define confusion matrix

nnc_confusion_matrix = confusion_matrix(y_test, nnc_pred)

  

# Create a Plotly figure for the confusion matrix heatmap

fig = go.Figure(data=go.Heatmap(

z=nnc_confusion_matrix,

x=['Predicted Negative', 'Predicted Positive'],

y=['Actual Negative', 'Actual Positive'],

colorscale='Viridis', # Mimicking 'Pastel2'

hoverongaps=False,

showscale=False,

hovertemplate='Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>'

))

  

# Update layout

fig.update_layout(

title='Neural Network Confusion Matrix',

xaxis_title='Predicted Label',

yaxis_title='Actual Label',

width= 600 ,

height= 400

)

  

# Show the interactive plot

fig.show()

  

  

```

# Define fpr, tpr, and thresholds

fpr, tpr, thresholds = roc_curve(y_test, nnc_pred)

```

```

# Define AUC

roc_auc = auc(fpr, tpr)

```

```

# Create a Plotly figure for the ROC curve

fig = go.Figure()

```

```

# Add ROC curve

fig.add_trace(go.Scatter(

x=fpr,

y=tpr,

mode='lines',

line=dict(color='darkorange', width= 2 ),

name=f'ROC curve (area = {roc_auc:.2f})'

))

```

```

# Add diagonal line

fig.add_trace(go.Scatter(

x=[ 0 ,  1 ],

y=[ 0 ,  1 ],

mode='lines',

line=dict(color='navy', width= 2 , dash='dash'),

showlegend=False

))

```

```

# Update layout

fig.update_layout(

title='Receiver Operating Characteristic (ROC) Curve',

xaxis_title='False Positive Rate',

yaxis_title='True Positive Rate',

xaxis=dict(range=[0.0, 1.0]),

yaxis=dict(range=[0.0, 1.05]),

width= 1200 ,

height= 600 ,

legend=dict(x=0.8, y=0.2)

)

```

```

# Show the interactive plot

fig.show()

```

##  Compare Models and Find best Model

  

```

## List of tuples containing (model, accuracy)

model_accuracy_list = [('Logistic Regression', lgr_accuracy, lgr_roc), ("Decision Tree Classifier", dtc_accuracy, dtc_roc),

("Random Forest Classifier", rfc_accuracy, rfc_roc),

("Gradient Boosting Classifier", gbc_accuracy, gbc_roc),

('Support Vector Machines', svc_accuracy, svc_roc),

('KNeighbours Classifiers', knn_accuracy, knn_roc),

('XG Boosting', xgc_accuracy, xgc_roc), ('Neural Network', nnc_accuracy, nnc_roc)]

model_accuracy_list

```

```

## Threshold for minimum accuracy

min_accuracy_threshold = 0.75

```

```

best_model = None

best_accuracy = 0.0

```

```

## Find the best model with the highest accuracy

for model, accuracy, roc_score in model_accuracy_list:

if accuracy > best_accuracy:

best_accuracy = accuracy

best_model = model

best_roc = roc_score

```

```

## Best Model Display

if best_model is not None:

print("Best Model:", best_model)

print('-'* 50 )

print("Validation Accuracy of the Best Model:", best_accuracy)

print('-'* 50 )

print('ROC AUC Score of the Best Model:', best_roc)

else:

print("No model met the accuracy threshold.")

```

  

##  Hyperparameter Tuning

  

#### From the list of models, "XG Boosting Model" has the highest accuracy (0.9059) among all models. This model would be a good candidate for

  

#### hyperparameter tuning and cross-validation to further improve its performance.

  

##  Tuning : RandomSearchCV

  

```

# Create a XGBoostingClassifier

Xg_classifier = XGBClassifier()

```

```

# Define the parameter distribution

param_dist = {

'n_estimators': randint( 50 ,  200 ),

'max_depth': randint( 3 ,  10 ),

'learning_rate': uniform(0.01, 0.3),

'subsample': uniform(0.6, 0.4),

'colsample_bytree': uniform(0.6, 0.4),

'min_child_weight': randint( 1 ,  10 ),

}

```

```

# Create RandomizedSearchCV object

random_search = RandomizedSearchCV(

Xg_classifier,

param_distributions=param_dist,

n_iter= 10 , # Number of random combinations to try

scoring='accuracy', # Use an appropriate scoring metric

cv= 5 , # Number of cross-validation folds

verbose= 1 ,

n_jobs=-1, # Use all available CPU cores

random_state= 42

)

```

```

# Fit the model to the data

random_search.fit(X_train, y_train)

```

```

# Get the best parameters and best estimator

best_params = random_search.best_params_

print("Best Hyperparameters:", best_params)

```

```

best_estimator = random_search.best_estimator_

print('Best Model :' , best_estimator)

```

```

# Make predictions on the test set

y_pred = best_estimator.predict(X_test)

```

```

# Evaluate the best model

```

```

# Accuracy Score

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy of Best Model :", accuracy)

print('-'* 50 )

```

```

# ROC AUC Score

roc_auc = roc_auc_score(y_test, y_pred)

print("ROC AUC Score of Best Model:", roc_auc)

print('-'* 50 )

```

```

# CLassification Report

print("\nClassification Report:\n")

print(classification_report(y_test, y_pred))

print('-'* 50 )

```

```

# Calculate confusion matrix

conf_matrix = confusion_matrix(y_test, y_pred)

```

```

# Create a DataFrame from the confusion matrix for better visualization

conf_matrix_df = pd.DataFrame(conf_matrix,

index=['Actual Negative', 'Actual Positive'],

columns=['Predicted Negative', 'Predicted Positive'])

```

```

# Plot the confusion matrix heatmap

fig= px.imshow(confmatrixdf,

```

  

```

fig  px.imshow(conf_matrix_df,

text_auto=True,

color_continuous_scale='Viridis', # Use a compatible colormap

labels={'color': 'Count'},

x=['Predicted Negative', 'Predicted Positive'],

y=['Actual Negative', 'Actual Positive'])

```

```

fig.update_layout(

title='Best Model Confusion Matrix',

xaxis_title='Predicted Label',

yaxis_title='Actual Label',

coloraxis_showscale=False

)

```

```

fig.show()

```

#### There is an increase in accuracy of model performance with original training data

  

##  Cross Validation

  

```

# Create a KFold object

kf = KFold(n_splits= 5 , shuffle=True, random_state= 42 )

```

```

# Perform cross-validation

cv_results = cross_val_score(best_estimator, X_train, y_train, cv=kf, scoring='accuracy')

```

```

# Print the results

print("Cross-Validation Results:", cv_results)

print("Average Accuracy:", cv_results.mean())

```

#### The best estimator achieved an accuracy of approximately 91.3%, which is consistent with the cross-validation accuracy of 90.3%. This

  

#### suggests that the chosen hyperparameters provide a stable and reliable model with minimal variance across different folds.

  

#  Hyperparameter tuning and cross-validation results:

  

##  Feature Importance

  

```

feature_importances = best_estimator.feature_importances_

```

```

# Create a DataFrame to display feature importances

feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})

feature_importance = feature_importance_df.sort_values(by='Importance', ascending=True)

```

```

# Display feature importances

print(feature_importance)

```

```

fig = px.bar(feature_importance,

x='Importance',

y='Feature',

orientation='h',

title='Feature Importance in Random Forest')

```

```

fig.update_layout(

xaxis_title='Importance',

yaxis_title='Feature'

)

```

```

fig.show()

```

#### Feature importances indicate the contribution of each feature to the model's predictions.A higher importance value suggests a stronger  
#### influence on the model's decision-making.
