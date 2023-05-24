import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#Importing Scipytest

from scipy.stats import *


# Reading and getting information of dataset

df = pd.read_csv("StudentsPerformance .csv")
print(df.head())
print(df.info())

# Checking for null values.
print(df.isna().sum())

#Checking for duplicate value.

print(df.duplicated())

#Checking unique value.

print(df.nunique())

print(df['gender'].unique())

print(df['race/ethnicity'].unique())

print(df['parental level of education'].unique())

print(df['lunch'].unique())

print(df['test preparation course'].unique())

print(df['math score'].unique())

print(df['reading score'].unique())

print(df['writing score'].unique())

'''Since there are no null value, duplicate values or out of range values present in the dataset we will move ahead and rename the columns.'''

#Renaming the columns.

df.rename(columns={"gender":"Gender","race/ethnicity":"Ethnicity","parental level of education":"Parent_Education","lunch":"Lunch","math score":"Math","reading score":"Reading","writing score":"Writing","test preparation course":"Pre_preparation"},inplace=True)

print(df.head())

#Checking 'Gender' Column
df['Gender'].value_counts().plot(kind='bar')
plt.show()

#Checking 'Ethnicity' Column
df['Ethnicity'].value_counts().plot(kind='bar')
plt.show()

#Removing the word 'group' from the 'Ethnicity' column for easier classification

df['Ethnicity'] = df['Ethnicity'].str.replace('group', '')

#Checking if changes has been made
df['Ethnicity']

# Changing Ethnicity and Parent_Education column into category data type

df['Ethnicity'] = df['Ethnicity'].astype('category')

df['Parent_Education'] = df['Parent_Education'].astype('category')

#Checking if changes has been made
print(df.dtypes)

#Capitalizing the first word in the 'Parent_Education' column

df['Parent_Education'] = df['Parent_Education'].str.title()

print(df['Parent_Education'])

# Cleaning up the Parent_Education column by giving it order according to the level of education of the parents

level_ordered = ["Some High School", "High School", "Some College", "Associate'S Degree", "Bachelor'S Degree", "Master'S Degree" ]

df['Parent_Education'] = df['Parent_Education'].astype('category')

df["Parent_Education"] = df["Parent_Education"].cat.set_categories(level_ordered, ordered=True)


#Checking if changes has been made
print(df['Parent_Education'].unique().sort_values())

#Making index as a column for identification

df.reset_index(inplace=True)
#Renaming the 'index' column to 'id'
df = df.rename(columns = {'index' : 'id'})

#Checking if changes has been made
print(df)

#Checking and understanding 'Parent_Education' column

df['Parent_Education'].value_counts().plot(kind='barh', color='orange')

plt.xlabel('Parent')
plt.ylabel('Level of Education')
plt.title('Amount of Parents in each Level of Education', weight='bold')
plt.show()

#Checking 'Lunch' column

df['Lunch'].value_counts().plot.pie(autopct='%1.0f%%')

plt.title('Standard Lunch vs Free/Reduced Lunch', weight='bold')
plt.ylabel('')
plt.show()

#Exploring Pre_preparation column

df['Pre_preparation'].value_counts().plot.pie(autopct='%1.0f%%')

plt.title('Completed Test Prep vs Not-Completed Test Prep', weight='bold')
plt.ylabel('')
plt.show()

# Looking at the test scores between subjects

test_score = df[['Math', 'Reading', 'Writing']]

test_plot = sns.boxplot(data=test_score)

test_plot.set_title('Comparison of Student Test Scores Between Subjects', y=1.03,
                   fontweight='heavy', size='x-large')

test_plot.set_xlabel('Subject', fontweight='bold')
test_plot.set_ylabel('Score', fontweight='bold')

plt.show()

#Specific details regarding the plot

print(df.describe())

# Plotting the distribution of all the test scores.

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
 
fig.suptitle('Test Scores Distribution', fontweight='heavy', size='xx-large')

# Plotting the distribution of all the test scores.

math_hist = sns.histplot(ax=axes[0], data=df, x='Math', bins = 20, kde=True, color='green')
reading_hist = sns.histplot(ax=axes[1], data=df, x='Reading', bins = 20, kde=True, color='blue')
writing_hist = sns.histplot(ax=axes[2], data=df, x='Writing', bins = 20, kde=True, color='orange')

math_hist.set_xlabel('Math Score', fontweight='bold')
math_hist.set_ylabel('Student', fontweight='bold')

reading_hist.set_xlabel('Reading Score', fontweight='bold')
reading_hist.set_ylabel('Student', fontweight='bold')

writing_hist.set_xlabel('Writing Score', fontweight='bold')
writing_hist.set_ylabel('Student', fontweight='bold')

plt.show()

#The distribution among Ethnicity

df['Ethnicity'].value_counts().plot.pie(autopct='%1.0f%%')

plt.title('The Distribution Among Ethnicity Group', weight='bold')
plt.ylabel('')

plt.show()

#Student's Performance Compared By Each Ethnicity/Group

#Distinction in Math scores between groups 

math_violin = sns.violinplot(data=df, x='Math', y='Ethnicity', inner='quartile', scale='count')

math_violin.set_xlabel('Math Score', fontweight='bold')
math_violin.set_ylabel('Ethnicity/Group', fontweight='bold')
math_violin.set_title('Math Score Comparison Between Ethnicity/Groups', fontweight='heavy')

plt.show()
#Distinction in Reading scores between groups 

reading_violin = sns.violinplot(data=df, x='Reading', y='Ethnicity', inner='quartile', scale='count')

reading_violin.set_xlabel('Reading Score', fontweight='bold')
reading_violin.set_ylabel('Ethnicity/Group', fontweight='bold')
reading_violin.set_title('Reading Score Comparison Between Ethnicity/Groups', fontweight='heavy')

plt.show()

#Distinction in Writing scores between groups 

writing_violin = sns.violinplot(data=df, x='Writing', y='Ethnicity', inner='quartile', scale='count')

writing_violin.set_xlabel('Writing Score', fontweight='bold')
writing_violin.set_ylabel('Ethnicity/Group', fontweight='bold')
writing_violin.set_title('Writing Score Comparison Between Ethnicity/Groups', fontweight='heavy')

plt.show()

#The Effect of Parental Level Of Education in Student's Performance

#The Effect of Parental Level Of Education in Student's Performance[Math]

parental_order = ["Some High School", "High School", "Some College", "Associate'S Degree", "Bachelor'S Degree", "Master'S Degree" ]

math_score_median = df.groupby('Parent_Education')['Math'].median()

parental_math = sns.boxplot(data=df, x='Parent_Education', y='Math', order=parental_order)
parental_math = sns.lineplot(data=math_score_median, color='green', linewidth = 3)

parental_math.set_xlabel('Parental Level of Education', fontweight='bold')
parental_math.set_ylabel('Math Score', fontweight='bold')
parental_math.set_title('Comparsion of Math Score Among The Education Level of Parent', fontweight='heavy', y= 1.05)

plt.xticks(rotation=10)

plt.show()

#The Effect of Parental Level Of Education in Student's Performance[Reading]

parental_order = ["Some High School", "High School", "Some College", "Associate'S Degree", "Bachelor'S Degree", "Master'S Degree" ]

reading_score_median = df.groupby('Parent_Education')['Reading'].median()

parental_reading = sns.boxplot(data=df, x='Parent_Education', y='Reading', order=parental_order)
parental_reading = sns.lineplot(data=reading_score_median, color='orange', linewidth = 3)

parental_reading.set_xlabel('Parental Level of Education', fontweight='bold')
parental_reading.set_ylabel('Reading Score', fontweight='bold')
parental_reading.set_title('Comparsion of Reading Score Among The Education Level of Parent', fontweight='heavy', y= 1.05)

plt.xticks(rotation=10)

plt.show()

#The Effect of Parental Level Of Education in Student's Performance[Writing]

parental_order = ["Some High School", "High School", "Some College", "Associate'S Degree", "Bachelor'S Degree", "Master'S Degree" ]

writing_score_median = df.groupby('Parent_Education')['Writing'].median()

parental_writing = sns.boxplot(data=df, x='Parent_Education', y='Writing', order=parental_order)
parental_writing = sns.lineplot(data=reading_score_median, color='yellow', linewidth = 3)

parental_writing.set_xlabel('Parental Level of Education', fontweight='bold')
parental_writing.set_ylabel('Writing Score', fontweight='bold')
parental_writing.set_title('Comparsion of Writing Score Among The Education Level of Parent', fontweight='heavy', y= 1.05)

plt.xticks(rotation=10)

plt.show()

#Changing the dataframe from wide to long data format

dflongdata = df[['id', 'Pre_preparation', 'Math', 'Reading', 'Writing']]

dflong = pd.melt(dflongdata, id_vars=('id', 'Pre_preparation'), value_vars=('Math', 'Reading', 'Writing'), var_name='test_subject', value_name='score')

print(dflong)

#Tidying up the test_subject column

dflong['test_subject'] = dflong['test_subject'].str.replace('_score','')

dflong['test_subject'] = dflong['test_subject'].str.title()

print(dflong['test_subject'])

# Summarizing the data for later visualization

dflong_median = dflong.groupby(['test_subject', 'Pre_preparation'], as_index = False)['score'].median()
print(dflong_median)

#Plotting the difference between students who completed the test_prep vs students who did not

test_prep_plot = sns.barplot(data=dflong_median, x='test_subject', y='score', hue='Pre_preparation')

test_prep_plot.set_xlabel('Test Subjects', fontweight='bold')
test_prep_plot.set_ylabel('Student Median Score', fontweight='bold')
test_prep_plot.set_title('Comparison Between Student Score who took the Prepartion Test', fontweight='heavy')

plt.legend(bbox_to_anchor=(1.02, 0.55), loc='upper left', borderaxespad=0)

plt.show()


#Hypothesis Testing

#Changing into data frame to long data for the calculation

dflongdata2 = df[['id', 'Parent_Education', 'Math',]]

dflong2 = pd.melt(dflongdata2, id_vars=('Parent_Education'), value_vars=('Math'), value_name='score')

print(dflong2)

#Dropping the Variable Column

dflong2.drop(columns='variable', inplace=True)
#Subsetting the data 

bachelor = dflong2[dflong2.Parent_Education == "Bachelor'S Degree"]
master = dflong2[dflong2.Parent_Education == "Master'S Degree"]

#Computing p_value
stat, p_value1 = ttest_ind(bachelor['score'], master['score'], equal_var=True)
print(p_value1)

#alpha Value is 0.05 or 5%
if p_value1 < 0.05:
    print('We Reject Null Hypothesis')
else:
    print('We accept Null Hypothesis')

#Subsetting the Data

some_high_schoolsc = dflong2[dflong2.Parent_Education == "Some High School"]
#Computing p_value
stat, p_value2 = ttest_ind(some_high_schoolsc['score'], master['score'], equal_var=True)
print(p_value2)

#alpha Value is 0.05 or 5%
if p_value2 < 0.05:
    print('We Reject Null Hypothesis')
else:
    print('We accept Null Hypothesis')


#Multiple Linear Regression Model

#Creating the variables of X and y

X = df[['Parent_Education','Reading']]
y = df['Writing']

#Transforming the parentalLOE column with Label Encoder

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()

X['Parent_Education'] = labelencoder.fit_transform(X['Parent_Education'])

print(X)

#Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=21)
#Fitting Multiple Linear Reggression to the Training Set

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Predictiing the Test Set Result

y_pred = regressor.predict(X_test)
print(y_pred)

#Calculating the Coefficients

regressor.coef_

#Calculating the Intercept

regressor.intercept_

#Calculating the R squared Value

from sklearn.metrics import r2_score
r2_score(y_test, y_pred)

#Actual vs Predited Test Scores

plt.scatter(y_test, y_pred)
plt.xlabel('Actual', fontweight='bold')
plt.ylabel('Predicted', fontweight = 'bold')
plt.title('Actual vs Predicted Test Scores', fontweight='heavy')

plt.show()
