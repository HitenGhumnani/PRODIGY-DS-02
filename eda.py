# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('data.csv')

# Understand the data
print(df.head())
print(df.info())
print(df.describe())

# Handle missing values
print(df.isnull().sum())

# Fill missing values in 'Age' with the median age
df['Age'] = df['Age'].fillna(df['Age'].median())

# Drop the 'Cabin' column since it has too many missing values
df = df.drop(columns=['Cabin'])

# Fill missing values in 'Embarked' with the most common value 'S'
df['Embarked'] = df['Embarked'].fillna('S')

# Convert categorical data
df = pd.get_dummies(df, columns=['Sex', 'Embarked'])

# Explore the data
sns.histplot(df['Age'], bins=30)
plt.show()

sns.countplot(x='Pclass', data=df)
plt.show()

sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.show()

# Identify patterns and trends
sns.barplot(x='Pclass', y='Survived', hue='Sex_male', data=df)
plt.show()

sns.kdeplot(df[df['Survived'] == 1]['Age'], shade=True, label='Survived')
sns.kdeplot(df[df['Survived'] == 0]['Age'], shade=True, label='Not Survived')
plt.legend()
plt.show()