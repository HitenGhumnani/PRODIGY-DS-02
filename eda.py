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
df['Age'].fillna(df['Age'].median(), inplace=True)
df.drop(columns=['Cabin'], inplace=True)
df['Embarked'].fillna('S', inplace=True)

# Convert categorical data
df = pd.get_dummies(df, columns=['Sex', 'Embarked'])

# Drop non-numeric columns for correlation matrix
df_numeric = df.drop(columns=['Name', 'Ticket'])

# Explore the data
plt.figure(figsize=(10, 6))
sns.histplot(df['Age'], bins=30)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Count')

plt.figure(figsize=(10, 6))
sns.countplot(x='Pclass', data=df)
plt.title('Class Distribution')
plt.xlabel('Pclass')
plt.ylabel('Count')

plt.figure(figsize=(10, 6))
sns.heatmap(df_numeric.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')

# Identify patterns and trends
plt.figure(figsize=(10, 6))
sns.barplot(x='Pclass', y='Survived', hue='Sex_male', data=df)
plt.title('Survival Rate by Class and Sex')
plt.xlabel('Pclass')
plt.ylabel('Survival Rate')

plt.figure(figsize=(10, 6))
sns.kdeplot(df[df['Survived'] == 1]['Age'], shade=True, label='Survived')
sns.kdeplot(df[df['Survived'] == 0]['Age'], shade=True, label='Not Survived')
plt.title('Survival Distribution by Age')
plt.xlabel('Age')
plt.ylabel('Density')
plt.legend()

# Show all plots at once
plt.show()