import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

print("📥 Fetching Titanic dataset...")
try:
    data = pd.read_csv('https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv')
    print("✅ Dataset imported successfully!")
except Exception as err:
    print(f"❌ Could not load dataset: {err}")
    exit()

print("\n🧹 Cleaning the dataset...")

if 'Age' in data.columns:
    data['Age'] = data['Age'].fillna(data['Age'].median())
    print("✔ Filled missing 'Age' values.")

if 'Embarked' in data.columns:
    most_common_port = data['Embarked'].mode().iloc[0]
    data['Embarked'] = data['Embarked'].fillna(most_common_port)
    print("✔ Filled missing 'Embarked' values.")

if 'Cabin' in data.columns:
    data = data.drop(columns=['Cabin'])
    print("✔ Removed 'Cabin' column.")

print("✅ Data cleaning done.\n")

print("📊 Generating visualizations...")
sb.set_style("ticks")

plt.figure(figsize=(9, 5))
sb.countplot(data=data, x='Pclass', hue='Survived', palette='Set2')
plt.title('Passenger Class vs Survival', fontsize=15)
plt.xlabel('Class Type')
plt.ylabel('Passenger Count')
plt.legend(title='Survival', labels=['No', 'Yes'])
plt.savefig('plot_class_survival.png')
plt.show()

plt.figure(figsize=(9, 5))
sb.countplot(data=data, x='Sex', hue='Survived', palette='coolwarm')
plt.title('Gender vs Survival', fontsize=15)
plt.xlabel('Gender')
plt.ylabel('Passenger Count')
plt.savefig('plot_gender_survival.png')
plt.show()

plt.figure(figsize=(9, 5))
sb.histplot(data['Age'], bins=28, kde=True, color='purple')
plt.title('Passenger Age Distribution', fontsize=15)
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.savefig('plot_age_dist.png')
plt.show()

grid = sb.FacetGrid(data, col='Survived', height=5)
grid.map(sb.histplot, 'Age', bins=25, color='orange')
grid.set_axis_labels('Age', 'Count')
grid.fig.suptitle('Age vs Survival Status', y=1.03)
plt.savefig('plot_age_survival.png')
plt.show()

print("✅ All plots saved successfully!")
