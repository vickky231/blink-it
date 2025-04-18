
Importing Dependencies

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

Loading Dataset

df=pd.read_excel("BlinkIt grocery Dataset.xlsx")

Analysing Dataset

df.shape
df.columns
df.isnull().sum()

df.info()

df.duplicated()

<class 'pandas.core.frame.DataFrame'>
RangeIndex: 8523 entries, 0 to 8522
Data columns (total 12 columns):
 #   Column                     Non-Null Count  Dtype  
---  ------                     --------------  -----  
 0   Item Fat Content           8523 non-null   object 
 1   Item Identifier            8523 non-null   object 
 2   Item Type                  8523 non-null   object 
 3   Outlet Establishment Year  8523 non-null   int64  
 4   Outlet Identifier          8523 non-null   object 
 5   Outlet Location Type       8523 non-null   object 
 6   Outlet Size                8523 non-null   object 
 7   Outlet Type                8523 non-null   object 
 8   Item Visibility            8523 non-null   float64
 9   Item Weight                7060 non-null   float64
 10  Sales                      8523 non-null   float64
 11  Rating                     8523 non-null   float64
dtypes: float64(4), int64(1), object(7)
memory usage: 799.2+ KB

0       False
1       False
2       False
3       False
4       False
        ...  
8518    False
8519    False
8520    False
8521    False
8522    False
Length: 8523, dtype: bool

Customer Behaviour Analysis

active_users = df['Item Identifier'].nunique()
user_retention_rate = (df[df['Rating'] >= 4]['Item Identifier'].nunique() / active_users) * 100
average_session_duration = df['Item Visibility'].mean()
popular_product_categories = df['Item Type'].value_counts()
plt.figure(figsize=(10, 5))
popular_product_categories.plot(kind='bar', color='skyblue')
plt.title('Popular Product Categories')
plt.xlabel('Category')
plt.ylabel('Count')
plt.show()

No description has been provided for this image

import matplotlib.pyplot as plt

# Calculate metrics
active_users = df['Item Identifier'].nunique()
user_retention_rate = (df[df['Rating'] >= 4]['Item Identifier'].nunique() / active_users) * 100
average_session_duration = df['Item Visibility'].mean()
popular_product_categories = df['Item Type'].value_counts()

# Plot a pie chart
plt.figure(figsize=(10, 5))
popular_product_categories.plot(kind='pie', autopct='%1.1f%%', colors=plt.cm.Paired.colors)
plt.title('Popular Product Categories')
plt.ylabel('')  # Hides the default y-label that pie charts often show
plt.show()

No description has been provided for this image
Sales Matrix Analysis

sales_metrics = df.groupby('Item Type')['Sales'].sum()
plt.figure(figsize=(10, 5))
sales_metrics.plot(kind='bar', color='green')
plt.title('Sales by Product Category')
plt.xlabel('Category')
plt.ylabel('Total Sales')
plt.show()

No description has been provided for this image

import matplotlib.pyplot as plt
sales_metrics = df.groupby('Item Type')['Sales'].sum()
plt.figure(figsize=(10, 5))
sales_metrics.plot(kind='pie', autopct='%1.1f%%', colors=plt.cm.Paired.colors)
plt.title('Sales by Product Category')
plt.ylabel('')  # Hides the y-label for a cleaner pie chart
plt.show()

No description has been provided for this image

Analysing Top Categories

category_sales = df.groupby('Item Type')['Sales'].sum().sort_values(ascending=False)

# Top 5 categories
top_categories = category_sales.head(5)
print("Top 5 Product Categories Based on Sales:")
print(top_categories)

Top 5 Product Categories Based on Sales:
Item Type
Fruits and Vegetables    178124.0810
Snack Foods              175433.9224
Household                135976.5254
Frozen Foods             118558.8814
Dairy                    101276.4616
Name: Sales, dtype: float64

# Plotting top 5 categories
plt.figure(figsize=(10, 6))
top_categories.plot(kind='bar', color='skyblue')
plt.title('Top 5 Product Categories by Sales')
plt.xlabel('Product Category')
plt.ylabel('Total Sales')
plt.xticks(rotation=45)
plt.show()

No description has been provided for this image
Delivery Performance Analysis

# Calculate metrics
average_delivery_time = df['Item Visibility'].mean()
on_time_delivery_rate = (df[df['Item Visibility'] < 0.05]['Item Visibility'].count() / len(df)) * 100

# Set the figure size
plt.figure(figsize=(12, 6))

# Plot histogram with density
sns.histplot(df['Item Visibility'], bins=30, kde=True, color='orange', stat="density")

# Add vertical lines for mean and median
plt.axvline(average_delivery_time, color='blue', linestyle='dashed', linewidth=1.5, label=f'Mean: {average_delivery_time:.2f}')
plt.axvline(df['Item Visibility'].median(), color='green', linestyle='dashed', linewidth=1.5, label=f'Median: {df['Item Visibility'].median():.2f}')

# Add title and labels
plt.title('Distribution of Item Visibility (Proxy for Delivery Time)')
plt.xlabel('Item Visibility')
plt.ylabel('Density')

# Add legend
plt.legend()

# Show plot
plt.show()

No description has been provided for this image
Improve Delivery Time

To analyze delivery times, we'll look at the Item Visibility (assumed as a proxy for delivery time in this case) and aim to identify ways to reduce high delivery times.

# Set the figure size
plt.figure(figsize=(12, 6))

# Plot histogram with density
sns.histplot(df['Item Visibility'], bins=30, kde=True, color='orange')

# Calculate mean and median
mean_visibility = df['Item Visibility'].mean()
median_visibility = df['Item Visibility'].median()

# Add vertical lines for mean and median
plt.axvline(mean_visibility, color='blue', linestyle='dashed', linewidth=1.5, label=f'Mean: {mean_visibility:.2f}')
plt.axvline(median_visibility, color='green', linestyle='dashed', linewidth=1.5, label=f'Median: {median_visibility:.2f}')

# Add annotations for mean and median
plt.text(mean_visibility + 0.01, plt.gca().get_ylim()[1]*0.9, f'Mean: {mean_visibility:.2f}', color='blue', fontsize=10)
plt.text(median_visibility + 0.01, plt.gca().get_ylim()[1]*0.8, f'Median: {median_visibility:.2f}', color='green', fontsize=10)

# Add title and labels
plt.title('Distribution of Item Visibility (Delivery Time Proxy)', fontsize=16)
plt.xlabel('Item Visibility', fontsize=14)
plt.ylabel('Frequency', fontsize=14)

# Add legend
plt.legend()

# Show plot
plt.show()

No description has been provided for this image

# Recommendations to improve delivery times
# Assuming the visibility data represents time, we may consider any value above a certain threshold as delayed.
# Determine a threshold for on-time delivery (e.g., Item Visibility < 0.05)
threshold = 0.05
on_time_deliveries = df[df['Item Visibility'] < threshold]
delayed_deliveries = df[df['Item Visibility'] >= threshold]

print(f"On-Time Deliveries: {len(on_time_deliveries)}")
print(f"Delayed Deliveries: {len(delayed_deliveries)}")

On-Time Deliveries: 4051
Delayed Deliveries: 4472

Suggestions for improvement:

Optimize delivery routes. Increase the number of delivery personnel during peak hours. Partner with more local suppliers to reduce delivery times.
Monitor and Improve On-Time Delivery Rate

We can calculate the on-time delivery rate and provide suggestions for improvement

# Calculate the on-time delivery rate
total_deliveries = len(df)
on_time_delivery_count = len(on_time_deliveries)
on_time_delivery_rate = (on_time_delivery_count / total_deliveries) * 100

print(f"On-Time Delivery Rate: {on_time_delivery_rate:.2f}%")

On-Time Delivery Rate: 47.53%

# Set the figure size
plt.figure(figsize=(10, 6))

# Create a bar plot
sns.barplot(x=delivery_status, y=delivery_counts, palette='viridis')

# Add annotations on the bars
for i, count in enumerate(delivery_counts):
    plt.text(i, count + max(delivery_counts) * 0.02, f'{count:,}', ha='center', fontsize=12)

# Add title and labels
plt.title('On-Time vs. Delayed Deliveries', fontsize=16, weight='bold')
plt.xlabel('Delivery Status', fontsize=14)
plt.ylabel('Number of Deliveries', fontsize=14)

# Improve the layout and style
sns.despine(trim=True)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Show the plot
plt.show()

C:\Users\aryan\AppData\Local\Temp\ipykernel_6524\3926425649.py:5: FutureWarning: 

Passing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0. Assign the `x` variable to `hue` and set `legend=False` for the same effect.

  sns.barplot(x=delivery_status, y=delivery_counts, palette='viridis')

No description has been provided for this image

delivery_status = ['On-Time', 'Delayed']
delivery_counts = [on_time_delivery_count, len(delayed_deliveries)]  # Ensure these variables are defined

# Pie chart for delivery status
plt.figure(figsize=(8, 8))

# Create the pie chart
plt.pie(delivery_counts, labels=delivery_status, autopct='%1.1f%%', colors=['#4CAF50', '#FF5722'], startangle=140, wedgeprops={'edgecolor': 'black'})

# Add title
plt.title('Proportion of On-Time vs. Delayed Deliveries', fontsize=16, weight='bold')

# Show plot
plt.show()

No description has been provided for this image
Recommendations to improve on-time delivery rate:

Implement real-time tracking and updates for customers. Provide incentives for delivery personnel to meet delivery time targets.-Analyze the reasons for delays and address common issues (e.g., traffic, weather)
Statistical Modelling

Predict potential demand based on historical sales data Feature selection: We'll use 'Item Weight' and 'Item Visibility' as features for simplicity

features = df[['Item Weight', 'Item Visibility']]
target = df['Sales']
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
# Remove rows with missing values in X_train and y_train
X_train_clean = X_train.dropna()
y_train_clean = y_train[X_train_clean.index]

# Fit the model
model = LinearRegression()
model.fit(X_train_clean, y_train_clean)

  LinearRegression
?
i

LinearRegression()

# Clean the test set by removing rows with missing values in X_test
X_test_clean = X_test.dropna()

# Ensure that y_test corresponds to the cleaned X_test
y_test_clean = y_test[X_test_clean.index]

# Predict on the cleaned test set
y_pred = model.predict(X_test_clean)

# Evaluation
mse = mean_squared_error(y_test_clean, y_pred)
r2 = r2_score(y_test_clean, y_pred)

# Print model evaluation metrics
print(f"Mean Squared Error: {mse}")
print(f"R2 Score: {r2}")

Mean Squared Error: 4044.944867107343
R2 Score: 0.0014024526577205432

# Visualization
# Set the visualisation style
sns.set_style("whitegrid")
accuracy = r2 * 100
print(f"Model Accuracy: {accuracy:.2f}%")

Model Accuracy: 0.14%

# Model Performance Visualization
plt.figure(figsize=(12, 8))

# Scatter plot of actual vs predicted values after cleaning the data
plt.scatter(y_test_clean, y_pred, color='dodgerblue', alpha=0.7, edgecolor='black', s=60, marker='o', label='Predicted vs Actual')

# Add a diagonal line representing perfect predictions
max_value = max(max(y_test_clean), max(y_pred))
min_value = min(min(y_test_clean), min(y_pred))
plt.plot([min_value, max_value], [min_value, max_value], color='darkorange', linestyle='--', linewidth=2, label='Perfect Prediction')

# Add annotations for mean squared error and R2 score
mse = mean_squared_error(y_test_clean, y_pred)
r2 = r2_score(y_test_clean, y_pred)
plt.text(min_value + (max_value - min_value) * 0.05, max_value - (max_value - min_value) * 0.1,
         f'MSE: {mse:.2f}\nR²: {r2:.2f}', fontsize=12, color='darkred', bbox=dict(facecolor='white', alpha=0.7, edgecolor='black'))

# Title and labels
plt.title('Actual vs. Predicted Sales', fontsize=18, weight='bold')
plt.xlabel('Actual Sales', fontsize=14)
plt.ylabel('Predicted Sales', fontsize=14)

# Add a grid for better readability
plt.grid(True, linestyle='--', alpha=0.6)

# Add legend
plt.legend()

# Set aspect ratio to be equal
plt.gca().set_aspect('equal', adjustable='box')

# Show plot
plt.show()

No description has been provided for this image

 

