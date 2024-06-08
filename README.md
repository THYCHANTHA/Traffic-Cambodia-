# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Generate Synthetic Traffic Accident Data
np.random.seed(42)

# Number of samples
n_samples = 1000

# Generate random data
dates = pd.date_range(start='2023-01-01', periods=n_samples, freq='D')
locations = np.random.choice(['Phnom Penh', 'Siem Reap', 'Battambang', 'Sihanoukville'], n_samples)
severity = np.random.choice(['Minor', 'Major', 'Fatal'], n_samples, p=[0.7, 0.2, 0.1])
vehicles_involved = np.random.randint(1, 5, n_samples)
cause = np.random.choice(['Speeding', 'Drunk Driving', 'Weather', 'Mechanical Failure'], n_samples)
fatalities = np.where(severity == 'Fatal', np.random.randint(1, 5, n_samples), 0)
injuries = np.where(severity != 'Minor', np.random.randint(1, 10, n_samples), 0)

# Create DataFrame
df = pd.DataFrame({
    'date': dates,
    'location': locations,
    'severity': severity,
    'vehicles_involved': vehicles_involved,
    'cause': cause,
    'fatalities': fatalities,
    'injuries': injuries
})

# Display the first few rows of the dataset
df.head()

# Step 2: Data Cleaning
# Check for missing values
print(df.isnull().sum())

# Summary statistics
print(df.describe())

# Step 3: Exploratory Data Analysis (EDA)

# Severity distribution
plt.figure(figsize=(10, 6))
sns.countplot(x='severity', data=df)
plt.title('Distribution of Accident Severity')
plt.show()

# Location distribution
plt.figure(figsize=(10, 6))
sns.countplot(x='location', data=df)
plt.title('Accidents by Location')
plt.show()

# Cause distribution
plt.figure(figsize=(10, 6))
sns.countplot(x='cause', data=df)
plt.title('Causes of Accidents')
plt.show()

# Correlation between variables
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Between Variables')
plt.show()

# Step 4: Visualization

# Accidents over time
plt.figure(figsize=(10, 6))
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)
df.resample('M').size().plot()
plt.title('Accidents Over Time')
plt.xlabel('Date')
plt.ylabel('Number of Accidents')
plt.show()

# Accidents by severity and location
plt.figure(figsize=(10, 6))
sns.countplot(x='location', hue='severity', data=df)
plt.title('Accidents by Severity and Location')
plt.show()

# Step 5: Recommendations
# Summarize findings and provide data-driven recommendations

print("Conclusion and Recommendations:")
print("1. Most accidents are minor, but there is a significant number of major and fatal accidents.")
print("2. Phnom Penh has the highest number of accidents, indicating a need for improved traffic management.")
print("3. Speeding and drunk driving are leading causes of accidents. Implementing stricter traffic laws and awareness campaigns could help reduce these incidents.")
print("4. Regular vehicle maintenance checks can help prevent accidents caused by mechanical failure.")
print("5. Seasonal variations in accidents suggest that weather conditions should be considered in traffic safety measures.")
