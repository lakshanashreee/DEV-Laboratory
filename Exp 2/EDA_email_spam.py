# Step 1: Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import string

# Step 2: Load the dataset
df = pd.read_csv('spam.csv', encoding='latin-1')[['v1', 'v2']]
df.columns = ['label', 'message']

# Step 3: Dataset overview
print(df.head())
print(df.info())
print(df.describe())

# Step 4: Check for missing values
print("Missing values:\n", df.isnull().sum())

# Step 5: Class distribution
print("Label counts:\n", df['label'].value_counts())

# Plot spam vs ham
sns.countplot(data=df, x='label')
plt.title("Spam vs Ham Count")
plt.show()

# Step 6: Add message length column
df['length'] = df['message'].apply(len)

# Step 7: Histogram of message lengths
df['length'].plot.hist(bins=50)
plt.title("Message Length Distribution")
plt.xlabel("Length")
plt.show()

# Step 8: WordClouds
spam_words = ' '.join(df[df['label'] == 'spam']['message'])
ham_words = ' '.join(df[df['label'] == 'ham']['message'])

spam_wc = WordCloud(width=500, height=300, background_color='black').generate(spam_words)
ham_wc = WordCloud(width=500, height=300, background_color='white').generate(ham_words)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(spam_wc, interpolation='bilinear')
plt.title("Spam Word Cloud")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(ham_wc, interpolation='bilinear')
plt.title("Ham Word Cloud")
plt.axis('off')
plt.show()

# Step 9: Statistical insight
print("Average Message Length (Spam):", df[df['label']=='spam']['length'].mean())
print("Average Message Length (Ham):", df[df['label']=='ham']['length'].mean())
