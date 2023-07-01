#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')


# In[2]:


# Importing Dataset Telco
data=pd.read_csv("telcom_data.csv")


# In[3]:


data.head() 


# In[4]:


data.shape # Large dataset having 150001 rows and 55 columns


# In[5]:


data.columns


# In[6]:


len(data['Dur. (ms)'])


# In[7]:


df={"Bearer Id":data["Bearer Id"],'MSISDN/Number':data['MSISDN/Number'],'Start ms':data['Start ms'],'End ms':data['End ms'],'Dur. (ms)':data['Dur. (ms)'],
   'Dur. (ms)':data['Dur. (ms)'],'Avg RTT DL (ms)':data['Avg RTT DL (ms)'],'Avg RTT UL (ms)':data['Avg RTT UL (ms)'],
   'Activity Duration DL (ms)':data['Activity Duration DL (ms)'],
   'Activity Duration UL (ms)':data['Activity Duration UL (ms)'],}


# In[8]:


d=pd.DataFrame(df)


# In[9]:


d


# In[10]:


d.info()


#   

#   

#   

# # Sessions Frequency

#  In the context of network data or telecommunications, session frequency typically refers to the number of sessions or connections established by a specific user or device

# In[11]:



session_frequency = data.groupby(by=['MSISDN/Number'])['Dur. (ms)'].transform('count')


# In[ ]:





# In[12]:



len(session_frequency)


# In[13]:


# As the session frequency increases the count decreases that means count of session frquency is low


# In[14]:


session_frequency.plot(kind='hist', bins=10)

plt.xlabel('Session Frequency')
plt.ylabel('Count')
plt.title('Distribution of Session Frequency')
plt.show()


# In[15]:




# Create a distribution plot using distplot
sns.distplot(session_frequency, kde=True)

plt.xlabel('Session Frequency')
plt.ylabel('Density')
plt.title('Distribution of Session Frequency')

plt.show()



# # Session frquency based on hanset type and Manufacturer

# In[16]:


# Group the data by 'Handset Type' and 'MSISDN/Number' and count the number of unique sessions
session_f = data.groupby(['Handset Type', 'MSISDN/Number'])['Dur. (ms)'].count().reset_index()

# Rename the column containing the session frequency count
session_f.rename(columns={'Dur. (ms)': 'Session F'}, inplace=True)

# Group the data by 'Handset Type' and calculate the total session frequency
type_frequency = session_f.groupby('Handset Type')['Session F'].sum().reset_index()

# Sort the data by session frequency in descending order
type_frequency = type_frequency.sort_values('Session F', ascending=False).head(10)

# Plot the session frequency by handset type
plt.figure(figsize=(10, 6))
sns.barplot(data=type_frequency, x='Handset Type', y='Session F')
plt.xlabel('Handset Type')
plt.ylabel('Session Frequency')
plt.title('Top 10 Handset Types by Session Frequency')
plt.xticks(rotation=90)
plt.show()


# In[17]:


# Group the data by 'Handset Manufacturer' and 'MSISDN/Number' and count the number of unique sessions
session_frequency1 = data.groupby(['Handset Manufacturer', 'MSISDN/Number'])['Dur. (ms)'].count().reset_index()

# Rename the column containing the session frequency count
session_frequency1.rename(columns={'Dur. (ms)': 'Session Frequency1'}, inplace=True)

# Group the data by 'Handset Manufacturer' and calculate the total session frequency
manufacturer_frequency = session_frequency1.groupby('Handset Manufacturer')['Session Frequency1'].sum().reset_index()

# Sort the data by session frequency in descending order
manufacturer_frequency = manufacturer_frequency.sort_values('Session Frequency1', ascending=False).head(10)

# Plot the session frequency by handset manufacturer
plt.figure(figsize=(10, 6))
sns.barplot(data=manufacturer_frequency, x='Handset Manufacturer', y='Session Frequency1')
plt.xlabel('Handset Manufacturer')
plt.ylabel('Session Frequency1')
plt.title('Top 10 Handset Manufacturers by Session Frequency')
plt.xticks(rotation=90)
plt.show()


# In[18]:


# Group the data by 'Handset Type' and 'MSISDN/Number' and count the number of unique sessions
session_frequency_type = data.groupby(['Handset Type', 'MSISDN/Number'])['Dur. (ms)'].count().reset_index()

# Rename the column containing the session frequency count
session_frequency_type.rename(columns={'Dur. (ms)': 'Session F'}, inplace=True)

# Group the data by 'Handset Type' and calculate the total session frequency
type_frequency = session_frequency_type.groupby('Handset Type')['Session F'].sum().reset_index()

# Sort the data by session frequency in descending order
type_frequency = type_frequency.sort_values('Session F', ascending=False).head(10)

# Group the data by 'Handset Manufacturer' and 'MSISDN/Number' and count the number of unique sessions
session_frequency_manufacturer = data.groupby(['Handset Manufacturer', 'MSISDN/Number'])['Dur. (ms)'].count().reset_index()

# Rename the column containing the session frequency count
session_frequency_manufacturer.rename(columns={'Dur. (ms)': 'Session F'}, inplace=True)

# Group the data by 'Handset Manufacturer' and calculate the total session frequency
manufacturer_frequency = session_frequency_manufacturer.groupby('Handset Manufacturer')['Session F'].sum().reset_index()

# Sort the data by session frequency in descending order
manufacturer_frequency = manufacturer_frequency.sort_values('Session F', ascending=False).head(10)

# Plotting
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot session frequency by handset type
sns.barplot(data=type_frequency, x='Handset Type', y='Session F', ax=axes[0])
axes[0].set_xlabel('Handset Type')
axes[0].set_ylabel('Session Frequency')
axes[0].set_title('Top 10 Handset Types by Session Frequency')
axes[0].tick_params(axis='x', rotation=90)

# Plot session frequency by handset manufacturer
sns.barplot(data=manufacturer_frequency, x='Handset Manufacturer', y='Session F', ax=axes[1])
axes[1].set_xlabel('Handset Manufacturer')
axes[1].set_ylabel('Session Frequency')
axes[1].set_title('Top 10 Handset Manufacturers by Session Frequency')
axes[1].tick_params(axis='x', rotation=90)

plt.tight_layout()
plt.show()


# In[19]:


# Plotting a pie chart for session frequency by handset type
plt.figure(figsize=(8, 6))
plt.pie(type_frequency['Session F'], labels=type_frequency['Handset Type'], autopct='%1.1f%%')
plt.title('Session Frequency by  top 10 Handset Type')
plt.show()


# 
# 

#  

# # Session Duration 

# In[20]:


# Calculate session duration by subtracting "End ms" from "Start ms"
Session_Duration = data['Dur. (ms)']



# Print the session duration for each "Bearer Id"
print(Session_Duration)


# In[21]:


plt.figure(figsize=(10, 6))
plt.hist(Session_Duration, bins=10, edgecolor='black')
plt.title('Session Duration Distribution')
plt.xlabel('Session Duration (ms)')
plt.ylabel('Frequency')
plt.show()


# In[22]:


d.describe()


#   

#  

#  

#  

# # The session total traffic
# 
# 
# It calculates the total number of pages visited on a website in a single go.

# In[23]:


total_traffic = data['Total DL (Bytes)'] + data['Total UL (Bytes)']


# In[24]:


# adding the values of 'Total DL (Bytes)' and 'Total UL (Bytes)' columns and stores the result in a new column called 'Session_Total_Traffic'.


# In[25]:


plt.figure(figsize=(10, 6))
plt.hist(total_traffic, bins=10, edgecolor='black')
plt.title('Total Traffic Distribution')
plt.xlabel('Total Traffic')
plt.ylabel('Frequency')
plt.show()


# In[26]:


total_traffic


# In[27]:


plt.figure(figsize=(10, 6))
plt.scatter(Session_Duration, total_traffic)
plt.title('Total Traffic vs. Session Duration')
plt.xlabel('Session Duration')
plt.ylabel('Total Traffic')
plt.show()


# In[28]:




plt.figure(figsize=(10, 6))
sns.scatterplot(Session_Duration, total_traffic)
plt.title('Total Traffic vs. Session Duration')
plt.xlabel('Session Duration')
plt.ylabel('Total Traffic')
plt.show()


# In[29]:


d.columns


# In[30]:


b={'MSISDN/Number':data['MSISDN/Number'],"session_frequency":session_frequency,"session_duration":Session_Duration,'total_traffic':total_traffic}


# In[31]:


c=pd.DataFrame(b)


# In[32]:


c


# In[33]:



for i in c.columns:
   
    # check if the datatype is object or not
    if c[i].dtypes == "object":
        
        mode_value = c[i].mode()[0]
        c[i].fillna(mode_value, inplace = True)
    elif c[i].dtypes != "object":
        mean_value = c[i].mean()
        c[i].fillna(mean_value, inplace = True)


# In[34]:


c


# In[35]:


dframe=pd.concat([c,data],axis=0)
dframe


# In[36]:


dframe.columns


# # task 3    Experience Analytics

# #### Task 3.1 - Aggregate, per customer, the following information (treat missing & outliers by replacing with the mean or the mode of the corresponding variable):
# 
# #### • Average TCP retransmission
# #### • Average RTT
# #### • Handset type
# #### • Average throughput

# In[37]:


data.columns


# In[38]:


# Calculate average TCP retransmission
avg_tcp_retransmission = dframe.groupby('MSISDN/Number')[['TCP DL Retrans. Vol (Bytes)', 'TCP UL Retrans. Vol (Bytes)']].mean().mean(axis=1)
avg_tcp_retransmission.fillna(avg_tcp_retransmission.mean(), inplace=True)

# Calculate average RTT
avg_rtt = dframe.groupby('MSISDN/Number')[['Avg RTT DL (ms)', 'Avg RTT UL (ms)']].mean().mean(axis=1)
avg_rtt.fillna(avg_rtt.mean(), inplace=True)

# Replace missing values in 'Handset Type' with mode

handset_mode = dframe.groupby('MSISDN/Number')['Handset Type'].agg(lambda x: x.mode().values[0] if len(x.mode()) > 0 else None)

# Calculate average throughput
avg_throughput = dframe.groupby('MSISDN/Number')[['Avg Bearer TP DL (kbps)', 'Avg Bearer TP UL (kbps)']].mean().mean(axis=1)
avg_throughput.fillna(avg_throughput.mean(), inplace=True)

# Create a new DataFrame with aggregated information
aggregated_data = pd.DataFrame({
    'Avg_TCP_Retransmission': avg_tcp_retransmission,
    'Avg_RTT': avg_rtt,
    'Handset_Type': handset_mode,
    'Avg_Throughput': avg_throughput
})

# Reset the index to make 'MSISDN/Number' a column instead of the index
aggregated_data.reset_index(inplace=True)

# Display the aggregated data
print(aggregated_data)


# In[39]:


aggregated_data.info()


# In[40]:


aggregated_data.isnull().sum()


# In[41]:


aggregated_data.describe()


# # Filling Missing and null values

# In[42]:


# Filling the missing null values
for i in aggregated_data.columns:
    if  aggregated_data[i].dtypes != "object":
        mean_value = aggregated_data[i].mean()
        aggregated_data[i].fillna(mean_value, inplace = True)


# In[43]:


aggregated_data.isnull().sum()


# In[ ]:





# # Treating Outliers

# In[44]:


# Filter out non-numeric columns
numeric_cols = aggregated_data.select_dtypes(include=np.number).columns

# Plot boxplots for numeric columns
for col in numeric_cols:
    sns.boxplot(data=aggregated_data, x=col)
    plt.show()


# In[45]:


aggregated_data.columns


# In[46]:


for col in numeric_cols:
    sns.barplot(data=aggregated_data, x=col)
    plt.show()


# In[47]:


# Filter out non-numeric columns
numeric_cols = aggregated_data.select_dtypes(include=np.number).columns

# Calculate percentiles for numeric columns
q1 = np.percentile(aggregated_data[numeric_cols], 25)
q2 = np.percentile(aggregated_data[numeric_cols], 50)
q3 = np.percentile(aggregated_data[numeric_cols], 75)

print(f"My Q1 = {q1}, Q2 = {q2}, Q3 = {q3}")


# In[48]:


iqr=q3-q1
iqr


# In[49]:


lower_range=q1-iqr*1.5
upper_range=q3+iqr*1.5
print(f"Lower range= {lower_range} , Upper Range= {upper_range}")


# In[50]:


def find_outliers_iqr(df, threshold=1.5):
    outliers = pd.DataFrame()
    for column in df.columns:
        if df[column].dtype != 'object':  # Exclude non-numeric columns
            q1 = df[column].quantile(0.25)
            q3 = df[column].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            column_outliers = df[(df[column] < lower_bound) |(df[column] > upper_bound)]
            outliers = pd.concat([outliers, column_outliers])
            
            # Replace outliers with the mean
            df[column] = np.where((df[column] < lower_bound) | (df[column] > upper_bound),
                                  df[column].mean(),
                                  df[column])
            
    return outliers


# In[51]:


data=find_outliers_iqr(aggregated_data)


# ##### treat outliers by replacing with the mean  of the corresponding variable

# In[52]:



numeric_cols = aggregated_data.select_dtypes(include=np.number).columns

# Plot boxplots for numeric columns
for col in numeric_cols:
    sns.boxplot(data=data, x=col)
    plt.show()


# In[ ]:





# # Task 3.2 - Compute & list 10 of the top, bottom and most frequent:
# 
# #### a. TCP values in the dataset. 
# #### b. RTT values in the dataset.
# #### c. Throughput values in the dataset.

# In[53]:


# Univariate Analysis
categorical_data = []
for col in data.columns:
    if data[col].dtypes != "object":
        categorical_data.append(col)


# In[54]:


for i in data.columns:
    print("*"*50)
    print("The datatype for {} is {}".format(i,data[i].dtypes))
    # check if the datatype is object or not
    if data[i].dtypes != "object":
        print("Top 10 categories for ", i )
        print(data[i].value_counts().sort_values(ascending = False).head(10))
        print("*"*50)


# In[55]:


for i in data.columns:
    print("*"*50)
    print("The datatype for {} is {}".format(i,data[i].dtypes))
    # check if the datatype is object or not
    if data[i].dtypes != "object":
        print("Bottom 10 categories for ", i )
        print(data[i].value_counts().sort_values(ascending = True).head(10))
        print("*"*50)


# In[56]:


# Top 10 TCP values
top_10_tcp = data['Avg_TCP_Retransmission'].nlargest(10)
print("Top 10 TCP values:")
print(top_10_tcp)
print(50*"*")

# Bottom 10 TCP values
print('\n',50*"*")
bottom_10_tcp = data['Avg_TCP_Retransmission'].nsmallest(10)
print("Bottom 10 TCP values:")
print(bottom_10_tcp)
print('\n',50*"*")

# Most frequent TCP values
print('\n',50*"*")
most_frequent_tcp = data['Avg_TCP_Retransmission'].value_counts().head(10)
print("Most frequent TCP values:")
print(most_frequent_tcp)
print('\n',50*"*")

# Top 10 RTT values
print('\n',50*"*")
top_10_rtt = data['Avg_RTT'].nlargest(10)
print("Top 10 RTT values:")
print(top_10_rtt)
print('\n',50*"*")

# Bottom 10 RTT values
print('\n',50*"*")
bottom_10_rtt = data['Avg_RTT'].nsmallest(10)
print("Bottom 10 RTT values:")
print(bottom_10_rtt)
print('\n',50*"*")

# Most frequent RTT values
print('\n',50*"*")
most_frequent_rtt = data['Avg_RTT'].value_counts().head(10)
print("Most frequent RTT values:")
print(most_frequent_rtt)
print('\n',50*"*")

# Top 10 throughput values
print('\n',50*"*")
top_10_throughput = data['Avg_Throughput'].nlargest(10)
print("Top 10 throughput values:")
print(top_10_throughput)
print('\n',50*"*")

# Bottom 10 throughput values
print('\n',50*"*")
bottom_10_throughput = data['Avg_Throughput'].nsmallest(10)
print("Bottom 10 throughput values:")
print(bottom_10_throughput)
print('\n',50*"*")

# Most frequent throughput values
print('\n',50*"*")
most_frequent_throughput = data['Avg_Throughput'].value_counts().head(10)
print("Most frequent throughput values:")
print(most_frequent_throughput)
print('\n',50*"*")


# In[57]:


# Plot histogram for top 10 TCP values
plt.figure(figsize=(10, 5))
sns.histplot(data=top_10_tcp, bins=10,kde=True)
plt.title("Histogram of Top 10 TCP Values")
plt.xlabel("TCP Values")
plt.ylabel("Frequency")
plt.show()

# Plot histogram for bottom 10 TCP values
plt.figure(figsize=(10, 5))
sns.histplot(data=bottom_10_tcp, bins=10,kde=True)
plt.title("Histogram of Bottom 10 TCP Values")
plt.xlabel("TCP Values")
plt.ylabel("Frequency")
plt.show()

# Plot histogram for most frequent TCP values
plt.figure(figsize=(10, 5))
sns.histplot(data=most_frequent_tcp, bins=10,kde=True)
plt.title("Histogram of Most Frequent TCP Values")
plt.xlabel("TCP Values")
plt.ylabel("Frequency")
plt.show()


# # Task 3.3 - Compute & report:
# 
# #### d. The distribution of the average throughput per handset type and provide interpretation for your findings.
# 
# #### e. The average TCP retransmission view per handset type and provide interpretation for your findings.

# In[58]:


data.columns


# In[59]:


data['Handset_Type'].value_counts()


# In[60]:



# Compute average throughput per handset type
avg_throughput_per_type = data.groupby('Handset_Type')['Avg_Throughput'].mean()

# Compute average TCP retransmission per handset type
avg_tcp_retransmission_per_type = data.groupby('Handset_Type')['Avg_TCP_Retransmission'].mean()

# Create a bar plot of average throughput per handset type
plt.figure(figsize=(12, 6))
plt.bar(avg_throughput_per_type.index, avg_throughput_per_type)
plt.xlabel('Handset Type')
plt.ylabel('Average Throughput')
plt.title('Average Throughput per Handset Type')
plt.xticks(rotation=90)
plt.show()

# Create a bar plot of average TCP retransmission per handset type
plt.figure(figsize=(12, 6))
plt.bar(avg_tcp_retransmission_per_type.index, avg_tcp_retransmission_per_type)
plt.xlabel('Handset Type')
plt.ylabel('Average TCP Retransmission')
plt.title('Average TCP Retransmission per Handset Type')
plt.xticks(rotation=90)
plt.show()


# In[61]:


top_n = 10
top_handset_types = aggregated_data['Handset_Type'].value_counts().head(top_n).index

# Extract the average throughput for each handset type
avg_throughput_per_type = aggregated_data.groupby('Handset_Type')['Avg_Throughput'].mean()

# Create a bar plot of average throughput per handset type
plt.figure(figsize=(12, 6))
plt.bar(avg_throughput_per_type.index, avg_throughput_per_type)
plt.xlabel('Handset Type')
plt.ylabel('Average Throughput')
plt.title(f'Average Throughput per Handset Type (Top {top_n})')
plt.xticks(rotation=90)
plt.show()


# In[62]:


# Select the top N most common handset types
top_n = 10
top_handset_types = aggregated_data['Handset_Type'].value_counts().head(top_n).index

# Extract the average TCP retransmission for each handset type
avg_tcp_retransmission_per_type = aggregated_data.groupby('Handset_Type')['Avg_TCP_Retransmission'].mean()

# Create a bar plot of average TCP retransmission per handset type
plt.figure(figsize=(12, 6))
plt.bar(avg_tcp_retransmission_per_type.index, avg_tcp_retransmission_per_type)
plt.xlabel('Handset Type')
plt.ylabel('Average TCP Retransmission')
plt.title(f'Average TCP Retransmission per Handset Type (Top {top_n})')
plt.xticks(rotation=90)
plt.show()


# In[63]:


# Select the top 10 most common handset types
top_n = 10
top_handset_types = aggregated_data['Handset_Type'].value_counts().head(top_n).index

# Filter the data for the top handset types
top_handset_data = aggregated_data[aggregated_data['Handset_Type'].isin(top_handset_types)]

# Create a bar plot of average TCP retransmission per handset type
plt.figure(figsize=(12, 6))
plt.bar(top_handset_data['Handset_Type'], top_handset_data['Avg_TCP_Retransmission'])
plt.xlabel('Handset Type')
plt.ylabel('Average TCP Retransmission')
plt.title(f'Average TCP Retransmission per Top {top_n} Handset Types')
plt.xticks(rotation=90)
plt.show()


# In[64]:


# Select the bottom 10 least common handset types
bottom_n = 10
bottom_handset_types = aggregated_data['Handset_Type'].value_counts().tail(bottom_n).index

# Filter the data for the bottom handset types
bottom_handset_data = aggregated_data[aggregated_data['Handset_Type'].isin(bottom_handset_types)]

# Create a bar plot of average TCP retransmission per handset type
plt.figure(figsize=(12, 6))
plt.bar(bottom_handset_data['Handset_Type'], bottom_handset_data['Avg_TCP_Retransmission'])
plt.xlabel('Handset Type')
plt.ylabel('Average TCP Retransmission')
plt.title(f'Average TCP Retransmission per Bottom {bottom_n} Handset Types')
plt.xticks(rotation=90)
plt.show()


# In[ ]:





# # Task 3.4 - Using the experience metrics above, perform a k-means clustering (where k = 3) to segment users into groups of experiences and provide a brief description of each cluster. (The description must define each group based on your understanding of the data)
# 

# In[77]:


experience_data = aggregated_data[['Avg_TCP_Retransmission', 'Avg_RTT', 'Avg_Throughput']]


# In[78]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaled_data = scaler.fit_transform(experience_data)


# In[79]:


from sklearn.cluster import KMeans

k = 3
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(scaled_data)


# In[80]:


cluster_labels = kmeans.labels_


# In[81]:


cluster_means = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=experience_data.columns)


# In[82]:


# Analyze the clusters
cluster_means = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=experience_data.columns)

# Print the cluster means
print("Cluster Means:")
print(cluster_means)

# Add cluster labels to the aggregated_data DataFrame
aggregated_data['Cluster'] = cluster_labels

# Display the updated aggregated_data DataFrame
print("Updated aggregated_data:")
print(aggregated_data)


# In[83]:


# Description of each cluster
for cluster in range(k):
    print("\nCluster", cluster)
    print("Number of users:", len(aggregated_data[aggregated_data['Cluster'] == cluster]))
    print("Cluster Mean:")
    print(cluster_means.iloc[cluster])


# Cluster 0:
# 
# Number of users: 43,087
# This cluster has relatively lower average TCP retransmission (7.67 million), moderate average RTT (55.82 ms), and higher average throughput (7,079 kbps).
# Cluster 1:
# 
# Number of users: 20,804
# This cluster has higher average TCP retransmission (19.96 million), higher average RTT (109.11 ms), and lower average throughput (3,487 kbps).
# Cluster 2:
# 
# Number of users: 42,966
# This cluster has higher average TCP retransmission (19.89 million), lower average RTT (40.87 ms), and lower average throughput (3,548 kbps).
# These descriptions provide an overview of the characteristics of each cluster based on the average values of the experience metrics

# In[84]:


# Analyze the clusters
cluster_means = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=experience_data.columns)
# Print the cluster means
for i, cluster_mean in enumerate(cluster_means.iterrows()):
    print(f"Cluster {i}:")
    print("Number of users:", aggregated_data['Cluster'].value_counts()[i])
    print("Cluster Mean:")
    print(cluster_mean[1])
    print()


# In[85]:


# Cluster descriptions
cluster_descriptions = {
    0: "Cluster 0 represents users with a relatively stable and high-performing network experience. They have lower average TCP retransmission (7.67 million), moderate average RTT (55.82 ms), and higher average throughput (7,079 kbps).",
    1: "Cluster 1 represents users with higher network latency and lower throughput. They have higher average TCP retransmission (19.96 million), higher average RTT (109.11 ms), and lower average throughput (3,487 kbps).",
    2: "Cluster 2 represents users with higher TCP retransmission and a mix of network performance. They have higher average TCP retransmission (19.89 million), lower average RTT (40.87 ms), and lower average throughput (3,548 kbps)."
}

# Print cluster descriptions
for i in range(k):
    print("Cluster", i, "Description:")
    print(cluster_descriptions[i])
    print()


# In[88]:


# Create scatter plots for each experience metric
cluster_sizes = aggregated_data['Cluster'].value_counts().sort_index()
for metric in experience_data.columns:
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=aggregated_data, x='Cluster', y=metric)
    plt.xlabel('Cluster')
    plt.ylabel(metric)
    plt.title(f'{metric} by Cluster')
    plt.show()

# Explore cluster characteristics
num_clusters = 3  # Replace with the actual number of clusters

for i, cluster_mean in enumerate(cluster_means.iterrows()):
    cluster_label = i
    num_users = cluster_sizes[i]

    # Explore cluster characteristics
    cluster_data = aggregated_data[aggregated_data['Cluster'] == i]
    cluster_description = f"Cluster {i}:\nNumber of users: {num_users}\nCluster Mean:\n{cluster_mean[1]}"

    # Perform additional analysis or calculations based on the cluster characteristics

    # Provide actionable insights or recommendations based on the analysis
    print(cluster_description)
    print("Actionable Insights:")
    print("Based on the analysis, it is recommended to...")
    print()

# Conduct statistical tests
import scipy.stats as stats


for metric in experience_data.columns:
    cluster_data = [aggregated_data[aggregated_data['Cluster'] == i][metric] for i in range(num_clusters)]
    f_value, p_value = stats.f_oneway(*cluster_data)
    print(f"ANOVA test for {metric}:")
    print("F-value:", f_value)
    print("p-value:", p_value)
    print()


# In[89]:



# Create a scatter plot for each pair of experience metrics
plt.figure(figsize=(12, 8))
for i, metric1 in enumerate(experience_data.columns):
    for j, metric2 in enumerate(experience_data.columns):
        plt.subplot(3, 3, i * 3 + j + 1)
        for cluster_label in range(3):
            cluster_data = experience_data[aggregated_data['Cluster'] == cluster_label]
            plt.scatter(cluster_data[metric1], cluster_data[metric2], label=f'Cluster {cluster_label}', alpha=0.5)
        plt.xlabel(metric1)
        plt.ylabel(metric2)
plt.legend()
plt.tight_layout()
plt.show()


# In[90]:


# Explore cluster characteristics using box plots
plt.figure(figsize=(12, 6))
for i, metric in enumerate(experience_data.columns):
    plt.subplot(1, 3, i + 1)
    sns.boxplot(x='Cluster', y=metric, data=aggregated_data)
    plt.xlabel('Cluster')
    plt.ylabel(metric)
plt.tight_layout()
plt.show()


# In[ ]:





# In[ ]:




