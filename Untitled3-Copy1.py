#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls


# In[ ]:


df = pd.read_csv(r"‪C:\Users\sanha\Downloads\extentkiva.csv")
df


# In[6]:


print(df_kiva.shape)

plt.figure(figsize=(16, 5))

cols = df_kiva.columns

uniques = [len(df_kiva[col].unique()) for col in cols]
sns.set(font_scale=1.1)
ax = sns.barplot(cols, uniques, palette='hls', log=True)
ax.set(xlabel='Feature', ylabel='log(unique count)', title='Number of unique per feature')
for p, uniq in zip(ax.patches, uniques):
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 10,
            uniq,
            ha="center") 
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
plt.show()


# In[ ]:


print(df_kiva.describe())


# In[ ]:


df_kiva.head()


# In[ ]:


##exploring the Funded and and Loan Amount
trace0 = go.Histogram(x=np.log(df_kiva['lender_count'] + 1), 
                      name='Log',
                      nbinsx=30,
                      histnorm='probability')

trace1 = go.Histogram(x=df_kiva['lender_count'], 
                      name='Normal', 
                      nbinsx=600)

trace2 = go.Bar(
    x = df_kiva["lender_count"].value_counts()[:200].index.values,
    y = df_kiva["lender_count"].value_counts()[:200].values, 
    marker=dict(
        color=df_kiva["lender_count"].value_counts()[:200].values,
        colorscale = 'Viridis'
    )
)

fig = tls.make_subplots(rows=2, cols=2, specs=[[{'colspan': 2}, None], [{}, {}]],
                          subplot_titles=('Lender Count Filter first 200',
                                          'Lender Count Log Dist',
                                          'Lender Count Normal Dist'))

fig.append_trace(trace2, 1, 1)
fig.append_trace(trace0, 2, 1)
fig.append_trace(trace1, 2, 2)

fig['layout'].update(showlegend=True, title='Lender Count Distribuition', 
                     bargap=0.05, 
                     height=700, width=800)

py.iplot(fig, filename='custom-sized-subplot-with-subplot-titles')
print("Description of distribuition")
print(df_kiva[['funded_amount','loan_amount']].describe())

plt.figure(figsize=(12,10))

plt.subplot(221)
g = sns.distplot(np.log(df_kiva['funded_amount'] + 1))
g.set_title("Funded Amount Distribuition", fontsize=15)
g.set_xlabel("")
g.set_ylabel("Frequency", fontsize=12)

plt.subplot(222)
g1 = plt.scatter(range(df_kiva.shape[0]), np.sort(df_kiva.funded_amount.values))
g1= plt.title("Funded Amount Residual Distribuition", fontsize=15)
g1 = plt.xlabel("")
g1 = plt.ylabel("Amount(US)", fontsize=12)

plt.subplot(223)
g2 = sns.distplot(np.log(df_kiva['loan_amount'] + 1))
g2.set_title("Loan Amount Distribuition", fontsize=15)
g2.set_xlabel("")
g2.set_ylabel("Frequency", fontsize=12)

plt.subplot(224)
g3 = plt.scatter(range(df_kiva.shape[0]), np.sort(df_kiva.loan_amount.values))
g3= plt.title("Loan Amount Residual Distribuition", fontsize=15)
g3 = plt.xlabel("")
g3 = plt.ylabel("Amount(US)", fontsize=12)

plt.subplots_adjust(wspace = 0.3, hspace = 0.3,
                    top = 0.9)

plt.show()


# In[ ]:


#Term in Months
months = df_kiva.term_in_months.value_counts()

plt.figure(figsize=(25,8))

plt.subplot(122)
g = sns.distplot(np.log(df_kiva['term_in_months'] + 1))

g.set_title("Term in Months Log", fontsize=25)
g.set_xlabel("")
g.set_ylabel("Count", fontsize=18)

plt.subplot(121)
g1 = sns.distplot(df_kiva['term_in_months'])
g1.set_title("Term in Months without filter", fontsize=25)
g1.set_xlabel("")
g1.set_ylabel("Frequency", fontsize=18)

trace0 = go.Bar(
            x = df_kiva["term_in_months"].value_counts()[:60].index.values,
            y = df_kiva["term_in_months"].value_counts()[:60].values,
            marker=dict(
                color=df_kiva["term_in_months"].value_counts()[:60].values,
                colorscale = 'Viridis',
                reversescale = True
        )
    )

data = [trace0]

layout = go.Layout(
    
)

layout = go.Layout(
    yaxis=dict(
        title='Count'
    ),
    xaxis=dict(
        title='Lender Counts Distribuition'
    ),
    title='The Month terms distribuition'
)

fig = go.Figure(data=data, layout=layout)


plt.subplots_adjust(wspace = 0.2, hspace = 0.3,top = 0.9)

plt.show()
py.iplot(fig, filename='term-months')


# In[ ]:


##values through the sectors.
df_kiva['loan_amount_log'] = np.log(df_kiva['loan_amount'])
df_kiva['funded_amount_log'] = np.log(df_kiva['funded_amount'] + 1)
df_kiva['diff_fund'] = df_kiva['loan_amount'] / df_kiva['funded_amount'] 

plt.figure(figsize=(12,14))

plt.subplot(312)
g1 = sns.boxplot(x='sector', y='loan_amount_log',data=df_kiva)
g1.set_xticklabels(g1.get_xticklabels(),rotation=45)
g1.set_title("Loan Distribuition by Sectors", fontsize=15)
g1.set_xlabel("")
g1.set_ylabel("Loan Amount(log)", fontsize=12)

plt.subplot(311)
g2 = sns.boxplot(x='sector', y='funded_amount_log',data=df_kiva)
g2.set_xticklabels(g2.get_xticklabels(),rotation=45)
g2.set_title("Funded Amount(log) by Sectors", fontsize=15)
g2.set_xlabel("")
g2.set_ylabel("Funded Amount", fontsize=12)

plt.subplot(313)
g3 = sns.boxplot(x='sector', y='term_in_months',data=df_kiva)
g3.set_xticklabels(g3.get_xticklabels(),rotation=45)
g3.set_title("Term Frequency by Sectors", fontsize=15)
g3.set_xlabel("")
g3.set_ylabel("Term Months", fontsize=12)

plt.subplots_adjust(wspace = 0.2, hspace = 0.6,top = 0.9)
plt.show()


# In[ ]:


##the activies by the top 3 sectors
plt.figure(figsize=(12,14))

plt.subplot(311)
g1 = sns.countplot(x='activity', data=df_kiva[df_kiva['sector'] == 'Agriculture'])
g1.set_xticklabels(g1.get_xticklabels(),rotation=45)
g1.set_title("Activities by Agriculture Sector", fontsize=15)
g1.set_xlabel("")
g1.set_ylabel("Count", fontsize=12)

plt.subplot(312)
g2 = sns.countplot(x='activity', data=df_kiva[df_kiva['sector'] == 'Food'])
g2.set_xticklabels(g2.get_xticklabels(),rotation=80)
g2.set_title("Activities by Food Sector", fontsize=15)
g2.set_xlabel("")
g2.set_ylabel("Count", fontsize=12)

plt.subplot(313)
g3 = sns.countplot(x='activity', data=df_kiva[df_kiva['sector'] == 'Retail'])
g3.set_xticklabels(g3.get_xticklabels(),rotation=90)
g3.set_title("Activiies by Retail Sector", fontsize=15)
g3.set_xlabel("")
g3.set_ylabel("Count", fontsize=12)

plt.subplots_adjust(wspace = 0.2, hspace = 0.8,top = 0.9)
plt.show()


# In[ ]:


##the most frequent countrys
country_counts = df_kiva['country'].value_counts().head(40)

countrys_loan = go.Bar(
    x=country_counts.values[::-1],
    y=country_counts.index[::-1],
    orientation = 'h',
    marker=dict(
        color=country_counts.values[::-1],
        colorscale = 'Viridis'
    ),
)

layout = go.Layout(
    title="Frequency Loan Distribuition by Country",
    width=700,
    height=900,
    )

data = [countrys_loan]

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename="Country-counts")


# In[ ]:


##counting of countrys
data = go.Bar(
    x = df_kiva_loc.country.value_counts().values[50::-1],
    y = df_kiva_loc.country.value_counts().index[50::-1],
    orientation = 'h',
    marker=dict(
        color=df_kiva_loc.country.value_counts().values[50::-1],
        colorscale = 'Viridis'
    ),
)

layout = go.Layout(
    title='TOP 50 Countries Around the world with Loans',
    width=800,
    height=1000,
    )
figure = go.Figure(data=[data], layout=layout)
py.iplot(figure, filename="Loans-country-count")


# In[ ]:




