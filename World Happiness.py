import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
data = pd.read_csv("world-happiness-report-2021.csv")
print(data.head)
data_set = data
print(data_set.rename(columns={'Ladder score': 'Ranking'}, inplace=True))
print(data.isnull().sum())
order = data.sort_values('Ranking')
print(order.head())
df = data
print(df.sort_values('Ranking'))
data1=data[['Country name','Regional indicator','Ranking','Logged GDP per capita',
         'Social support','Healthy life expectancy','Freedom to make life choices','Generosity','Perceptions of corruption',
         'Ladder score in Dystopia']]
print(data1.shape)
print(data1.isnull().sum())
plt.figure(figsize=(15,10))
sns.barplot(data=data1,x='Regional indicator',y='Ranking')
plt.xticks(rotation=45)
plt.show()
print(data1.corr()[['Ranking']].sort_values(by='Ranking',ascending=False))
top_5_happiest=data1.sort_values(by='Ranking',ascending=False).head(5)
top_5_unhappiest=data1.sort_values(by='Ranking',ascending=False).tail(5)
plt.figure(figsize=(15,6))
sns.barplot(data=data1,x=top_5_happiest['Country name'],y=top_5_happiest['Ranking'])
plt.show()
plt.figure(figsize=(15,6))
sns.barplot(data=data1,x=top_5_unhappiest['Country name'],y=top_5_unhappiest['Ranking'])
plt.show()
print(data1.shape)
data = data[:50]
plt.figure(figsize=(20,13))
sns.scatterplot(data= data, x='Ranking',y='Logged GDP per capita',s=100)
plt.title('Life Satisfication vs GDP per Captia',fontsize=20)
plt.xlabel('Life Satisfication',fontsize=16)
plt.ylabel('GDP per Captia',fontsize=16)
for i in range(len(data)):
    plt.text(s =data.loc[i,'Country name'], x=data.loc[i,'Ranking']+0.01,y=data.loc[i,'Logged GDP per capita']+0.01, fontsize=10)
plt.show()
x = np.array(data['Ranking'])
for i in range(len(x)):
    fit = -0.0747281 *x*x*x + 1.05045076*x*x -3.16337971*x +7.10682006
plt.figure(figsize=(20,13))
sns.scatterplot(data= data, x='Ranking',y='Logged GDP per capita',s=100)
plt.title('Life Satisfication vs GDP per Captia',fontsize=20)
plt.xlabel('Life Satisfication',fontsize=16)
plt.ylabel('GDP per Captia',fontsize=16)
plt.plot(x,fit,lw=8,alpha=0.3)
for i in range(len(data)):
    plt.text(s =data.loc[i,'Country name'], x=data.loc[i,'Ranking']+0.01,y=data.loc[i,'Logged GDP per capita']+0.01, fontsize=10)
plt.show()
x = np.array(data['Ranking'])
for i in range(len(x)):
    fit = -0.0747281 *x*x*x + 1.05045076*x*x -3.16337971*x +7.10682006
bound_country = ['Kuwait','Guatemala','Jamaica','Costa Rica','Kosovo','Brazil','El Salvador','Uzbekistan','United Arab Emirates','Singapore','Ireland','Luxembourg']
plt.figure(figsize=(20,13))
sns.scatterplot(data= data, x='Ranking',y='Logged GDP per capita',s=100)
plt.title('Life Satisfication vs GDP per Captia',fontsize=20)
plt.xlabel('Life Satisfication',fontsize=16)
plt.ylabel('GDP per Captia',fontsize=16)
plt.plot(x,fit,lw=8,alpha=0.3)
for i in range(len(data)):
    if data.loc[i,'Country name'] in bound_country:
        plt.text(s =data.loc[i,'Country name'], x=data.loc[i,'Ranking']+0.01,y=data.loc[i,'Logged GDP per capita']+0.01, fontsize=10,color='lightgrey')
        plt.scatter(x=data.loc[i,'Ranking'],y=data.loc[i,'Logged GDP per capita'],color ='lightgrey',s=100)
    else:
        plt.text(s =data.loc[i,'Country name'], x=data.loc[i,'Ranking']+0.01,y=data.loc[i,'Logged GDP per capita']+0.01, fontsize=10)
plt.show()
x = np.array(data['Ranking'])
for i in range(len(x)):
    fit = -0.0747281 *x*x*x + 1.05045076*x*x -3.16337971*x +7.10682006
bound_country = ['Kuwait','Guatemala','Jamaica','Costa Rica','Kosovo','Brazil','El Salvador','Uzbekistan','United Arab Emirates','Singapore','Ireland','Luxembourg']
ax = plt.figure(figsize=(20,13))
ax = plt.axes()
ax.set(facecolor = "whitesmoke")
sns.scatterplot(data= data, x='Ranking',y='Logged GDP per capita',s=100,color = 'gray')
plt.title('Life Satisfication vs GDP per Captia',fontsize=20)
plt.xlabel('Life Satisfication',fontsize=16)
plt.ylabel('GDP per Captia',fontsize=16)
plt.plot(x,fit,lw=8,alpha=0.3)
for i in range(len(data)):
    if data.loc[i,'Country name'] in bound_country:
        plt.text(s =data.loc[i,'Country name'], x=data.loc[i,'Ranking']+0.01,y=data.loc[i,'Logged GDP per capita']+0.01, fontsize=10,color='gainsboro')
        plt.scatter(x=data.loc[i,'Ranking'],y=data.loc[i,'Logged GDP per capita'],color ='gainsboro',s=100)
    else:
        plt.text(s =data.loc[i,'Country name'], x=data.loc[i,'Ranking']+0.01,y=data.loc[i,'Logged GDP per capita']+0.01, fontsize=10,color = 'dimgray')
plt.show()
x = np.array(data['Ranking'])
for i in range(len(x)):
    fit = -0.0747281 *x*x*x + 1.05045076*x*x -3.16337971*x +7.10682006
bound_country = ['Kuwait','Guatemala','Jamaica','Costa Rica','Kosovo','Brazil','El Salvador','Uzbekistan','United Arab Emirates','Singapore','Ireland','Luxembourg']
ax = plt.figure(figsize=(20,13))
ax = plt.axes()
ax.set(facecolor = "white")
sns.scatterplot(data= data, x='Ranking',y='Logged GDP per capita',s=100,color = 'gray')
plt.title('Life Satisfication vs GDP per Captia',fontsize=30,color = 'gray')
plt.ylabel('GDP per Captia',fontsize=16)
plt.plot(x,fit,lw=8,alpha=0.3)
for i in range(len(data)):
    plt.text(s =data.loc[i,'Country name'], x=data.loc[i,'Ranking']+0.01,y=data.loc[i,'Logged GDP per capita']+0.01, fontsize=10)
plt.show()
x = np.array(data['Ranking'])
for i in range(len(x)):
    fit = -0.0747281 *x*x*x + 1.05045076*x*x -3.16337971*x +7.10682006
bound_country = ['Kuwait','Guatemala','Jamaica','Costa Rica','Kosovo','Brazil','El Salvador','Uzbekistan','United Arab Emirates','Singapore','Ireland','Luxembourg']
plt.figure(figsize=(20,13))
sns.scatterplot(data= data, x='Ranking',y='Logged GDP per capita',s=100)
plt.title('Life Satisfication vs GDP per Captia',fontsize=20)
plt.xlabel('Life Satisfication',fontsize=16)
plt.ylabel('GDP per Captia',fontsize=16)
plt.plot(x,fit,lw=8,alpha=0.3)
for i in range(len(data)):
    if data.loc[i,'Country name'] in bound_country:
        plt.text(s =data.loc[i,'Country name'], x=data.loc[i,'Ranking']+0.01,y=data.loc[i,'Logged GDP per capita']+0.01, fontsize=10,color='lightgrey')
        plt.scatter(x=data.loc[i,'Ranking'],y=data.loc[i,'Logged GDP per capita'],color ='lightgrey',s=100)
    else:
        plt.text(s =data.loc[i,'Country name'], x=data.loc[i,'Ranking']+0.01,y=data.loc[i,'Logged GDP per capita']+0.01, fontsize=10)
plt.show()
x = np.array(data['Ranking'])
for i in range(len(x)):
    fit = -0.0747281 *x*x*x + 1.05045076*x*x -3.16337971*x +7.10682006
bound_country = ['Kuwait','Guatemala','Jamaica','Costa Rica','Kosovo','Brazil','El Salvador','Uzbekistan','United Arab Emirates','Singapore','Ireland','Luxembourg']
ax = plt.figure(figsize=(20,13))
ax = plt.axes()
ax.set(facecolor = "whitesmoke")
sns.scatterplot(data= data, x='Ranking',y='Logged GDP per capita',s=100,color = 'dimgray')
plt.title('Life Satisfication vs GDP per Captia',fontsize=20)
plt.xlabel('Life Satisfication',fontsize=16)
plt.ylabel('GDP per Captia',fontsize=16)
plt.plot(x,fit,lw=8,alpha=0.3)
for i in range(len(data)):
    if data.loc[i,'Country name'] in bound_country:
        plt.text(s =data.loc[i,'Country name'], x=data.loc[i,'Ranking']+0.01,y=data.loc[i,'Logged GDP per capita']+0.01, fontsize=10,color='gainsboro')
        plt.scatter(x=data.loc[i,'Ranking'],y=data.loc[i,'Logged GDP per capita'],color ='gainsboro',s=100)
    else:
        plt.text(s =data.loc[i,'Country name'], x=data.loc[i,'Ranking']+0.01,y=data.loc[i,'Logged GDP per capita']+0.01, fontsize=10,color = 'dimgray')
plt.show()
x = np.array(data['Ranking'])
for i in range(len(x)):
    fit = -0.0747281 *x*x*x + 1.05045076*x*x -3.16337971*x +7.10682006
bound_country = ['Kuwait','Guatemala','Jamaica','Costa Rica','Kosovo','Brazil','El Salvador','Uzbekistan','United Arab Emirates','Singapore','Ireland','Luxembourg']
ax = plt.figure(figsize=(18,13))
ax = plt.axes()
ax.set(facecolor = "white")
sns.scatterplot(data= data, x='Ranking',y='Logged GDP per capita',s=100,color = 'gray')
plt.title('Life Satisfication vs GDP per Captia',fontsize=30,color = 'gray')
plt.axis('off')
plt.plot(x,fit,lw=8,alpha=0.3)
for i in range(len(data)):
    if data.loc[i,'Country name'] in bound_country:
        plt.text(s =data.loc[i,'Country name'], x=data.loc[i,'Ranking']+0.01,y=data.loc[i,'Logged GDP per capita']+0.01, fontsize=10,color='gainsboro')
        plt.scatter(x=data.loc[i,'Ranking'],y=data.loc[i,'Logged GDP per capita'],color ='gainsboro',s=100)
    else:
        plt.text(s =data.loc[i,'Country name'], x=data.loc[i,'Ranking']+0.01,y=data.loc[i,'Logged GDP per capita']+0.01, fontsize=10,color = 'dimgray')
plt.text(s='x: Life Satisfication',x=7.3,y=9.4,fontsize=20,color='gray')
plt.text(s='y: GDP per Captia',x=7.3,y=9.3,fontsize=20,color='gray')
plt.show()
df = pd.DataFrame([['Country name', 'Ranking', 'Logged GDP per capita']])
print(df)
df2 = pd.DataFrame([['Healthy life expectancy', 'Freedom to make life choices']])
print(df2)
frames = [df, df2]
print(frames)
result = pd.concat(frames)
print(result)
print(data.columns)

