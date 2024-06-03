# Clustering-Antarctic-Penguin-Species


## Table of Contents

- [Project Overview](#project-overview)
- [Data Sources](#data-sources)
- [Tools](#tools)
- [Data Cleaning/Preparation](#data-cleaningpreparation)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Data Analysis](#data-analysis)
- [Results/Findings](#resultsfindings)
- [Recommendations](#recommendations)
- [Limitations](#limitations)
- [References](#references)

### Project Overview
---

This data analysis project aims to support a team of researchers who have been collecting data about penguins in Antartica. They have not been able to record the species of penguin, but they know that there are three species that are native to the region: **Adelie**, **Chinstrap**, and **Gentoo**, so the task is to apply data science skills to help them identify groups in the dataset.

![Alt text](https://imgur.com/orZWHly.png)
source: @allison_horst https://github.com/allisonhorst/penguins


### Data Sources

Penguins Data: The primary dataset used for this analysis is the "penguins.csv" file, containing detailed information about each observed penguin to be categorized.

**Origin of this data** : Data were collected and made available by Dr. Kristen Gorman and the Palmer Station, Antarctica LTER, a member of the Long Term Ecological Research Network.

**The dataset consists of 5 columns.**

- culmen_length_mm: culmen length (mm)
- culmen_depth_mm: culmen depth (mm)
- flipper_length_mm: flipper length (mm)
- body_mass_g: body mass (g)
- sex: penguin sex

### Tools

- Python - Data Cleaning, Visualizations, Machine Learning Modeling
- Jupyter Notebook - Data Analysis


### Data Cleaning/Preparation

In the initial data preparation phase, we performed the following tasks:
1. Data loading and inspection.
2. Handling missing values and outliers  .
4. Data cleaning and formatting.

### Exploratory Data Analysis

EDA involved exploring the penguins data to answer key questions, such as:

- What is the overall distribution of penguin data?
- Are there any outliers that might affect the data?
- Are the scales of the data comparable and usable for machine learning models?



### Data Analysis

The data approximates a roughly normal distribution for its numerical columns, but there are two significant outliers for 'flipper_lenght_mm'. In addition, the scale of the 'body_mass_g' is vastly different from the other features and needs to be scaled so it doesn't affect the classification algorithm in the machine learning model disproportionately.

```python
fig,axes=plt.subplots(4,1,figsize=(10,15))
sns.histplot(x="culmen_length_mm", data=penguins_df,ax=axes[0])
sns.histplot(x="culmen_depth_mm", data=penguins_df,ax=axes[1])
sns.histplot(x="flipper_length_mm", data=penguins_df,ax=axes[2])
sns.histplot(x="body_mass_g", data=penguins_df,ax=axes[3])
```
![Untitled](https://github.com/RicardoGrayson/Clustering-Antarctic-Penguin-Species/assets/63846918/56d256c8-d819-44b0-b8a3-d487a0eefba6)


```python
sns.boxplot(penguins_df)
plt.xticks(rotation=45)
```
![image](https://github.com/RicardoGrayson/Clustering-Antarctic-Penguin-Species/assets/63846918/0cbf21a1-a4bd-41ef-9a44-9fc68bc3652b)

Since we are unsure of how many species of penguin are included in the data, we will be running an unsupervised machine learning model using K-means clustering to classify the penguins. To do so, the 'sex' feature must first be converted into dummy variables, and the features must be scaled to have a uniform scale.

```python
penguins_preprocessed=pd.get_dummies(penguins_clean,drop_first=True)

print(penguins_preprocessed.describe())
print(penguins_preprocessed.info())
```

```python
scaler=StandardScaler()
penguins_preprocessed=scaler.fit_transform(penguins_preprocessed)
```
Next, we must perform a Principal Component Analysis (PCA) of the features to identify components with an explained variance greater than 10% to reduce the dimensionality of the dataset and 


### Results/Findings

The analysis results are summarized as follows:
1. The PCA of the scaled features identifies 3 components with an explained variance ratio greater than 10%
2. Performing an elbow analysis of K-means clusters reveals an optimal K cluster value of 4   
3. Performing K-means clustering on principal components of the dataset reveals 4 clusters of penguins, suggesting an additional species of penguin unique from the 3 previously identified by the researchers.

### Recommendations

Based on the analysis, we recommend the following actions:
- Invest in marketing and promotions during peak sales seasons to maximize revenue.
- Focus on expanding and promoting products in Category A.
- Implement a customer segmentation strategy to target high-LTV customers effectively.

### Limitations

I had to remove all zero values from budget and revenue columns because they would have affected the accuracy of my conclusions from the analysis. There are still a few outliers even after the omissions but even then we can still see that there is a positive correlation between both budget and number of votes with revenue.

### References

1. SQL for Businesses by werty.
2. [Stack Overflow](https://stack.com)



|Heading1|Heading2|
|--------|--------|
|Content|Content2|
|Python|SQL|

`column_1`

**bold**

*italic*
