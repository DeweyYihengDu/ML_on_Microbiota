# Investigating the impact of antibiotics on environmental microbiota through machine learning models

This project uses machine learning to explore the impact of different antibiotics on soil microbial communities.
## Data Description

The data that support the findings of this study are openly available under NCBI BioProject ID PRJNA576637 and are available at [PRJNA576637](https://www.ncbi.nlm.nih.gov/bioproject/PRJNA576637).

The data from PRJNA576637 includes fasta sequence files of soil microorganisms treated with different antibiotics, along with corresponding metadata information. OTU clustering of fasta sequence files was performed using Vsearch to generate the OTU table (in matrix folder).

## Data Preprocessing

+ We only consider data at the Order taxonomic level.
+ Ignore missing data.
+ In Model 3, replace the abundance of bacteria with the mean value across each plot to reduce noise.

## Machine Learning Models

### Model 1

Model 1 uses the abundance of all bacteria except the top 5 to predict the abundance of these 5 bacteria, and compares the performance of different models.

The following code can be used to get the performance of different machine learning algorithms in Model 1.

```bash
python code/Model1/model1.py
```


### Model 2

Model 2 uses the abundance of bacteria and the source of the sample to predict the situation of antibiotic pollution.

The following code can be used to get the number of feature values used by the optimal model.

```
python code/Model2/model2.py
```

### Model 3

Model 3 is based on short-term incubation data to predict long-term incubation data. Our machine learning algorithm is based on random forests, as random forests performed best in Model 1. We used two different methods to handle data, one is to use Incubation 3 days and Incubation 10 days data separately, and the other is to use all incubation data. We found that using all incubation data performed better.

The following code can be used to get the performance results of the model using different incubation data.

```
python code/Model3/model3.py
```