# ML on Microbiota
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


### Model 2


### Model 3
