# Veneto
Project on clustering of Veneto dialects.

The veneto_data.csv file has the following format:

    Area    Language    Location    word1   word2   word3   word4
    North   B           Bl          vaNZar  aveZ    seNZa   akOrDarse


In the first column, we report the area where the dialect is spoken ('North' stands for the Northern part of the region). In the second column, we report the dialect spoken ('B' stands for Bellunese). In the third column, we report its location ('Bl' stands for Belluno). The following colums report the presence of a certain word ('word1' has all the cognates for Italian 'avanzare'). There are about 600 columns in the dataset.


The script ```veneto.py``` takes this file and performs classifications. We split the data into a 70/30 train/test split and perform Logistic Regression. We also sample only 10% of our features at a time (equivalent to roughly 60 potential words, if reported).

We expected that a list of words would be a perfect predictor of the area of the speaker, but we found out that accuracy stops at 94%. In fact, there are cases in which some languages are not well sampled (especially in the areas of Grado and Trieste), but also interesting cases in which parallel sound changes created similar dialects in different regions (for instance, the case of Polesano). Using the whole feature table will not allow us to reach more than 96% in terms of accuracy.

Check the two files ```Veneto-LinguisticAnalysis.pdf``` and ```Veneto-MachineLearning.pdf``` for more information.
