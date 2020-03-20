# Adaboost
using adaboost for real world binary sentiment analysis.

we use adaboost for classifying the polarity of a given text into two classes - positive or negative.
We use movie reviews from IMDB as our data.

- review polarity.tar.gz - a sentiment analysis dataset of movie reviews from IMBD.
Extract its content in the same directory so you will have a folder called review polarity.
- process_data.py - code for loading and preprocessing the data.
- adaboost.py - the actual algorithm

we processes the data and represent every review as a 5000 vector x. 
The values of x are counts of the most common words in the dataset (excluding stopwords like “a” and “and”),
in the review that x represents.
Concretely, let w1, w2, ..., w5000 be the most common words in the data, given a
review ri we represent it as a vector xi ∈ N^5000 where xi,j is the number of times the word wj appears in ri.
The method parse data returns a training data, test data and a vocabulary.
The vocabulary is a dictionary that maps each index in the data to the word it represents (i.e. it maps j -> wj ).

as The class of weak learners we use the class of comparing a single word count to a threshold.
