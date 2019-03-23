import tensorflow as tf

'''
In this tutorial we will show how to train a recurrent neural network on a challenging task of language modeling. 
The goal of the problem is to fit a probabilistic model which assigns probabilities to sentences. 
It does so by predicting next words in a text given a history of previous words. For this purpose we will use the 
Penn Tree Bank (PTB) dataset, which is a popular benchmark for measuring the quality of these models, whilst being 
small and relatively fast to train
'''
'''
The dataset is already preprocessed and contains overall 10000 different words, including the end-of-sentence marker and
a special symbol (\<unk>) for rare words. In reader.py, we convert each word to a unique integer identifier, in order to
make it easy for the neural network to process the data
'''