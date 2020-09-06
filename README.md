# book-recommender-project
*Uses spaCy and sklearn to predict whether or not I will like a book based on the first 50 pages of text.*

In this project, I wanted to create a better way to decide whether to read a book. I collected the first 50 pages of text from 30 books that I finished reading and 10 books that I started reading but didn’t finish as my data. All books were read in 2020.

I split each book into five 10-page segments so the model would have more data to train on, and I preprocessed the text with spaCy and sklearn by tokenizing and lemmatizing, removing pronouns, punctuation, and stop words, and then vectorizing the text.

I put all the preprocessing steps into a pipeline, and spot-checked 6 different classification models known to perform well on text classification.

KNeighborsClassifier had the best performance at 82% accuracy and 85% precision; precision was my target metric. Finally, I tuned hyperparameters and tested the model on out-of-sample books.

I wrote an article about the process and findings for this project; you can find it [here](https://iamryanbrown.com/book-recommender). 
