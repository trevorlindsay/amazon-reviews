import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer
import sys


# Text file is encoded in latin-1
reload(sys)
sys.setdefaultencoding('latin-1')


class AmazonReviews(object):

    def __init__(self, max_words=2500, debug=False, is_regression=True):

        # Debug mode is for when changes are made to the model and need to run a quick test
        self.debug = debug

        print 'Reading in Amazon reviews from text file.'
        self.df = self.structureText()

        print 'Creating word count matrix.'
        word_counts = self.wordCounts(self.df.text, max_words)

        print 'Combining word count matrix with other features from reviews.'
        other_features = ['score', 'num_words', 'num_chars', 'readability']
        self.X = pd.merge(self.df[other_features], word_counts, left_index=True, right_index=True)

        # If there is a column named fit, rename it so no problems arise later on
        # E.g. hasattr(self.X, 'fit') would return True even though self.X is not an estimator
        try:
            self.X.rename(columns={'fit': 'fit_'}, inplace=True)
        except:
            pass

        # Determine what to use as target variable
        # If 'is_regression' is True, use actual ratio, else you binary classes
        self._regression = is_regression
        self.Y = self.df['helpfulnessRatio'] if self.regression else self.df['helpfulnessClass']

    def structureText(self):
        """
        Function to clean up raw text and store in a pandas dataframe
        """
        f = open('data/foods.txt', 'rb')
        raw_text = [line.strip() for line in f.readlines()]

        # Dictionary to track reviews
        # Keys are fields (e.g. 'text', 'rating', 'productID'), values are lists
        reviews = defaultdict(list)

        for i, line in enumerate(raw_text):

            # If line is blank, skip it
            if len(line) == 0:
                continue

            # Split the line between the field name and the value
            divider = line.find(': ')

            # If no '/' character in the line, skip it
            try:
                (field1, field2), value = line[:divider].split('/'), line[divider + 2:]
            except ValueError:
                continue

            # If there is a '/' character, but it doesn't split the field name, skip it
            if field1 not in ('review', 'product'):
                continue

            reviews[field2].append(value.encode('utf-8'))

            # If in debug mode, only read in the first 10,000 reviews
            if self.debug:
                if sum([False if len(x) == 10000 else True for x in reviews.values()]) == 0:
                    break

        f.close()

        # Store reviews in a pandas dataframe
        df = pd.DataFrame(reviews)

        # Convert helpfulness to a float, remove those where ratio is undefineds

        df['helpfulnessRatio'] = df.helpfulness.apply(self.helpfulnessRatio)
        df = df[~pd.isnull(df.helpfulnessRatio)]

        # Covert helpfulness ratio to a binary class variable
        df['helpfulnessClass'] = df.helpfulnessRatio.map(lambda x: 0 if x < 0.5 else 1)

        # Convert time to datetime
        df['time'] = pd.to_datetime(df.time.apply(int), unit='s')

        # Convert score to a float
        df['score'] = df.score.apply(float)

        # Create new features based on review text
        df['num_words'] = df.text.map(lambda x: len(x.split()))
        df['num_chars'] = df.text.map(lambda x: sum([len(word) for word in x.split()]))
        df['readability'] = df.text.apply(self.automatedReadabilityIndex)

        return df

    @staticmethod
    def helpfulnessRatio(x):
        """
        Helper function to convert helpfulness ratio to a float (from a string)
        """
        numerator, denominator = x.split('/')
        if numerator > denominator:
            return np.nan
        return np.nan if denominator == '0' else float(numerator) / float(denominator)

    @staticmethod
    def automatedReadabilityIndex(x):
        """
        Helper function to compute Automated Readability Index
        https://en.wikipedia.org/wiki/Automated_readability_index
        """
        words = len(x)
        chars = sum([len(word) for word in x.split()])
        sentences = sum([x.count('. '), x.count('! '), x.count('? '), 1])
        return 4.71 * (chars / words) + 0.5 * (words / sentences) - 21.43

    @staticmethod
    def wordCounts(text, max_words):
        """
        Creates a matrix of word counts from text of reviews
        :param text: pandas series, text of reviews
        :param max_words: int, maximum number of words to keep as features
        :return: pandas dataframe, matrix of 2,500 most prevalent words and how many times they appear in each review
        """
        vectorizer = CountVectorizer(max_features=max_words, stop_words='english')
        X = vectorizer.fit_transform(text.values).toarray()
        words = vectorizer.get_feature_names()
        return pd.DataFrame(X, index=text.index, columns=words)

    @property
    def regression(self):
        return self._regression

    @regression.setter
    def regression(self, is_regression):
        self._regression = is_regression
        self.Y = self.df.helpfulnessRatio.values if self.regression else self.df.helpfulnessClass.values

