import os
import pandas as pd
from bigan import GAN
import random
import tensorflow as tf
import xgboost as xgb
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix
from plot_confusion_matrix import plot_confusion_matrix

os.environ["CUDA_VISIBLE_DEVICES"]=""

class XGB:

    def __init__(self, num_historical_days, days, pct_change, test_size, generator_input_size,  gan_model_path):
        self.data = []
        self.labels = []
        self.test_data = []
        self.test_labels = []
      
        gan = GAN(num_features=5, num_historical_days=num_historical_days, generator_input_size=generator_input_size)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            # saver.restore(sess, gan_model_path)

            with open('../models/checkpoint', 'rb') as f:
                model_name = next(f).split('"')[1]
            saver.restore(sess, "../models/{}".format(model_name))
            files = [os.path.join('../stock_data', f) for f in os.listdir('../stock_data')]
            for file in files:
                print(file)
                #Read in file -- note that parse_dates will be need later
                df = pd.read_csv(file, index_col='Date', parse_dates=True)
                df = df[['Open','High','Low','Close','Volume']]
                #Normilize using a of size num_historical_days
                labels = df.Close.pct_change(days).map(lambda x: int(x > pct_change/100.0))

                df = ((df -
                df.rolling(num_historical_days).mean().shift(-num_historical_days))
                /(df.rolling(num_historical_days).max().shift(-num_historical_days)
                -df.rolling(num_historical_days).min().shift(-num_historical_days)))
                
                df['labels'] = labels
                #Drop the last 10 day that we don't have data for
                df = df.dropna()
                #Hold out the last year of trading for testing
                test_df = df[:test_size]
                #Padding to keep labels from bleeding
                df = df[test_size+num_historical_days:]
               
                data = df[['Open', 'High', 'Low', 'Close', 'Volume']].values
                labels = df['labels'].values
                for i in range(num_historical_days, len(df), num_historical_days):
                    features = sess.run(gan.encoding, feed_dict={gan.X:[data[i-num_historical_days:i]], gan.keep_prob:1.0})
                    self.data.append(features[0])
                    self.labels.append(labels[i-num_historical_days])
                data = test_df[['Open', 'High', 'Low', 'Close', 'Volume']].values
                labels = test_df['labels'].values
                for i in range(num_historical_days, len(test_df), 1):
                    features = sess.run(gan.encoding, feed_dict={gan.X:[data[i-num_historical_days:i]], gan.keep_prob:1.0})
                    self.test_data.append(features[0])
                    self.test_labels.append(labels[i-num_historical_days])



    def train(self, params, save_path, max_steps=100000, early_stopping_rounds=100):
        print('Training')
        print('Train Size = {}'.format(len(self.data)))
        print('Features size = {}'.format(len(self.data[0])))


        train = xgb.DMatrix(self.data, self.labels)
        test = xgb.DMatrix(self.test_data, self.test_labels)

        watchlist = [(test, 'test')]
        clf = xgb.train(params, train, max_steps, evals=watchlist, early_stopping_rounds=early_stopping_rounds)
        joblib.dump(clf, save_path)
        cm = confusion_matrix(self.test_labels, map(lambda x: int(x > 0.5), clf.predict(test)))
        print(cm)
        plot_confusion_matrix(cm, ['Down', 'Up'], normalize=False, title="Confusion Matrix")


