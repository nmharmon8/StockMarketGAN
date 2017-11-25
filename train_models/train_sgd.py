import os
import pandas as pd
from bigan import GAN
import random
import tensorflow as tf
import xgboost as xgb
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import SGDClassifier
from plot_confusion_matrix import plot_confusion_matrix

os.environ["CUDA_VISIBLE_DEVICES"]=""

class TrainXGBBoost:

    def __init__(self, num_historical_days, days=10, pct_change=10, test_size=504, gan_model_path='./deployed_models/bigan'):
        self.data = []
        self.s_data = []
        self.labels = []
        self.test_data = []
        self.test_labels = []
      
        gan = GAN(num_features=5, num_historical_days=num_historical_days,
                        generator_input_size=40)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            saver.restore(sess, gan_model_path)
            files = [os.path.join('./stock_data', f) for f in os.listdir('./stock_data')]
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
                for i in range(num_historical_days, len(df), 1):
                    features = sess.run(gan.features, feed_dict={gan.X:[data[i-num_historical_days:i]], gan.keep_prob:1.0})
                    self.data.append(features[0])
                    self.labels.append(labels[i-num_historical_days])
                data = test_df[['Open', 'High', 'Low', 'Close', 'Volume']].values
                labels = test_df['labels'].values
                for i in range(num_historical_days, len(test_df), 1):
                    features = sess.run(gan.features, feed_dict={gan.X:[data[i-num_historical_days:i]], gan.keep_prob:1.0})
                    self.test_data.append(features[0])
                    self.test_labels.append(labels[i-num_historical_days])



    def train(self):
        print('Training')
        print('Train Size = {}'.format(len(self.data)))
        print('Features size = {}'.format(len(self.data[0])))
        clf = SGDClassifier(loss='hinge')
        clf.fit(self.data, self.labels)
        joblib.dump(clf, 'models/clf.pkl')
        cm = confusion_matrix(self.test_labels, map(lambda x: int(x > 0.5), clf.predict(self.test_data)))
        print(cm)
        plot_confusion_matrix(cm, ['Down', 'Up'], normalize=False, title="Confusion Matrix")


boost_model = TrainXGBBoost(num_historical_days=20, days=10, pct_change=10)
boost_model.train()
