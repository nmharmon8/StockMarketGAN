import os
import pandas as pd
from gan import GAN
import random
import tensorflow as tf
import xgboost as xgb
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix


class TrainXGBBoost:

    def __init__(self, num_historical_days, days=10, pct_change=5):
        self.data = []
        self.labels = []
        self.test_data = []
        self.test_labels = []
        assert os.path.exists('./models/checkpoint')
        gan = GAN(num_features=5, num_historical_days=num_historical_days,
                        generator_input_size=200)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            with open('./models/checkpoint', 'rb') as f:
                model_name = next(f).split('"')[1]
            saver.restore(sess, "./models/{}".format(model_name))
            files = [os.path.join('./stock_data', f) for f in os.listdir('./stock_data')][:1]
            for file in files:
                print(file)
                #Read in file -- note that parse_dates will be need later
                df = pd.read_csv(file, index_col='Date', parse_dates=True)
                df = df[['Open','High','Low','Close','Volume']]
                # #Create new index with missing days
                # idx = pd.date_range(df.index[-1], df.index[0])
                # #Reindex and fill the missing day with the value from the day before
                # df = df.reindex(idx, method='bfill').sort_index(ascending=False)
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
                test_df = df[:365]
                #Padding to keep labels from bleeding
                df = df[400:]
                #This may not create good samples if num_historical_days is a
                #mutliple of 7
                for i in range(num_historical_days, len(df), num_historical_days):
                    features = sess.run(gan.features, feed_dict={gan.X:[df[['Open', 'High', 'Low', 'Close', 'Volume']].values[i-num_historical_days:i]]})
                    self.data.append(features[0])
                    self.labels.append(df['labels'].values[i-1])
                for i in range(num_historical_days, len(test_df), 1):
                    features = sess.run(gan.features, feed_dict={gan.X:[test_df[['Open', 'High', 'Low', 'Close', 'Volume']].values[i-num_historical_days:i]]})
                    self.test_data.append(features[0])
                    self.test_labels.append(test_df['labels'].values[i-1])



    def train(self):
        params = {}
        params['objective'] = 'multi:softprob'
        params['eta'] = 0.001
        params['num_class'] = 2
        params['max_depth'] = 9
        params['subsample'] = 0.01
        params['colsample_bytree'] = 0.5
        params['eval_metric'] = 'mlogloss'
        params['scale_pos_weight'] = 10
        params['silent'] = True
        params['gpu_id'] = 0
        params['max_bin'] = 16
        params['tree_method'] = 'gpu_hist'

        train = xgb.DMatrix(self.data, self.labels)
        test = xgb.DMatrix(self.test_data, self.test_labels)

        watchlist = [(train, 'train'), (test, 'test')]
        clf = xgb.train(params, train, 4, evals=watchlist)
        joblib.dump(clf, 'models/clf.pkl')
        cm = confusion_matrix(self.test_labels, map(lambda x: int(x[1] > .5), clf.predict(test)))
        print(cm)










boost_model = TrainXGBBoost(num_historical_days=20, pct_change=5)
boost_model.train()