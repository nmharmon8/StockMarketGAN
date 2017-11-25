import os
import pandas as pd
import random
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix
from plot_confusion_matrix import plot_confusion_matrix
from sklearn import metrics

os.environ["CUDA_VISIBLE_DEVICES"]=""

random.seed(42)

class RandomForest:

    def __init__(self, num_historical_days, days, pct_change, test_size, model, sess):
        p_data = []
        n_data = []
        self.data = []
        self.labels = []
        self.test_data = []
        self.test_labels = []
      
       
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
            local_labels = []
            batch = []
            for i in range(num_historical_days, len(df), num_historical_days):
                local_labels.append(labels[i-num_historical_days])
                batch.append(data[i- num_historical_days:i])

            features = sess.run(model.encoding, feed_dict={model.X:batch, model.keep_prob:1.0})
            for feature, label in zip(features, local_labels):
                if label == 1:
                    p_data.append(feature)
                else:
                    n_data.append(feature)

            data = test_df[['Open', 'High', 'Low', 'Close', 'Volume']].values
            labels = test_df['labels'].values
            batch = []
            for i in range(num_historical_days, len(test_df), 1):
                batch.append(data[i-num_historical_days:i])
                self.test_labels.append(labels[i-num_historical_days])
            features = sess.run(model.encoding, feed_dict={model.X:batch, model.keep_prob:1.0})
            self.test_data.extend(features)
                

        self.data.extend(random.sample(n_data, len(p_data)))
        self.data.extend(p_data)
        self.labels.extend([0]*len(p_data))
        self.labels.extend([1]*len(p_data))



    def train(self, save_path, n_estimators=10, max_depth=None, max_features='auto'):
        print('Training')
        print('Train Size = {}'.format(len(self.data)))
        print('Features size = {}'.format(len(self.data[0])))

        clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, max_features=max_features, n_jobs=-1, random_state=42)
        clf.fit(self.data, self.labels)
        joblib.dump(clf, save_path)
        preds = clf.predict_proba(self.test_data)
        fpr, tpr, thresholds = metrics.roc_curve(map(lambda x: x+1, self.test_labels), map(lambda x: x[1], preds), pos_label=2)
        print(metrics.auc(fpr, tpr))
        cm = confusion_matrix(self.test_labels, map(lambda x: int(x[1] > 0.5), preds))
        print(cm)
        # plot_confusion_matrix(cm, ['Down', 'Up'], normalize=False, title="Confusion Matrix")


