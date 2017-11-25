import os
import pandas as pd
import random
import tensorflow as tf
import xgboost as xgb
from sklearn.externals import joblib
from sklearn import metrics
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from plot_confusion_matrix import plot_confusion_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
import pickle
import math

random.seed(42)
#plt.style.use('ggplot')


os.environ["CUDA_VISIBLE_DEVICES"]=""

class TestModel:

    def __init__(self, num_historical_days, get_gan_features=None, get_preds=None, name=None, days=10, pct_change=10, test_size=504):
        self.auc_metric = None
        self.X = []
        if not os.path.exists('./test_results'):
            os.makedirs('./test_results')

        predictions_path = '../test_results/predictions_{}_{}_{}_{}_{}.csv'.format(num_historical_days, days, pct_change, test_size, name)
        features_path = '../test_results/gan_features_{}_{}_{}_{}_{}.csv'.format(num_historical_days, days, pct_change, test_size, name)
        files = [os.path.join('../stock_data', f) for f in os.listdir('../stock_data')]
        self.symbols = [f.split('/')[-1] for f in files]
        if not os.path.exists(predictions_path) or not os.path.exists(features_path):
            #Init dataframe 
            test_df = pd.DataFrame(index=pd.read_csv(files[0], index_col='Date', parse_dates=True).index[:test_size - num_historical_days])
            print('Test size {}'.format(len(test_df)))
            for file, symbole in zip(files, self.symbols):
                print(file)
                df = pd.read_csv(file, index_col='Date', parse_dates=True)
                df = df[['Open','High','Low','Close','Volume']]
                #Normilize using a of size num_historical_days
                labels = df.Close.pct_change(days).map(lambda x: int(x > pct_change/100.0))
                returns = df.Close.pct_change(days)

                df = ((df -
                df.rolling(num_historical_days).mean().shift(-num_historical_days))
                /(df.rolling(num_historical_days).max().shift(-num_historical_days)
                -df.rolling(num_historical_days).min().shift(-num_historical_days)))
                
                df['labels'] = labels
                df['returns'] = returns
                df = df[:test_size]   
                results_df = df[['labels', 'returns']][:test_size-num_historical_days]          

                data = df[['Open', 'High', 'Low', 'Close', 'Volume']].values
                test_data = []
                batch = []
                for i in range(num_historical_days, test_size, 1):
                    batch.append(data[i-num_historical_days:i])
                features = get_gan_features(batch)
                test_data.extend(features)

                #First must make df with lables so that the dates match if days are missing
                self.X.extend(test_data)
                results_df['pred'] = map(lambda x: x, get_preds(test_data))
                test_df['labels_{}'.format(symbole)] = results_df['labels']
                test_df['returns_{}'.format(symbole)] = results_df['returns']
                test_df['pred_{}'.format(symbole)] = results_df.pred

            test_df.to_csv(predictions_path)
            pickle.dump(self.X, open(features_path, 'wb'))
    
        self.df = pd.read_csv(predictions_path, index_col='Date', parse_dates=True)
        self.X = pickle.load(open(features_path, 'rb'))
        self.preds = []
        self.labels = []
        self.returns = []
        for symbole in self.symbols:
            #Removing days that this stock was missing
            local_df = self.df[['labels_{}'.format(symbole), 'pred_{}'.format(symbole), 'returns_{}'.format(symbole)]].dropna()
            self.preds.extend(local_df['pred_{}'.format(symbole)])
            self.labels.extend(local_df['labels_{}'.format(symbole)])
            self.returns.extend(local_df['returns_{}'.format(symbole)])

        self.time_ordered_preds = []
        self.time_ordered_lables = []

        for i in range(len(self.df)):
            for symbole in self.symbols:
                if not math.isnan(self.df['pred_{}'.format(symbole)][i]):
                    self.time_ordered_lables.append(self.df['labels_{}'.format(symbole)][i])
                    self.time_ordered_preds.append(self.df['pred_{}'.format(symbole)][i])



    def auc(self, print_m=True):
        if self.auc_metric == None:
            fpr, tpr, thresholds = metrics.roc_curve(map(lambda x: x+1, self.labels), self.preds, pos_label=2)
            self.auc_metric = metrics.auc(fpr, tpr)
        if print_m:
            print(self.auc_metric)
        return self.auc_metric

    #Receiver operating characteristic
    def roc(self):
        # ax = plt.subplot(111)  
        # ax.spines["top"].set_visible(False)  
        # ax.spines["right"].set_visible(False)  
        # ax.get_xaxis().tick_bottom()  
    

        fpr, tpr, thresholds = metrics.roc_curve(map(lambda x: x+1, self.labels), self.preds, pos_label=2)
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % self.auc(print_m=False))
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.show()

    def plot_cm(self):
        cm = confusion_matrix(self.labels, map(lambda x: int(x > 0.5), self.preds))
        print(cm)
        plot_confusion_matrix(cm, ['Down', 'Up'], normalize=False, title="Confusion Matrix")

    def plot_cm_half_data(self):
        half_data_len = len(self.time_ordered_preds) / 2
        cm = confusion_matrix(self.labels[:half_data_len], map(lambda x: int(x > .5), self.preds[:half_data_len]))
        print(cm)
        plot_confusion_matrix(cm, ['Down', 'Up'], normalize=False, title="Confusion Matrix First Half")

        cm = confusion_matrix(self.labels[half_data_len:], map(lambda x: int(x > .5), self.preds[half_data_len:]))
        print(cm)
        plot_confusion_matrix(cm, ['Down', 'Up'], normalize=False, title="Confusion Matrix Seconde Half")

    def truncatedSVDAndTSNE(self):
        if len(self.X[0]) > 50:
            print('Running SVD')
            svd = TruncatedSVD(algorithm='randomized', n_components=50, n_iter=7,
            random_state=42, tol=0.0)
            X = svd.fit_transform(self.X)
        else:
            X = self.X
        print('Running TSNE')
        em = TSNE(n_components=2, random_state=42)
        X = em.fit_transform(random.sample(X, 1000))
        print('Making plot')
        plt.scatter(map(lambda x: x[0], X), map(lambda x: x[1], X), color=map(lambda x: 'r' if x==0 else 'g', self.labels))
        plt.show()

    def truncatedSVDAndTSNE3D(self):
        if len(self.X[0]) > 50:
            print('Running SVD')
            svd = TruncatedSVD(algorithm='randomized', n_components=50, n_iter=7,
            random_state=42, tol=0.0)
            X = svd.fit_transform(self.X)
        else:
            X = self.X
        print('Running TSNE')
        em = TSNE(n_components=3, random_state=42)
        X = em.fit_transform(random.sample(X, 1000))
        print('Making plot')
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(map(lambda x: x[0], X), map(lambda x: x[1], X),  map(lambda x: x[2], X), color=map(lambda x: 'r' if x==0 else 'g', self.labels))
        plt.show()


    def accuracy_over_time(self):
        global last_value 
        last_value = 0
        def get_mean_accuracy_for_label(x, label):
            global last_value
            accuracy = 0.0 
            count = 0
            for symbole in self.symbols:
                #Some stocks are missing days
                if not math.isnan(x['pred_{}'.format(symbole)]):
                    if x['labels_{}'.format(symbole)] == label:
                        count += 1
                        accuracy += int(int(x['pred_{}'.format(symbole)] > 0.5) == x['labels_{}'.format(symbole)])
            last_value = (accuracy / count) if count > 0 else last_value
            return (accuracy / count) if count > 0 else last_value
        accuracy_df = pd.DataFrame()
        accuracy_df['accuracy_of_positive_examples'] = self.df.apply(lambda x: get_mean_accuracy_for_label(x, label=1), axis=1)
        accuracy_df['50_day_rolling_mean'] = accuracy_df['accuracy_of_positive_examples'].rolling(50).mean()
        accuracy_df['std'] = accuracy_df['accuracy_of_positive_examples'].std()
        accuracy_df.plot()
        plt.show()

    def f1_score_over_time(self):
        def get_f1_score(x, average='binary'):
            preds = []
            labels = []
            for symbole in self.symbols:
                #Some stocks are missing days
                if not math.isnan(x['pred_{}'.format(symbole)]):
                    preds.append(int(x['pred_{}'.format(symbole)] > 0.5))
                    labels.append(x['labels_{}'.format(symbole)])
            return metrics.f1_score(labels, preds, average=average)
        score_df = pd.DataFrame()
        score_df['binary_f1_score'] = self.df.apply(lambda x: get_f1_score(x, average='binary'), axis=1)
        score_df['micro_f1_score'] = self.df.apply(lambda x: get_f1_score(x, average='micro'), axis=1)
        score_df['weighted_f1_score'] = self.df.apply(lambda x: get_f1_score(x, average='weighted'), axis=1)
        score_df.plot()
        plt.show()

    def precision_score_over_time(self):
        #tp / (tp + fp)
        def get_precision_score(x, average='binary'):
            preds = []
            labels = []
            for symbole in self.symbols:
                #Some stocks are missing days
                if not math.isnan(x['pred_{}'.format(symbole)]):
                    preds.append(int(x['pred_{}'.format(symbole)] > 0.5))
                    labels.append(x['labels_{}'.format(symbole)])
            return metrics.precision_score(labels, preds, average=average)
        #tp / (tp + fn)
        def get_recall_score(x, average='binary'):
            preds = []
            labels = []
            for symbole in self.symbols:
                #Some stocks are missing days
                if not math.isnan(x['pred_{}'.format(symbole)]):
                    preds.append(int(x['pred_{}'.format(symbole)] > 0.5))
                    labels.append(x['labels_{}'.format(symbole)])
            return metrics.recall_score(labels, preds, average=average)
        score_df = pd.DataFrame()
        score_df['binary_precision_score'] = self.df.apply(lambda x: get_precision_score(x, average='binary'), axis=1)
        score_df['micro_precision_score'] = self.df.apply(lambda x: get_precision_score(x, average='micro'), axis=1)
        score_df['weighted_precision_score'] = self.df.apply(lambda x: get_precision_score(x, average='weighted'), axis=1)
        score_df['binary_recall_score'] = self.df.apply(lambda x: get_recall_score(x, average='binary'), axis=1)
        score_df['micro_recall_score'] = self.df.apply(lambda x: get_recall_score(x, average='micro'), axis=1)
        score_df['weighted_recall_score'] = self.df.apply(lambda x: get_recall_score(x, average='weighted'), axis=1)
        score_df.plot()
        plt.show()

    def distribution_false_positive(self):
        fp_returns = []
        for r, p, l in zip(self.returns, self.preds, self.labels):
            if(int(p > 0.5) == 1 and int(p > 0.5) != l):
                fp_returns.append(r*100)
        print(len(fp_returns))
        plt.hist(fp_returns)
        plt.xlabel('Percent Returns')
        #plt.ylabel('True Positive Rate')
        plt.title('Distribution of False Positives')
        plt.show()

    def distribution_positive(self):
        p_returns = []
        for r, p, l in zip(self.returns, self.preds, self.labels):
            if(int(p > 0.5) == 1):
                p_returns.append(r*100)
        print(len(p_returns))
        plt.hist(p_returns, bins=60)
        plt.xlabel('Percent Returns')
        #plt.ylabel('True Positive Rate')
        plt.title('Distribution of Positive Predictions')
        plt.show()

    def mean_return_of_positive_preds(self):
        p_returns = []
        for r, p, l in zip(self.returns, self.preds, self.labels):
            if(int(p > 0.5) == 1):
                p_returns.append(r*100)
        mean_return = sum(p_returns) / len(p_returns)
        print(mean_return)
        return mean_return