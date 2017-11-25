import os
import pandas as pd
from gan import GAN
import random
import tensorflow as tf
import numpy as np

random.seed(42)
np.random.seed(42)
#tf.reset_default_graph()
tf.set_random_seed(0)

class TrainGan:

    def __init__(self, num_historical_days, batch_size=32, generator_input_size=50):
        self.batch_size = batch_size
        self.data = []
        self.generator_input_size = generator_input_size
        files = [os.path.join('./stock_data', f) for f in os.listdir('./stock_data')]
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
            df = ((df -
            df.rolling(num_historical_days).mean().shift(-num_historical_days))
            /(df.rolling(num_historical_days).max().shift(-num_historical_days)
            -df.rolling(num_historical_days).min().shift(-num_historical_days)))
            #Drop the last 10 day that we don't have data for
            df = df.dropna()
            #Hold out the last year of trading for testing
            #Padding to keep labels from bleeding
            #df = df[400:]
            #This may not create good samples if num_historical_days is a
            #mutliple of 7
            for i in range(num_historical_days, len(df), num_historical_days):
                self.data.append(df.values[i-num_historical_days:i])

        self.gan = GAN(num_features=5, num_historical_days=num_historical_days,
                        generator_input_size=self.generator_input_size)

    def random_batch(self, batch_size=128):
        print(len(self.data))
        batch = []
        while True:
            batch.append(random.choice(self.data))
            if (len(batch) == batch_size):
                yield batch
                batch = []

    def train(self, print_steps=100, display_data=100, save_steps=1000):
        if not os.path.exists('./models'):
            os.makedirs('./models')
        sess = tf.InteractiveSession()
        G_loss = 0
        G_count = 0
        D_loss = 0
        D_count = 0
        D_last_loss = 0
        G_last_loss = 0

        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        # if os.path.exists('./models/checkpoint'):
        #     with open('./models/checkpoint', 'rb') as f:
        #         model_name = next(f).split('"')[1]
        #     saver.restore(sess, "./models/{}".format(model_name))
        for i, X in enumerate(self.random_batch(self.batch_size)):
            if (D_last_loss + 0.0) >= G_last_loss or i < 1000:
                if i % 1 == 0:
                    _ = sess.run([self.gan.D_solver, self.gan.clip_D], feed_dict=
                            {self.gan.X:X, self.gan.keep_prob:0.2, self.gan.Z:self.gan.sample_Z(self.batch_size, self.generator_input_size)})
            D_loss_curr = sess.run(self.gan.D_loss, feed_dict=
                        {self.gan.X:X, self.gan.keep_prob:1.0, self.gan.Z:self.gan.sample_Z(self.batch_size, self.generator_input_size)})
            D_loss += D_loss_curr
            D_last_loss = D_loss_curr
            D_count += 1
            if G_last_loss >= (D_last_loss + 0.0) or i < 1000:
                if i % 1 == 0:
                    _ = sess.run([self.gan.G_solver],
                            feed_dict={self.gan.keep_prob:0.1, self.gan.Z:self.gan.sample_Z(self.batch_size, self.generator_input_size)})
            G_loss_curr = sess.run(self.gan.G_loss, feed_dict={self.gan.keep_prob:1.0, self.gan.Z:self.gan.sample_Z(self.batch_size, self.generator_input_size)})
            G_loss += G_loss_curr
            G_last_loss = G_loss_curr
            G_count += 1
            if (i+1) % print_steps == 0:
                if D_count > 0:
                    print('Step={} D_loss={}'.format(i, D_loss/D_count))
                if G_count > 0:
                    print('Step={} G_loss={}'.format(i, G_loss/G_count))
                G_loss = 0
                D_loss = 0
                D_count = 0
                G_count = 0
            if (i+1) % save_steps == 0:
                saver.save(sess, './models/gan.ckpt', i)


if __name__ == '__main__':
    gan = TrainGan(num_historical_days=20, batch_size=128, generator_input_size=25)
    gan.train()
