from isolation_forest import IForest
from bigan_shared_weights import GAN
import tensorflow as tf

gan = GAN(num_features=5, num_historical_days=20, generator_input_size=50)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    # saver.restore(sess, gan_model_path)
    with open('../models/checkpoint', 'rb') as f:
        model_name = next(f).split('"')[1]
    saver.restore(sess, "../models/{}".format(model_name))
    rf = IForest(num_historical_days=20, days=10, pct_change=5, test_size=504, model=gan, sess=sess)
    rf.train('../models/if_bigan.pkl', n_estimators=1000, max_features=50, contamination=0.1,  max_samples=1000)


