from random_forest import RandomForest
from bigan_shared_weights import GAN
import tensorflow as tf

gan = GAN(num_features=5, num_historical_days=20, generator_input_size=50)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess, '../deployed_models/bigan')
    # with open('../models/checkpoint', 'rb') as f:
    #     model_name = next(f).split('"')[1]
    # saver.restore(sess, "../models/{}".format(model_name))
    rf = RandomForest(num_historical_days=20, days=10, pct_change=10, test_size=504, model=gan, sess=sess)
    rf.train('../models/rf_bigan_sw.pkl', n_estimators=3000, max_depth=1, max_features=1)


