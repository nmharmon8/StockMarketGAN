from random_forest import RandomForest
from bigan_shared_weights import GAN
import tensorflow as tf
from sklearn.externals import joblib
from test_model import TestModel

gan = GAN(num_features=5, num_historical_days=20, generator_input_size=50)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess, '../deployed_models/bigan')
    rf = joblib.load('../deployed_models/rf_bigan_sw.pkl')

    def get_gan_features(batch):
    	return sess.run(gan.encoding, feed_dict={gan.X:batch, gan.keep_prob:1.0})

    def get_preds(X):
    	return map(lambda x: x[1], rf.predict_proba(X))


    tm = TestModel(num_historical_days=20, get_gan_features=get_gan_features, get_preds=get_preds, name='bigan_rf', days=10, pct_change=10, test_size=504)
   




# tm.auc()
tm.roc()
# tm.distribution_false_positive()
# tm.distribution_positive()
# tm.mean_return_of_positive_preds()
# tm.plot_cm()
# tm.plot_cm_half_data()
# tm.truncatedSVDAndTSNE()
