from xgb import XGB

xgb = XGB(num_historical_days=20, days=10, pct_change=10, test_size=504, generator_input_size=10, gan_model_path='../deployed_models/bigan')

params = {}
#params['objective'] = 'multi:softprob'
params['eta'] = 0.01
#params['num_class'] = 2
params['max_depth'] = 2
params['subsample'] = 0.1
params['colsample_bytree'] = 1.0
params['eval_metric'] = 'auc'
params['scale_pos_weight'] = 24


xgb.train(params, '../models/xgb_bigan.pkl')


