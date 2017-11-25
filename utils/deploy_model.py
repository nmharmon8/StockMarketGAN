from shutil import copyfile


with open('../models/checkpoint', 'rb') as f:
        model_name = next(f).split('"')[1]
  	model = "../models/{}".format(model_name)
  	copyfile('{}.data-00000-of-00001', '../deployed_models/bigan.data-00000-of-00001')
  	copyfile('{}.index', '../deployed_models/bigan.index')
  	copyfile('{}.meta', '../deployed_models/bigan.meta')
  	copyfile('../models/rf_bigan_sw.pkl', '../deployed_models/rf_bigan_sw.pkl')
