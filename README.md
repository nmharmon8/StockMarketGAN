# Unsupervised Stock Market Features Construction using Generative Adversarial Networks(GAN)

Check out the write up of the project and the current results -- [StockMarketGAN](https://nmharmon8.github.io/StockMarketGAN/)

##Setup

*Dependencies:*

*[Python 2.7](https://www.python.org/download/releases/2.7/)

*[Pandas](https://pandas.pydata.org/)

*[Tensorflow](https://www.tensorflow.org/)

*[scikit-learn](http://scikit-learn.org/stable/)

*[Matplotlib](https://matplotlib.org/)

*[XGBoost](https://github.com/dmlc/xgboost)

First clone the project.

```bash
git clone git@github.com:nmharmon8/StockMarketGAN.git
```

To set up the environment, run "source env.sh" from the root directory of the project. 

```bash
cd StockMarketGAN
source env.sh
```


Next get a free account with [Quandl](https://www.quandl.com/) and get an API key. Export your key. This can be added to the ~/.bashrc

```bash
export QUANDL_KEY=you_key_here
```

Download the ticker data from Quandl

```bash
cd utils
python get_stock_data.py
```

The stocks can be changed by editing utils/companylist.csv.

To train a model go to the train_models directory. And run the training script. 

```bash
cd train_models
python train_bigan_shared_weights.py
```
The model will continue to train forever. Every 100 steps it will save its weights to the model's directory. To finish training simple kill the script. 

```bash
Ctrl+c
```

When the GAN is finished training the Random Forest model can be trained using the GAN features. The Random Forest model will load the most recently trained GAN from the model's directory. 

```bash
python train_rf_bigan_shared_weights.py
```

The random forest script will hold out a test set and report AUC and print a confusion matrix.

To test your model deploy them to the deployed model directory.   

```
cd ../utils
python deploy_model.py
```
Warning this will overwrite the pre-trained models.

Next, run the test script.

```
cd ../test_models
python test_bigan_rf_shared_weights.py
```

Have Fun.







