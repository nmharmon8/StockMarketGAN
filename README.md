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

Next get a free account with [Quandl](https://www.quandl.com/) and get an api key. Export your key. This can be added to the ~/.bashrc

```bash
export QUANDL_KEY=you_key_here
```

Download the ticker data from Quandl

```bash
cd utils
python get_stock_data.py
```

The stocks can be change by editing utils/companylist.csv.

To train a model go to the train_models directory.

```bash
cd train_models







