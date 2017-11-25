# Unsupervised Stock Market Features Construction using Generative Adversarial Networks(GAN)
Deep Learning constructs feature using only raw data. The leaned representation of the data outperforms expert features for many modalities including Radio Frequency ([Convolutional Radio Modulation Recognition Networks](https://arxiv.org/pdf/1602.04105.pdf)), computer vision ([Convolutional Deep Belief Networks for Scalable Unsupervised Learning of Hierarchical Representations](https://www.cs.princeton.edu/~rajeshr/papers/icml09-ConvolutionalDeepBeliefNetworks.pdf)) and audio classification ([Unsupervised feature learning for audio classification using convolutional deep belief networks](http://www.robotics.stanford.edu/~ang/papers/nips09-AudioConvolutionalDBN.pdf)). 
# GAN 
In the case of Convolutional Neural Networks (CNN), the data representation is learned in a supervised fashion with respect to a task such as classification. GANs lean features in an unsupervised fashion. The competitive learning process for GANs results in more of the possible feature space being explored. This reduces that potential for features being overfitted to the training data. Since the features are constructed in an unsupervised process classification algorithms trained on the features will generalize on a smaller amount of data. In fact, GANs promote generalization beyond the training data. 
![gan.png]({{site.baseurl}}/media/gan.png)
The Generator is trained to generate data that looks like historical price data of the target stocks over a distribution. The Discriminator is trained to tell the difference between the data from the Generator and the real data. The loss from the Discriminator (how the Discriminator has learned to tell if a sample in real or fake) is used to train the Generator to defeat the Discriminator. The competition between the Generator and the Discriminator forces the Discriminator to distinguish random from real variability while the Generator learns to map a distribution into the sample space.    
This project explores Bidirectional Generative Adversarial Networks(BiGANs) based on the paper [Adversarial Feature Learning](https://arxiv.org/pdf/1605.09782.pdf). The primary difference in the BiGAN the Discriminator learn to determine the joint probability P(X, Z) = real/fake. Where X is the sample and Z is the generating distribution. This, in turn, means that the Generator learns to encode a real sample into its generating distribution.  ![BiGAN.png]({{site.baseurl}}/media/BiGAN.png)*Adversarial Feature Learning*.
This project makes a modification to the BiGAN. Rather than learning to encode a real sample into the generating distribution. The model learns to encode the features learned by the discriminator into the generating distribution. For historical stock data, this architecture outperformed the BiGAN architecture. 
# Approach 

**Data**
Historical prices of stocks are likely not very predictive of the future price of the stock, but it is free data. Technical indicators are calculated using the historical prices of stocks. Not being a trader I don't know the validity of technical indicators, but if a sufficient number investors use technical indicators to invest such that they move the market, then the historical prices of stocks should suffice to predict the direction of the market correctly more then 50% of the time.

**Training**
The GAN is trained on 96 stocks off the Nasdaq. Each stock is normalized using a 20-day rolling window (data-mean)/(max-min). The last 356 days (1.4 years) of trading are held out as a test set. Time series of 20 day periods are constructed and used as input to the GAN. Once the GAN is finished training, the activated weighs from the last convolutional lays in the Discriminator is used as the new representation of the data. These features have information that is useful for telling whether a given sample is real or fake. They are not guaranteed to be predictive of the direction of the stock market. XGBoost is trained to classify whether the stock will go up or down over some period of time using the features extracted from the Discriminator.

**Testing**
The data the was held out in the training phase is run through the Discriminator portion of the GAN and the activated weights from the last convolutional layer are extracted. The extracted features are then classified using the trained XGBoost model. Multiple models are trained to predict over different periods of time.

**Downloading Data**
The first step is to download the historicl stock data. I use ([Quandl](www.quandl.com)) as my data source. They provide the basic stock data for free. You will need to create a free account to get an api key. 

```python
import urllib2
import os

#Your API key 
quandl_api_key = 'odn2xyvCE-sKzMK7LfTX'

#List of stocks to download
#Add more that you want to lean on
stock_symbols = ['AAPL', 'GOOG', 'COST', 'FB', 'INTU', 'ISRG']

url = 'https://www.quandl.com/api/v3/datasets/WIKI/{}.csv?api_key={}'

if not os.path.exists('./stock_data'):
    os.makedirs('./stock_data')
    
for symbol in stock_symbols:
    print('Downloading {}'.format(symbol))
    try:
        stock_url = url.format(symbol, quandl_api_key)
        response = urllib2.urlopen(stock_url)
        quotes = response.read()
        with open(os.path.join('./stock_data', symbol), 'wb') as f:
            f.write(quotes)
    except Exception as e:
        print('Failed to download {}'.format(symbol))
```

Now that we have the data we will start writing the GAN. Start by importing TensorFlow and numpy. 

```python
import tensorflow as tf
import numpy as np
import os

#Set random seed for repudiability 
tf.set_random_seed(42)
np.random.seed(42)
```

Now we will declare a GAN class that takes the number of features (Open, Close, High, Low, Volume), the number of days in each time series and the size the input distribution to the generator. 

```python
class GAN():

    def __init__(self, num_features, num_historical_days, generator_input_size):
```

Now we can begin defining the network architecture. The first step in to define the inputs to the network. The network will have two inputs (stock data time series and sample from distribution)

```python
#GAN Class init method
	def __init__(self, num_features, num_historical_days, generator_input_size):
		
		#Real samples input
		self.X = tf.placeholder(tf.float32, 
				shape=[None, num_historical_days, num_features])

		#Sample form distribution for input to generator
		self.Z = tf.placeholder(tf.float32, 
				shape=[None, generator_input_size])


		#Reshape input for convolutional layers
		X = tf.reshape(self.X, [-1, num_historical_days, 1, num_features])
```

Now we define the network architecture for the generator. The Generator will take in a samples from a distribution Z and learn to create fake stock data. 

![GeneratorNetwork.png]({{site.baseurl}}/media/GeneratorNetwork.png)

```python
#GAN Class init method
		#Reshape input for convolutional layers 
		#(the features are like RGB in an image)
		X = tf.reshape(self.X, [-1, num_historical_days, 1, num_features])

		#The output of the generator is the number of features per day
		#times the number of days
		generator_output_size = num_features*num_historical_days
		with tf.variable_scope('generator'):

			#Define weights for the first layer in the generator
			W1 = tf.Variable(tf.truncated_normal(
				[generator_input_size, generator_output_size*10]))
			b1 = tf.Variable(tf.truncated_normal([generator_output_size*10]))

			#Multiply the weights and the bias and active
			h1 = tf.nn.sigmoid(tf.matmul(self.Z, W1) + b1)

			#Define weights for the second layer in the generator
			W2 = tf.Variable(tf.truncated_normal(
				[generator_output_size*10, generator_output_size*5]))
			b2 = tf.Variable(tf.truncated_normal([generator_output_size*5]))

			#Multiply the weights and the bias and active
			h2 = tf.nn.sigmoid(tf.matmul(h1, W2) + b2)

			#Define weights for the third layer in the generator
			W3 = tf.Variable(tf.truncated_normal(
				[generator_output_size*5, generator_output_size]))
			b3 = tf.Variable(tf.truncated_normal([generator_output_size]))

			#Final multiplication and add
			generated_data = tf.matmul(h2, W3) + b3
			#Make shape match what the discriminator expects
			generated_data = tf.reshape(generated_data, 
				[-1, num_historical_days, 1, num_features])

			#Keep track of the weights and biases
			generator_weights = [W1, b1, W2, b2, W3, b3]
```
The discriminator network will be defined slightly differently. This is because there are essentially two paths through the network. One path is formed by the real stock data samples feeding the discriminator and the second path is formed by the generator creating a sample and feeding the discriminator. Therefore we will defined the weights for the discriminator, but then make two path that use the weights. 
```python
#GAN Class init method
		with tf.variable_scope("discriminator"):
			#[filter_height, filter_width, in_channels, out_channels]
			#convolution kernel
			k1 = tf.Variable(tf.truncated_normal([3, 1, num_features, 32],
				stddev=0.1,seed=SEED, dtype=tf.float32))
			b1 = tf.Variable(tf.zeros([32], dtype=tf.float32))

			#second convolution kernel
			k2 = tf.Variable(tf.truncated_normal([3, 1, 32, 64],
				stddev=0.1,seed=SEED, dtype=tf.float32))
			b2 = tf.Variable(tf.zeros([64], dtype=tf.float32))

			#third convolution kernel 
			k3 = tf.Variable(tf.truncated_normal([3, 1, 64, 128],
				stddev=0.1,seed=SEED, dtype=tf.float32))
			b3 = tf.Variable(tf.zeros([128], dtype=tf.float32))

			#Fully connected layer
			W1 = tf.Variable(tf.truncated_normal([18*1*128, 128]))
			b4 = tf.Variable(tf.truncated_normal([128]))

			#Fully connected layer
			W2 = tf.Variable(tf.truncated_normal([128, 1]))

			#Keep track of weights
			discriminator_weights = [k1, b1, k2, b2, k3, b3, W1, b4, W2]
```
Now that the weights for the discriminator are defined we will make a function to construct the operation over the weights. The function can be called multiple times to make different paths.  
```python
#GAN Class init method
		def discriminator(X):
			#Create a convolution that will use the kernel defined 
			#above and convolve it over X
			conv = tf.nn.conv2d(X,k1,strides=[1, 1, 1, 1],padding='SAME')
			#active the result with relu
			relu = tf.nn.relu(tf.nn.bias_add(conv, b1))

			#Use the next kernel to convolve over the result 
			conv = tf.nn.conv2d(relu, k2,strides=[1, 1, 1, 1],padding='SAME')
			relu = tf.nn.relu(tf.nn.bias_add(conv, b2))

			conv = tf.nn.conv2d(relu, k3, strides=[1, 1, 1, 1], padding='VALID')
			relu = tf.nn.relu(tf.nn.bias_add(conv, b3))

			#Find the size of the result
			flattened_convolution_size = int(relu.shape[1]) 
				* int(relu.shape[2]) * int(relu.shape[3])
			
			flattened_convolution = features = tf.reshape(relu, 
				[-1, flattened_convolution_size])

			#Put convolutional features through fully connected layer
			h1 = tf.nn.relu(tf.matmul(flattened_convolution, W1) + b4)

			#Final fully connected layer
			D_logit = tf.matmul(h1, W2)

			#return the logit and the features
			return D_logit, features
```
Constructs the paths through the network using the discriminator function. The first path will take the real samples and run the discriminator on them. The second path will run the generator and run tha discriminator on the generated data.  
```python
#GAN Class init method
		#construct the first path through the network -- real samples 
		#into the discriminator
		D_logit_real, self.features = discriminator(X)

		#construct the second path through the network -- run generator 
		#and pass the generated data to the discriminator
        D_logit_fake, _ = discriminator(generated_data)
```
Finally will will write the loss function.
