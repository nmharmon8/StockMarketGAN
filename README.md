# Unsupervised Stock Market Features Construction using Generative Adversarial Networks(GAN)
Deep Learning constructs feature using only raw data. The leaned representation of the data outperforms expert features for many modalities including Radio Frequency ([Convolutional Radio Modulation Recognition Networks](https://arxiv.org/pdf/1602.04105.pdf)), computer vision ([Convolutional Deep Belief Networks for Scalable Unsupervised Learning of Hierarchical Representations](https://www.cs.princeton.edu/~rajeshr/papers/icml09-ConvolutionalDeepBeliefNetworks.pdf)) and audio classification ([Unsupervised feature learning for audio classification using convolutional deep belief networks](http://www.robotics.stanford.edu/~ang/papers/nips09-AudioConvolutionalDBN.pdf)). 
# GAN 
In the case of Convolutional Neural Networks (CNN), the data representation is learned in a supervised fashion with respect to a task such as classification. GANs lean features in an unsupervised fashion. The competitive learning process for GANs results in more of the possible feature space being explored. This reduces that potential for features being overfitted to the training data. Since the features are constructed in an unsupervised process classification algorithms trained on the features will generalize on a smaller amount of data. In fact, GANs promote generalization beyond the training data. 
![gan.png]({{site.baseurl}}/media/gan.png)
The Generator is trained to generate data that looks like historical price data of the target stocks over a distribution. The Discriminator is trained to tell the difference between the data from the Generator and the real data. The loss from the Discriminator (how the Discriminator has learned to tell if a sample in real or fake) is used to train the Generator to defeat the Discriminator. The competition between the Generator and the Discriminator forces the Discriminator to distinguish random from real variability while the Generator learns to map a distribution into the sample space.    
This project explores Bidirectional Generative Adversarial Networks(BiGANs) based on the paper [Adversarial Feature Learning](https://arxiv.org/pdf/1605.09782.pdf). The primary difference in the BiGAN the Discriminator learn to determine the joint probability P(X, Z) = real/fake. Where X is the sample and Z is the generating distribution. This, in turn, means that the Generator learns to encode a real sample into its generating distribution.  ![BiGAN.png]({{site.baseurl}}/media/BiGAN.png)

*Figuer from Adversarial Feature Learning*


This project makes a modification to the BiGAN. Rather than learning to encode a real sample into the generating distribution. The model learns to encode the features learned by the discriminator into the generating distribution. For historical stock data, this architecture outperformed the BiGAN architecture. 
# Approach 

**Data**
Historical prices of stocks are likely not very predictive of the future price of the stock, but it is free data. 

**Training**
The GAN is trained on 96 stocks off the Nasdaq. Each stock is normalized using a 20-day rolling window (data-mean)/(max-min). The last 2 years (504 days) of trading are held out as a test set. Time series of 20 day periods are constructed and used as input to the GAN. Once the GAN is finished training, the leaned encoding for the Discriminator features to the generation distribution is used as the new representation of the data. The features are not guaranteed to be predictive of the direction of the stock market, but for other modalities, they have been shown to work well. Random Forests is trained to classify whether the stock will gain 10% over the next 10 trading days. This creates an unbalanced training set so the majority class is undersampled before training the Random Forest. 

**Results**
Since the classes are unbalanced, due to not many stocks gaining 10% in 10 days, accuracy is a poor metric. If we always predicted that stocks would not go up then the accuracy would be above 90%. So instead of accuracy, we will use Area Under the Curve (AUC). Check out this video to learn more about [AUC](http://www.dataschool.io/roc-curves-and-auc-explained/). An AUC of 1 would be a perfect model while an AUC of 0.5 means that the model performs the same as randomly picking a label. We can visualize the performance of the classifier using a ROC curve. ![ReceiverOperatingCharacteristic.png]({{site.baseurl}}/media/ReceiverOperatingCharacteristic.png)
This show that the classifier is only a little better than random. 


