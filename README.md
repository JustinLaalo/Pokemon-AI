# Discription
A Comprehensive Analysis of Image-Based Pokemon and Pokemon Type Recognition

# Techstack
![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
 
# 1. Introduction

### 1.1 Background and motivation for the project 
For good reason, image classification is one of the most recognizable applications of artificial intelligence in the modern day. The ability to use software as a means to extract specific information from any form of imaging brings a lot of potential for streamlining countless vital operations. Image classification, for example, is particularly relevant in the medical field where it can be used to classify disease via imaging such as X-ray, CT, and MRI, providing faster and possibly more accurate diagnosis to patients. We wanted to break down this concept to a smaller scale and try to understand the foundational concepts that make image classification work. We recognize similar qualitative attributes to Pokemon such as Pokemon name and type that could be comparable to the process of classifying the type and status (benign/cancerous) of a tumor, or even unknown facts about species we have yet to see or encounter.

Furthermore, this project presents a unique opportunity for the members of this group to utilize modern-day technology and artificial intelligence to solve a nostalgic problem, “Who’s that Pokemon?”. “Who’s that Pokemon?” is a segment on the Pokemon animated TV series that premiered in the late 1990s. The segment became a mainstay of the series and less than a decade after its initial airing, the members of this group had the opportunity to create strong memories of yelling at the TV as we tried to solve the problem before the commercial break was over.

At its core, this project was an opportunity to integrate a piece of our childhood experience with modern applications and concepts of artificial intelligence at a small scale.

### 1.2 Related prior work

Due to their high accuracy, convolutional neural networks are widely used for image classification. A study on image classification for fauna notes that image classification in the field presents a challenging task as the data can come in many different poses and backgrounds. The ability of convolutional neural networks to take in input and assign importance to the aspects of the image like the animal itself, disregarding the background, makes it an accurate and reliable algorithm for image classification.

A study published in 2019 states that image classification is the primary driving force for processing massive amounts of medical images and uncovering information that goes unseen by the human eye. A shift towards deep learning from more traditional machine learning algorithms such as convolutional neural networks, principal component analysis, and support vector machines, is expected to push the capabilities in radiology to new heights. Whereas traditional machine learning methods require explicit user-defined features, deep learning uses extensive datasets without the need for user input.



### 1.3 Statement of objectives 

Our project aimed to detect the two primary attributes of a Pokemon: its name and type. We explored various iterations of the project and ultimately established three distinct objectives. These objectives involved the development of models to respectively identify a Pokemon based on a full-scale color image, determine its type using the same full-scale color image, and recognize the Pokemon based on a blacked-out image silhouette. Our goal was to ensure the functionality of all models regardless of the background and angle, achieved through the implementation of a neural network to standardize image inputs

# 2. Methods

###2.1 Methods from artificial intelligence used
One of the methods we used for our model in the project was callbacks. Oftentimes, these callback functions are used for early stopping a model in training, adjusting the learning rate, and logging metrics such as accuracy and loss. A call-back function is very similar to the concept of convergence where we stop at a certain threshold.
	
In our case, we created a callback class that allowed us to stop the model in training once it reached a validation accuracy of 80 percent. The other methods we used in our code include activation functions, initializer functions for our layers, regularizing functions, optimizers, and loss functions. 
If we go line by line for the __init__ function for our model class, the first line creates a learning rate callback that uses the ReduceLROnPlateau() from keras. This function allows us to dynamically change the learning rate as we get closer to a point where the model's accuracy starts to plateau. Once we are at that point, we want to reduce the learning rate (“take smaller steps to convergence”) so the model can still improve. We used the activation function tf.keras.activation.relu because it helps prevent linearity in a CNN, which eludes generalization and promotes the detection of features.

As for our initializer function, we used the HeNormal because this initializer does particularly well when paired with activation functions similar to Relu since the Relu isn’t necessarily symmetric. We decided to use HeNormal instead of Xavier because Xavier relies on the fact that the activation function is linear whereas Henormal doesn't make that assumption. Note that the initializer functions are responsible for initializing the weights and biases for a layer. After these parameters are initialized, the model will update them and find the most optimal values.

Next, we declared our regularization function which we use to prevent overfitting. Overfitting is a problem that often occurs when a model is learning the patterns and features of an image along with random patterns in the data. Ultimately, a model that is overfitted to a dataset may produce good results on the data it was trained upon, however, if tested with data/images never seen before, it may fail to perform well. Essentially, the tf.keras.regularizers.l2 solves the problem of overfitting by penalizing classes with large weights. This means that the model learns patterns that are consistent ultimately leading to better accuracy of the data and more accurate predictions when tested upon unseen data.

Lastly, we used optimizer functions and loss functions. Both of these functions are necessary since optimizers play a major role in improving the model by updating parameters such as weights and biases while the loss function is responsible for measuring the error between the real values and the predicted values; ultimately, we want to reduce the error as much as possible. In our case, we used tf.keras.optimizers.RMSProp is our optimizer since it performs well when it comes to data that is scattered. In our case, we had to classify a large number of Pokemon thus, our data was scattered. As for our loss function, we used tf.keras.losses.SparseCategoricalCrossentrophy. The reason we chose this loss function is that it performs well when it comes to predicting a single class when the data contains multiple classes.

Our model is based on CNN, which stands for Convolution Neural Network which is great for computing data that's set within a grid. Within a CNN we start with the Convolution Layer, where within this layer it goes through the grid-like data of the image and processes it through a filter. This input data, in this case image, is much larger than the output grid-like data once it goes through the filter. This filter slides across the image, learning features of the image such as texture, edges, and color. The next layer is the Max Pooling layer, this layer takes the output grid-like data from the convolution layer and then creates an even smaller output. This summarizes the output convolution layer, which helps decrease the amount of computation required. After repeating a couple of times, it then goes into the flattening layer which takes the data and makes it into a 1D array. After which we use the dropout layer to help prevent overfitting, in this layer we are temporarily dropping neurons for the current iteration, allowing each neuron to be more independent. Then in our dense layer, we pass the 1D arrays from the previous layers and compact them with each passing iteration. Lastly in our softmax layer, we are taking our last iteration of the dense layer and calculating the probability of each of the classes. 

Another method of artificial intelligence that we incorporated into the project was data preprocessing. Before the images are even passed into the model, we have to make sure they meet certain requirements. In our case, we used the tf.keras.preprocessing.image_dataset_from_directory() to make sure that the images from the dataset had a standard size of 60x60 as well as the subsets the dataset should be broken down into ie. training, validation, testing. We also had to clean up the dataset we imported from Kaggle because .png images were causing some errors, so we converted some to jpg/jpeg format.
2.2 Dataset (if applicable)
The data set we used was a collection of Pokemon from the first generation, where we had files for each of the Pokemon with their names. Within each of the Pokemons folders we had approximately 60 images, one thing we made sure of was that each of the Pokemon had relatively equal amounts of images. One thing we did to increase our accuracy was take out all the evolutions of Pokemon from the set. For our type dataset we had a folder for each of the types, within the folders we included images of Pokemon that are of that type (again we made it a relatively equal amount of images). Lastly, for our grayscale model, we created a dataset that held blacked-out images of Pokemon where the edges within the images were highlighted in white. We derived our dataset from a dataset we found on the internet of Pokemon in the first generation.

# 2.3 Validation strategy or experiments 
Image classification colored model:
We used the hold-out strategy, where we split our set into training, validation, and training. We also tested on real-life animals, to check whether it is looking at features of the image rather than just overlaying the images.

Image classification grayscale model:
With this model, we just used the hold-out strategy like we did with the first model, since the accuracy for this model we significantly lower than the model that took color into account

Type Classification model:
This model also used the hold-out validation strategy, but it also used 2 different ways of leave-on out strategy. For the first one, we took an image of a completely new Pokemon that wasn’t included in the dataset, for the second leave-one-out strategy we decided to just use a new image of the Pokemon. Both instances test whether the model just looks for a similar Pokemon image in the set, rather than looking at the features of the Pokemon

In the end since the hold-out strategy was the only common validation strategy among the models, we decided to compare each of the models using that method. The first model ended up with around a 79% accuracy, followed by the Type classification model with a 46% accuracy and lastly the greyscale model with a 35% accuracy. The greyscale model was first an experiment for deciding the type of the pokemon since the number of edges of a flying type pokemon with wings, would have more than a pokemon without wings. But in the end we learned that we could use a CNN to identify features of a pokemon and which would make it more accurate than an image that solely compares the outline of a pokemon. This was proven to be true since the lowest accuracy we got was from the greyscale model.


# 3. Results

### 3.1 Qualitative results
The above pictures show some of the qualitative results generated when testing our trained model. For example, it shows correct predictions of Pokemon along with some incorrect predictions that only occurred in the third model (Black and White). For the types classifier, we were aiming to predict at least one of the two types correctly. All the predictions made from the first two models were correct and were displayed in the demonstration as well. Furthermore, we had some edge cases as well, for example, we used a real-life image of a rhino that looks similar to a Pokemon called Rhyhorn. Our model was able to guess the names and types perfectly. 

### 3.2 Quantitative results



For quantitative results, we generated plots while training. We generated two plots, one comparing the training accuracy with validation accuracy, and one comparing the training loss and validation loss. They both help visualize the metrics, the training component shows how well the model performs when training data is given as input, while the validation component shows how the model performs on data never seen before. Furthermore, we can calculate the sensitivity by finding the ratio of positive predictions to total predictions. In our case, the first model had a sensitivity of 1 since out of the 8 images in the demonstration, all 8 were correctly predicted.

# 4. Discussion

### 4.1 Limitations of the work and directions for future work 
A possible direction we were hoping to take in the future with the model we created was identifying animals with just images and also what type each of the animals is (i.e. Carnivorous, Herbivores or Omnivores). This is similar to what we did with our model but one of the limitations we would run into with identifying the animal would be a large enough dataset, when we were doing our model. We needed to create/find a large enough dataset to get more accuracy from the model because it has more to train itself on. Another Limitation we would run into would be mutations, on earth there are around 8.7 million different species and with that, there are even more mutations that these species undergo each time. So creating a model to identify all animals on earth seems a little unrealistic, but we could come close to it by giving the most likely approximation of the animal it could be with features expressed on the animal itself. Since most of the mutations derive from a base animal, for example, a two-headed snake is still a snake with the mutation of an extra head. By looking at other features of the animal like its color, texture, and overall looks (excluding the head) we can derive that it is still a snake. This can also solve the problem of not having a big enough dataset, even though by doing this approach we might not have great accuracy, and some of the time we would have the wrong answer. This approach of approximating the animal will give us an answer but with some margin for error. Now another limitation that would arise if we did this method would be determining whether the animal is poisonous or not. This is because most of the animals in the world don’t show their poisonous features, such as the lionfish's spines. One way to solve this is using the fact that most poisonous animals have aposematic coloring meaning that their skin is colored in a way to warn predators that they are poisonous, but doing this way would also bring another problem to the table. This is because some animals adapt that coloring even if they aren’t poisonous, this would be hard to identify and it would lead to some misinformation.  

### 4.2 Implications of the work 
The project aimed to identify Pokemon names and types using color images, recognize Pokemon from blacked-out images and determine types based on visual cues. Using convolutional neural networks (CNNs) and AI techniques, the project succeeded notably in identifying Pokemon names and types from color images. However, limitations arose in recognizing Pokemon solely from blacked-out images which can be seen through the accuracy we received from testing where most of the Pokemon in the predictions were wrong giving us an accuracy of around 35%. Even though the types of the Pokemon were lower than half, our original goal was to have one of the 2 types to be correct. This allowed the model to have some leeway, so our actual accuracy was a little bit higher.While the achieved accuracy indicated moderate success for the blacked-out images, the accuracy from the colored images and identifying the types had great success. Most likely because these images had more qualities for the convolution neural network to use, such as texture and color. In conclusion, given the predictions made by the model, we can be sure that it isn’t simply choosing random actions but rather making logical decisions meaning we achieved desired goals, since our accuracies were above 1 percent (1/amount of classes, we had 63 pokemon).

Justin, Mir and Noah





