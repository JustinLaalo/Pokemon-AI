import tensorflow as tf
import matplotlib.pyplot as plt

class callBack(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get("val_accuracy") > 0.80):
            self.model.stop_training = True
class Model:
    def __init__(self):
        self.learningRateCallback = tf.keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=3, min_lr=1e-6)
        self.stopingCallback = callBack()
        self.activation = tf.keras.activations.relu
        self.initializer = tf.keras.initializers.HeNormal(seed=1)
        self.l2_regularizer = tf.keras.regularizers.l2(0.001)
        self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001)
        self.loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.AUTO)
        
    
    def loadData(self, split, seed, sub, imageHeight, imageWidth, batchSize):
        return tf.keras.preprocessing.image_dataset_from_directory(
            "/home/mir/Desktop/PokemonProject/kagglePokemonSet/archive/pokemon",
            image_size = (imageHeight, imageWidth),
            batch_size = batchSize,
            validation_split = split,
            seed = seed,
            subset = sub
            )
    
    def callLoadData(self):
        self.trainingSet = self.loadData(0.1, 1, "training", 60, 60, 16)
        self.validationSet = self.loadData(0.1, 1, "validation", 60, 60, 16)
        self.numClasses = len(self.trainingSet.class_names)

    def createModel(self):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Rescaling(1./255, input_shape=(60, 60,3)))
        model.add(tf.keras.layers.Conv2D(32,(3,3), activation=self.activation, kernel_initializer=self.initializer, kernel_regularizer=self.l2_regularizer))
        model.add(tf.keras.layers.MaxPooling2D())
        model.add(tf.keras.layers.Conv2D(64, (3,3), activation=self.activation, kernel_initializer=self.initializer, kernel_regularizer=self.l2_regularizer))
        model.add(tf.keras.layers.MaxPooling2D())
        model.add(tf.keras.layers.Conv2D(128, (3,3), activation=self.activation, kernel_initializer=self.initializer, kernel_regularizer=self.l2_regularizer))
        model.add(tf.keras.layers.MaxPooling2D())
        model.add(tf.keras.layers.Conv2D(256, (3,3), activation=self.activation,kernel_initializer=self.initializer, kernel_regularizer=self.l2_regularizer))
        model.add(tf.keras.layers.MaxPooling2D())
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.Dense(256, activation=self.activation,kernel_initializer=self.initializer, kernel_regularizer=self.l2_regularizer))
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.Dense(self.numClasses))
        return model

    def compileModel(self, model):
        model.compile(
        optimizer = self.optimizer,
        loss = self.loss,
        metrics=['accuracy'])

    def trainModel(self, model):
        plottingInfo = model.fit(
        self.trainingSet,
        validation_data = self.validationSet,
        epochs = 80,
        callbacks = [self.learningRateCallback, self.stopingCallback])
        return plottingInfo
        
    def saveModel(self, model):
        model.save("../models/nameModel")

def startTraining():
    currModel = Model()
    currModel.callLoadData()
    model = currModel.createModel()
    currModel.compileModel(model)
    plottingInfo = currModel.trainModel(model)
    currModel.saveModel(model)
    return plottingInfo

def savePlot(metric, epocs, training, validation):
    plt.plot(epocs, training, 'b', label='Training ' + metric)
    plt.plot(epocs, validation, 'r', label='Validation ' + metric)
    plt.title('Training and Validation ' + metric)
    plt.xlabel('Epochs')
    plt.ylabel(metric)
    plt.legend()
    plt.savefig("../plots/name/" + metric + ".png")
    plt.clf()

def plotInfo(plottingInfo):
    trainLoss = plottingInfo.history["loss"]
    trainAccuracy = plottingInfo.history["accuracy"]
    validationLoss = plottingInfo.history["val_loss"]
    validationAccuracy = plottingInfo.history["val_accuracy"]
    epochs = len(trainAccuracy)
    savePlot("Loss", range(epochs), trainLoss, validationLoss)
    savePlot("Accuracy", range(epochs), trainAccuracy, validationAccuracy)
    
plottingInfo = startTraining()
plotInfo(plottingInfo)