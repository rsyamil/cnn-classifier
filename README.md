# cnn-classifier

This is a simple guide to a vanilla convolutional neural network for classification and transfer learning, and is intended for beginners. 

## Convolutional neural network (CNN) for classification

The [first Jupyter Notebook](https://github.com/rsyamil/cnn-classifier/blob/master/cnn_classifier.ipynb) has straight-forward code to download the Fashion-MNIST dataset from Keras. The Fashion-MNIST dataset is used for this guide as some of the classes contain overlapping features and a human may find it challenging to distinguish between the classes. For example, images belonging to classes *top*, *coat* and *shirt* can be hard to tell apart. A convolutional neural network has convolving filters that can be trained to identify salient features in these images.

![Dataset](/readme/dataset.png)

More specifically, the architecture of the simple convolutional neural network involves three convolutional layers followed by two dense layers, before the cross-entropy loss function is evaluated. The first dense layer serves as a bottleneck layer with an output dimension of just two, to allow us to easily visualize the activations for each class labels once the model is trained. 

![Architecture](/readme/architecture.png)

The convolutional filters (or also called kernels) are responsible in extracting important high-level and low-level features from the images. The dense layers then determine the combination of these features that can be used to distinguish between one class label and another class label. 

![Filters](/readme/trained_filters.png)

The CNN is first trained with the entire dataset consisting of ten classes of images. The trained filters are extracted and visualized. When these trained filters are convolved on any given image, they produce intermediate representations of the image. These representations correspond to important distinguishing features in the image, such as edges and corners. For example, an input image of a boot may produce activations that indicate the presence of soles or heels and these features are then used in the following layers to determine class label association. 

![Activations](/readme/activations.png)

With a trained CNN, a feature space representation of each test image is obtained by running a forward pass on the set of test images. This feature space representation is not exactly the typical dimension reduction that involves transformation and back-transformation between high-dimensional representation and low-dimensional representation. However, it is a quick way to illustrate the inner workings of the CNN when used for classification tasks. The extracted feature space below shows clear clustering of the test images (color-coded by class label) in 2D space.

![Feature](/readme/feature_space.png)

For a simple transfer learning application in the [second Jupyter Notebook](https://github.com/rsyamil/cnn-classifier/blob/master/cnn-classifier_transfer_learning.ipynb), the Fashion-MNIST dataset is split into two parts. The first part is the *training-split* dataset (consisting of images from class labels *three* to *nine*) that is used to pre-train the CNN. The second part is the *transfer* dataset (consisting of images from class labels *zero* to *two*).

![Dataset_transfer](/readme/dataset_transfer.png)

After the CNN is trained with the *training-split* dataset, the convolutional filters are used to extract salient features from the *transfer* dataset. This is performed by locking the weights of the convolutional layers and only allowing the weights for the dense layers to be retrained using the *transfer* dataset.

![Architecture_transfer](/readme/architecture_transfer.png)

The feature space obtained from a CNN that is pre-trained only with the *training-split* dataset (left scatter plot) shows that images from class labels *zero* to *two* that have not been seen by the pre-trained CNN are mostly predicted as being from classes *dress*, *coat* or *shirt*. This is due to the similarity in features that are present in the images. When the *transfer* dataset is introduced to the pre-trained CNN and the dense layers are allowed to be re-trained, the feature space obtained (right scatter plot) shows different cluster for each of the class label. 

![Feature_transfer](/readme/feature_space_transfer.png)

In some applications, classifiers are trained on a very large dataset and transfer learning is done when the pre-trained model is fine-tuned on a smaller set of data. Transfer learning is an exciting concept and we may see interesting applications down the road!
