# IMDb_review 
Guide to IMDb Sentiment Analysis with TensorFlow Hub
1. Import Libraries
This code starts by importing necessary libraries, including TensorFlow, TensorFlow Hub, TensorFlow Datasets, and NumPy.
2. Check TensorFlow and GPU Availability
It checks the version of TensorFlow, whether eager execution is enabled, the TensorFlow Hub version, and the availability of a GPU.
3. Load the IMDb Dataset
It loads the IMDb movie reviews dataset, splits it into training, validation, and test sets, and prepares it for training.
4. Preprocess the Data
The training, validation, and test data are preprocessed by shuffling, batching, and prefetching. This improves data loading efficiency during training.
5. Load Pre-trained Embedding Layer
It loads a pre-trained text embedding layer from TensorFlow Hub. This embedding layer converts text data into numerical vectors, making it suitable for use in a neural network.
6. Build the Neural Network Model
A Sequential model is created using TensorFlow's Keras API. It consists of:
The pre-trained embedding layer from TensorFlow Hub.
A Dense hidden layer with 16 units and ReLU activation function.
The output layer with 1 unit and a sigmoid activation function for binary sentiment classification.
7. Compile the Model
The model is compiled with an Adam optimizer, binary cross-entropy loss function (for binary classification), and accuracy as a metric.
8. Train the Model
The model is trained using the training data for 10 epochs with batch sizes of 512. Validation data is used for model evaluation during training. Training progress is displayed.
9. Evaluate the Model
After training, the model is evaluated on the test data to measure its performance. The results, including accuracy, are displayed.
10. Plot Training Progress
The code also includes a plot of the training and validation accuracy over epochs using Matplotlib. This allows you to visualize how the model's performance changes during training.
How to Use:
Users can execute this code in a Python environment that has the required libraries installed.
Ensure you have TensorFlow, TensorFlow Hub, TensorFlow Datasets, and Matplotlib installed.
Simply run the code, and it will automatically:
Load the IMDb dataset.
Preprocess the data for training.
Load a pre-trained text embedding layer.
Build and compile a neural network model.
Train the model.
Evaluate its performance.
Plot the training progress.
Users can use this code as a template for performing sentiment analysis on IMDb movie reviews or adapt it for other text classification tasks.
