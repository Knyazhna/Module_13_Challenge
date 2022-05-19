# Venture Funding with Deep Learning

This assignment is on deep learning. Assignment done on basis of Neural Networking. The business team has given you a CSV file containing more than 34,000 organizations that have received funding from Alphabet Soup over the years. The CSV file contains a variety of information about each business, including whether or not it ultimately became successful. With your knowledge of machine learning and neural networks, you decide to use the features in the provided dataset to create a binary classifier model that will predict whether an applicant will become a successful business.

---
## Technologies

Venture Funding with Deep Learning project leverages python 3.7 with the following packages:

  [Pandas](https://github.com/pandas-dev/pandas "Pandas") 
  
We will be using GoogleColab to run this project.
  
 --- 
  ## Installation Guide
First install the following libraries and dependencies.

```
# Imports
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,OneHotEncoder
```

---
## Usage

* Steps for the Challenge:
    * Prepare the data for use on a neural network model.

    * Compile and evaluate a binary classification model using a neural network.

    * Optimize the neural network model.

* Prepare the Data for Use on a Neural Network Model
    Using scikit-learn’s StandardScaler(), preprocess the dataset done so later can be used it to compile and evaluate the neural network model.

    Open the venture_funding_with_deep_learning, and to complete the following data preparation steps:

    Read the applicants_data.csv file into a Pandas DataFrame. Review the DataFrame, looking for categorical variables that will need to be encoded, as well as columns that could eventually define your features and target variables.

    Drop the “EIN” (Employer Identification Number) and “NAME” columns from the DataFrame, because they are not relevant to the binary classification model.

    Encode the dataset’s categorical variables using OneHotEncoder, and then place the encoded variables into a new DataFrame.

    Add the original DataFrame’s numerical variables to the DataFrame containing the encoded variables.

    Note To complete this step, you will employ the Pandas concat() function that was introduced earlier in this course.

    Using the preprocessed data, create the features (X) and target (y) datasets. The target dataset should be defined by the preprocessed DataFrame column “IS_SUCCESSFUL”. The remaining columns should define the features dataset.

    Split the features and target sets into training and testing datasets.

    Use scikit-learn's StandardScaler to scale the features data.

* Compile and Evaluate a Binary Classification Model Using a Neural Network
    Using the knowledge of TensorFlow to design a binary classification deep neural network model. This model should use the dataset’s features to predict whether an Alphabet Soup–funded startup will be successful based on the features in the dataset. Consider the number of inputs before determining the number of layers that your model will contain or the number of neurons on each layer. Then, compile and fit your model. Finally, evaluate your binary classification model to calculate the model’s loss and accuracy.

    To do so, complete the following steps:

    Create a deep neural network by assigning the number of input features, the number of layers, and the number of neurons on each layer using Tensorflow’s Keras. Hint You can start with a two-layer deep neural network model that uses the relu activation function for both layers.

    Compile and fit the model using the binary_crossentropy loss function, the adam optimizer, and the accuracy evaluation metric. Hint When fitting the model, start with a small number of epochs, such as 20, 50, or 100.

    Evaluate the model using the test data to determine the model’s loss and accuracy.

    Save and export your model to an HDF5 file, and name the file AlphabetSoup.h5.

* Optimize the Neural Network Model
    Using the knowledge of TensorFlow and Keras, optimize your model to improve the model's accuracy. Even if you do not successfully achieve a better accuracy, you'll need to demonstrate at least two attempts to optimize the model. You can include these attempts in your existing notebook. Or, you can make copies of the starter notebook in the same folder, rename them, and code each model optimization in a new notebook.

    Note You will not lose points if your model does not achieve a high accuracy, as long as you make at least two attempts to optimize the model.

    To do so, complete the following steps:

    Define at least three new deep neural network models (the original plus 2 optimization attempts). With each, try to improve on your first model’s predictive accuracy. Rewind Recall that perfect accuracy has a value of 1, so accuracy improves as its value moves closer to 1. To optimize your model for a predictive accuracy as close to 1 as possible, you can use any or all of the following techniques:

    Adjust the input data by dropping different features columns to ensure that no variables or outliers confuse the model.

    Add more neurons (nodes) to a hidden layer.

    Add more hidden layers.

    Use different activation functions for the hidden layers.

    Add to or reduce the number of epochs in the training regimen.

    After finishing your models, display the accuracy scores achieved by each model, and compare the results.

    Save each of your models as an HDF5 file.

    HDF5 file saved in Resources folder in the name of: Aphabetsoup.h5 Alternative_file1.h5 Alternative_file2.h5


---
## Contributors

* Brought to you by Olga Koryachek.
* Email: olgakoryachek@live.com
* [LinkedIn](https://www.linkedin.com/in/olga-koryachek-a74b1877/?msgOverlay=true "LinkedIn")

---
## License

Licensed under the [MIT License](https://choosealicense.com/licenses/mit/)
