## Basic-Neural-Network
An extension of Samson Zhang's simple Neural Network.

A fully-connected Neural Network built from scratch using only NumPy to classify Handwritten digits from MNIST dataset.
Forward propagation, backpropagation, and gradient descent have all been implemented manually.

## Results
* Validation accuracy 95%-96%
* Improved from initial 90% by extending network and increasing neurons per layer.

## Confusion Matrix
![App Dashboard](https://github.com/Defrost-Studios/Basic-Neural-Network/blob/main/Screenshot%202026-08-26%20225517.png)

According to the table, the value that the Neural Network gets incorrect the most is value 9. the Model makes 4 errors, mistaking the 
value 9 to be 7. The other errors it makes are mostly scattered, with notable ones being mainly around value 3. It seems as though the 
model incorrected predicted the value 3 nine times scattered across the row, with the highest predictions being 2, 5, and 7.

## Layers
* Input layer: 784 units (28×28 pixel images, flattened)
* Hidden layer 1: 128 units, ReLU activation
* Hidden layer 2: 64 units, ReLU activation
* Output layer: 10 units (digit classes 0–9), Softmax activation

## requirements and Running
In order to run the model, you need the Mnist_full.csv file. You can find the file here: https://drive.google.com/file/d/17Q2KDxXxRQJc3rsUXmVGLD-M_SNs2tBy/view?usp=sharing.
(about a 125 mb download)

Run This in terminal:
```
pip install numpy pandas matplotlib scikit-learn
python mnist_nn.py
```

## Credits
* Base code implemented using Samson Zhang's tutorial (https://www.youtube.com/watch?v=w8yWXqWQYmU)
* Used Ai tools selectively to resolve implementation bugs and Format Confusion Matrix Table
