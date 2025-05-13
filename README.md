# Project: Univariate and multivariate time series forecasting of pressure sensors in particle accelerators using RNN, LSTM, and Transformer models

## Context and Problem Statement
- A particle accelerator is a large machine which accelerates particles (such as electron, protron) and produces synchrotron radiation (such as X-ray, Infrared, Ultraviolet). The radiation can be used for a wide range of research in materials science, chemistry and medicine.
- The high quality of particles and synchrotron radiation produced from the accelerator require stable vacuum systems. These vacuum systems rely on several pressure sensors placed around the accelerator to provide real-time pressure levels.
- Unforeseen pressure fluctuations, if not detected and addressed promptly, can lead to significant disruptions of the accelerator and potential damage of sensitive instruments.
- In order to early detect and accurately predict the pressure change, deep learning techniques including Recurrent Neural Networks (RNNs), Long Short-Term Memory (LSTM) networks, and Transformer models were applied.
- This project also aims to investigate the effectiveness of different deep learning architectures for predicting univariate and multivariate time series data from pressure sensors by comparing the performance, evaluation, and forecasting of these three models.
- The reason why I consider multivariate time series forecasting: because a pressure value from a sensor is dependent not only on its previous values but also on the pressure values of other sensors in the accelerator. Thus, the pressure values from all sensors should be considered to forecast the future pressure values.
- In this project, hyperparameter tuning with BayesianOptimization using a Gaussian process was performed for finding optimal number of neurons in hidden layers and optimal dropout rate in dropout layers.
- Regularization techniques consisting of dropout layer, batch normalization layer, and early stopping were used to address overfitting in the neural network.
- Custom callbacks of monitoring validation loss, early stopping, and learning rate scheduling were determined for improving model performance.

## Data source
- The pressure values were recorded every minute within 2 months (01/01/23 - 27/02/23) from 17 pressure sensors of Methodology Light Source (MLS), Germany.
- MLS is an electron circular accelerator providing synchrotron radiation in terahertz (THz) to extreme ultraviolet (EUV) regime.
- The pressure dataset includes 5 million timestamps and 17 features.

## Model
### 1. RNN model
- RNN is a type of DL model where the output from the previous time step is fed as an input to the current time step. The output from the previous time step is called a hidden state.
- The hidden state stores historical information of the sequence up to the current time step. At any time step, it is computed from the current input data and the previous hidden state.

### 2. LSTM model
- LSTM is a special kind of RNN and still includes the hidden state like RNN with an additional state, that is a cell state.
- The cell state acts as the long-term memory of the network that takes only useful information from the current input data and the previous hidden state of each time step and keeps it along the network to ensure that the gradient can pass across many time steps without vanishing or exploding. Thus, the cell state makes LSTM able to capture long-term dependencies without the vanishing and exploding gradient problems.
- In LSTM, the output of any time step is dependent on three values, that are the current input data,  the previous hidden state (the output of the previous time step), and the cell state. The previous hidden state refers to the short-term memory and the cell state refers to the long-term memory that is the meaning of Long Short-Term Memory or LSTM.
- There are two states that are being transferred between cells; the hidden state and  the cell state.
- LSTMs use three gates including forget gate, input gate and output gate, to filter out useless information and keep useful information from the sequential data throughout the network.

### 3. Transformer model
- Transformer model is a type of deep learning model that can capture long-range dependencies between input data in a sequence by using self-attention mechanisms.
- The self-attention mechanisms allow the transformer model to attend to all positions in the input sequence simultaneously to capture long-range dependencies and focus on relevant parts of the input sequence. That means they provide a parallel computation.
- The transformer model is split into 2 parts, an encoder and a decoder parts. For time series forecasting tasks, the encoder process is only required. The encoder operates on each element of the input sequence and projects them into query, key, and value vectors.

## Process
1. Data import
2. Data wrangling >>> select data points and features, pearson correlation
3. Data preprocessing for univariate time series >>> MinMaxScaler, split data, create sequential data
4. Data preprocessing for multivariate time series >>> MinMaxScaler, split data, create sequential data
5. Custom callbacks >>> Monitoring validation loss, Early stopping, and Learning rate scheduling
6. RNN model with hyperparameter tuning
7. LSTM model with hyperparameter tuning
8. Transformer model with hyperparameter tuning >>> Multi-head self-attention layer, Feed-forward neural network layer, Residual Connection, and Layer Normalization
9. Model training and hyperparameter tuning
   - Create a Keras Tuner >>> BayesianOptimization/Hyperband/RandomSearch
   - Train model with hyperparameter search
   - Train model again with best hyperparameters
   - Plot model performance
   - Predict model and plot prediction
   - Evaluate model
   - Save model, history, prediction, evaluation, and best hyper tuning
10. RNN with univariate times series
11. LSTM with univariate times series
12. Transformer with univariate times series
13. RNN with multivariate times series
14. LSTM with multivariate times series
15. Transformer with multivariate times series
16. Visualization of validation performance with accuracy and loss curves
17. Model evaluation for all approaches
18. Visualization of model predictions for all approaches
19. Time series forecasting for all approaches
