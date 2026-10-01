# Diagnostic Test

1) Repeat the training and evaluation of the kNN algorithm, but now testing different values for the number of neighbors.

2) Plot the accuracy for the test set together with another metric of your choice - check scikit for evaluation metrics. Save the figure.

3) Select the optimal number of neighbors, save the correspoding model in a pkl as shown above and submit on Fénix a zip with the following folders:

*   **figures** - containing the generated figure
*   **training_code** - this folder must contain all the code you developed during your experiments to train and select the best number of neighbors
*  **best_model** - this folder must contain:

    - a my_model.py file that contains and applies the predict function to a dataset and returns the predictions
    
    - the pickle with the best model (model.pkl)
    - aux.py with additional functions that your model may need