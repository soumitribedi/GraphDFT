
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

def evaluation(model,model_type,data_loader,device,save_fig_dir):
    """
    Evaluates a trained model on a test dataset and computes performance metrics.

    This function sets the model to evaluation mode, iterates over the test dataset 
    provided by the `data_loader`, and performs predictions. It computes and prints 
    the Mean Squared Error (MSE) and R-squared (R2) score between the true and 
    predicted values. Additionally, it generates a scatter plot to visualize the 
    correlation between the true and predicted values.

    Parameters:
    -----------
    model : torch.nn.Module
        The trained PyTorch model to be evaluated.
    
    data_loader : torch_geometric.data.DataLoader
        A data loader that provides batches of data for evaluation. The batches 
        are expected to contain the following attributes: `x`, `edge_index`, 
        `edge_attr`, `batch`, and `y`.
    
    device : torch.device
        The device on which the model and data should be loaded 
        (e.g., 'cpu' or 'cuda').

    Returns:
    --------
    None
        This function does not return any value but prints the evaluation metrics 
        and displays a scatter plot comparing the true and predicted values.

    Notes:
    ------
    - The function assumes that the `model` has a forward method that takes in 
      node features (`x`), edge indices (`edge_index`), edge attributes (`edge_attr`), 
      and a batch index (`batch`), and returns predictions.
    - The evaluation metrics are computed using scikit-learn functions.
    - The scatter plot is generated using Matplotlib.
    """
    model.eval()

    print("\n Evaluating the model on the test case\n")
    
    true_values = []
    predictions = []

    with torch.no_grad():
        for batch in data_loader:
            # batch = batch.to(device)

            # Forward pass to get predictions
            if model_type=="GCN":
                pred, _ = model(batch.x, batch.edge_index, batch.batch)
            elif model_type=="NNSimple":
                x, y, _ = batch
                pred, _ = model(x)
                pred = pred.squeeze(1)
            else:
                pred, _ = model(batch.x, batch.edge_index, batch.edge_attr.float(), batch.batch)

            # Collect true and predicted values
            if model_type=="GCN":
                true_values.append(batch.y.cpu())
            elif model_type=="NNSimple":
                true_values.append(y)
            predictions.append(pred.cpu())

    # Concatenate lists of tensors into a single tensor
    true_values = torch.cat(true_values)
    predictions = torch.cat(predictions)

    # Convert tensors to numpy arrays for metric calculations
    true_values_np = true_values.numpy()
    predictions_np = predictions.numpy()

    # Calculate evaluation metrics
    mse = mean_squared_error(true_values_np, predictions_np)
    r2 = r2_score(true_values_np, predictions_np)

    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"R-squared (R2 Score): {r2:.4f}")

    print("\nTrue Values vs Predicted Values:")
    for true_val, pred_val in zip(true_values_np, predictions_np):
        print(f"True: {true_val}, Predicted: {pred_val}")

    plt.figure(figsize=(8, 8))
    plt.scatter(true_values_np, predictions_np, alpha=0.5, color='b', label='Predictions')
    plt.plot([true_values_np.min(), true_values_np.max()], [true_values_np.min(), true_values_np.max()], 'r--', lw=2, label='Ideal Fit')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('Predicted vs True Values')
    plt.legend()
    plt.grid(True)
    if save_fig_dir:
        plt.savefig(save_fig_dir, dpi=500)
    plt.show()

    return