import torch
from utils import load_coeff, load_coeff_v2

def normalization_minmax_stack(real_tensor, pred_tensor):
    tensor = torch.cat((real_tensor, pred_tensor), dim=0)
    min_val = tensor.min()
    max_val = tensor.max()
    tensor_norm = (tensor - min_val) / (max_val - min_val)
    real_tensor_norm = tensor_norm[:real_tensor.shape[0]]
    pred_tensor_norm = tensor_norm[real_tensor.shape[0]:]
    return real_tensor_norm, pred_tensor_norm

def train_model_GCN(loader, model, optimizer, loss_fn, exc_loss_factor, vxc_loss_factor, device, mode=True):
    model.train(mode)  # Set the model to train mode if True and eval mode if False
    running_exc_loss = 0
    running_vxc_loss = 0
    running_combined_loss = 0

    for batch in loader:
        if vxc_loss_factor>0:
            batch.x.requires_grad = True
        # Reset gradients
        optimizer.zero_grad()
      
        pred, _ = model(batch.x, batch.edge_index, batch.batch)

        pred = pred.squeeze(1)
        loss = loss_fn(pred, batch.y)
        if vxc_loss_factor>0:
            lambda_vxc_pred = torch.autograd.grad(outputs=pred, inputs=batch.x, grad_outputs=torch.ones_like(pred), create_graph=True)[0].to(device)
            
            lambda_vxc_real = load_coeff(batch,device).to(torch.float64)
            lambda_vxc_real = lambda_vxc_real.view_as(lambda_vxc_pred)

            # do the normalization of real and pred together
            lambda_vxc_real_norm, lambda_vxc_pred_norm = normalization_minmax_stack(lambda_vxc_real, lambda_vxc_pred)
            vxc_loss = loss_fn(lambda_vxc_pred_norm, lambda_vxc_real_norm)
            
            combined_loss = exc_loss_factor * loss + vxc_loss * vxc_loss_factor
        else:
            combined_loss = loss

        if mode: # if training, calculate combined_loss gradient and update
            combined_loss.backward()
            optimizer.step()      
        running_exc_loss += loss.item()
        if vxc_loss_factor>0:
            running_vxc_loss += vxc_loss.item()
        else:
            running_vxc_loss = 0
        running_combined_loss += combined_loss.item()

    running_combined_loss = running_combined_loss / len(loader)
    running_exc_loss = running_exc_loss / len(loader)
    running_vxc_loss = running_vxc_loss / len(loader)

    return running_exc_loss, running_vxc_loss, running_combined_loss


def train_model_NN(loader, model, optimizer, loss_fn, exc_loss_factor, vxc_loss_factor, device, mode=True):
    model.train(mode)  # Set the model to train mode if True and eval mode if False
    running_exc_loss = 0
    running_vxc_loss = 0
    running_combined_loss = 0

    for batch in loader:
        x, y, name = batch
        if vxc_loss_factor>0:
            x.requires_grad = True
        # Reset gradients
        optimizer.zero_grad()

        pred, _ = model(x)
        
        pred = pred.squeeze(1)
        loss = loss_fn(pred, y)

        if vxc_loss_factor>0:
            lambda_vxc_pred = torch.autograd.grad(outputs=pred, inputs=x, grad_outputs=torch.ones_like(pred), create_graph=True)[0].to(device)

            lambda_vxc_real = load_coeff_v2(name,device).to(torch.float64)
            target_shape = lambda_vxc_pred.shape
            pad_length = target_shape[0] * target_shape[1] - lambda_vxc_real.numel()
            lambda_vxc_real_padded = torch.nn.functional.pad(lambda_vxc_real, (0, pad_length), mode='constant', value=0)
            lambda_vxc_real = lambda_vxc_real_padded.view(target_shape)
            
            lambda_vxc_real_norm, lambda_vxc_pred_norm = normalization_minmax_stack(lambda_vxc_real, lambda_vxc_pred)

            vxc_loss = loss_fn(lambda_vxc_pred_norm, lambda_vxc_real_norm)
            combined_loss = exc_loss_factor * loss + vxc_loss * vxc_loss_factor
        else:
            combined_loss = loss
    
        if mode: # if training, calculate combined_loss gradient and update
            combined_loss.backward()  # Compute gradients for the combined loss
            optimizer.step()      
        running_exc_loss += loss.item()

        if vxc_loss_factor>0:
            running_vxc_loss += vxc_loss.item()
        else:
            running_vxc_loss = 0

        running_combined_loss += combined_loss.item()

    running_combined_loss = running_combined_loss / len(loader)
    running_exc_loss = running_exc_loss / len(loader)
    running_vxc_loss = running_vxc_loss / len(loader)

    return running_exc_loss, running_vxc_loss, running_combined_loss