import torch
import nibabel as nib
import numpy as np


def torch_from_nib_path(path, device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')):
    return torch.from_numpy(nib.load(path).get_fdata()[np.newaxis, np.newaxis]).float().to(device)


def save_tensor_as_nii(tensor, name, vox=(1, 1, 1)):
    return nib.save(nib.Nifti1Image(tensor.squeeze().detach().cpu().numpy(), np.diag((*vox, 1))), name)


def save_array_as_nii(arr, name):
    return nib.save(nib.Nifti1Image(arr, np.eye(4)), name + '.nii')


def torch_from_numpy_path(path, device=torch.device('cuda')):
    return torch.from_numpy(np.load(path)[np.newaxis, np.newaxis]).float().to(device)