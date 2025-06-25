import pytest
import torch
from torch.testing import assert_close

from utils.descriptors import project_density_integrate

@pytest.mark.parametrize("M,N", [(3,2), (5,3)])
def test_project_density_integrate(M, N):
    torch.manual_seed(42)
    
    # Create random psi_tilde (M x N), rho (M x 1), and w (M x 1)
    psi_tilde = torch.randn(M, N, dtype=torch.double)
    rho = torch.randn(M, 1, dtype=torch.double)
    w = torch.randn(M, 1, dtype=torch.double)
    
    # Compute the expansion coefficients with the function
    c = project_density_integrate(psi_tilde, rho, w)

    # Check that the output shape is (N, 1)
    assert c.shape == (N, 1), f"Expected shape {(N,1)}, got {c.shape}"

    # Manually compute the expected result
    integrand = psi_tilde.conj() * rho * w  # (M x N)
    expected = torch.sum(integrand, dim=0, keepdim=True).T  # (N x 1)

    # Compare the computed and expected results
    assert_close(c, expected, rtol=1e-7, atol=1e-7)  