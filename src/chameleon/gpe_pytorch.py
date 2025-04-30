import gpytorch
import torch
import warnings

class ExactGPModel(gpytorch.models.ExactGP):
    """GPyTorch model for exact GP regression of multiple outputs and single/multiple inputs"""
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class MultitaskGPModel(gpytorch.models.ExactGP):
    """GPyTorch model for exact GP regression of multiple outputs with a more flexible structure"""

    def __init__(self, train_x, train_y, likelihood, num_tasks, rank=2):
        # Ensure consistent data types
        train_x = train_x.to(torch.float32)
        train_y = train_y.to(torch.float32)

        # Call the parent constructor with super()
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)

        self.num_tasks = num_tasks

        # Mean module (one per task)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=num_tasks
        )

        # More flexible kernel - using ScaleKernel around a MaternKernel
        base_kernel = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=2.5)
        )

        # Use higher rank for better task correlation modeling
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            base_kernel, num_tasks=num_tasks, rank=rank
        )

    def forward(self, x):
        # Ensure input is float32
        x = x.to(torch.float32)

        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)
