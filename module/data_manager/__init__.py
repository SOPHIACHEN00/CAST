from .manager import *
from .structured_classification_manager import *
from .structured_regression_manager import *
from .graphical_classification_manager import *

try:
    from .text_classification_manager import *
except ModuleNotFoundError:
    pass


# # NEW!
# from module.data_manager.structured_classification_manager import TicTacToe

# def get_data(seed, dataset, distribution, alpha, num_parts):
#     if dataset == "tictactoe":
#         loader = TicTacToe()
#     else:
#         raise NotImplementedError(f"Dataset {dataset} not supported yet.")

#     loader.read(test_ratio=0.2, shuffle_seed=seed, cuda=False)

#     if distribution == "quantity skew":
#         dirichlet_distribution = np.random.dirichlet(alpha=[alpha] * num_parts)
#         loader.ratio_split(dirichlet_distribution)
#     elif distribution == "uniform":
#         loader.uniform_split(num_parts)
#     else:
#         raise ValueError(f"Unknown distribution: {distribution}")

#     return loader
