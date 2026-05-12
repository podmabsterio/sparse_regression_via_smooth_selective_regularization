import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings(
    "ignore",
    message="Orthogonal matching pursuit ended prematurely*",
    category=RuntimeWarning,
)
