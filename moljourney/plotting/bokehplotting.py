import pandas as pd
import numpy as np
#from numpy.typing import ArrayLike
from pandas._typing import ArrayLike


def guess_categorical_numerical(data: ArrayLike ) -> None:
    categories = list(set(data))
    if len(categories) < 20:
        map_type = "categorical"
    else:
        map_type = "numeric"



def scatterplot(
    data: pd.DataFrame | None = None,
    x: str | np.ndarray | None = None,
    y: str | np.ndarray | None = None,
    hue: str | ArrayLike | None = None,
    style: str | ArrayLike | None = None,
    size: str | ArrayLike | None = None,
) -> None:
    """Create a scatter plot."""
    xdata, ydata = x, y
    if isinstance(x, str) and data is not None:
        xdata = data[x]
    if isinstance(y, str) and data is not None:
        ydata = data[y]
    
    if xdata is None and ydata is None:
        return None

    if xdata is None and ydata is not None:
        xdata = np.arange(len(ydata))

    if xdata is not None and ydata is None:
        ydata = np.arange(len(xdata))


