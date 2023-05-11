"""Methods for making predefined plots with matplotlib."""
import logging

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

from moljourney.preprocess import count_nan

LOGGER = logging.getLogger(__name__)
ALMOST_BLACK = "#262626"


def create_figure_and_axes() -> tuple[plt.Figure, plt.Axes]:
    """Create empty figure and axes."""
    fig, ax = plt.subplots(constrained_layout=True)
    return fig, ax


def style_pca_plot(
    ax: plt.Axes,
    pca: type[PCA],
    *,
    component1: int = 0,
    component2: int = 1,
    axis_label_text: str | None = "Scores",
    x_y_zero_lines: str | None = ":",
) -> None:
    """Style PCA plot with extra info.

    Parameters
    ----------
    ax : object like matplotlib.pyplot.Axes
        The axes to draw in.
    pca : object like sklearn.decomposition.PCA
        The fitted PCA object.
    component1 : int, optional
        The index of the PC used for the x-axis
    component2 : int, optional
        The index of the PC used for the y-axis
    axis_label_text : string, optional
        Can be used to put extra text of the labels of the
        axes. If None, it is not added.
    x_y_zero_lines : string, optional
        The line style for drawing x=0 and y=0 lines. If set
        to None, no lines are drawn.
    """
    var1 = pca.explained_variance_ratio_[component1] * 100
    var2 = pca.explained_variance_ratio_[component2] * 100
    if axis_label_text is None:
        axis_label_text = ""
    ax.set(
        xlabel=f"{axis_label_text} PC{component1+1} ({var1:.2g}%)".strip(),
        ylabel=f"{axis_label_text} PC{component2+1} ({var2:.2g}%)".strip(),
    )
    if x_y_zero_lines is not None:
        ax.axhline(y=0, ls=x_y_zero_lines, color=ALMOST_BLACK, lw=1)
        ax.axvline(x=0, ls=x_y_zero_lines, color=ALMOST_BLACK, lw=1)


def _add_fraction_axis(
    ax: plt.Axes, where: str, norm: int, fraction: float | None = None
) -> None:
    """Add extra axis with a percentage of original axis.

    Parameters
    ----------
    ax : object like matplotlib.pyplot.Axes, optional
        The axis to use for the figure. If None are give a new axis
        will be created.
    norm : int
        Norm for calculating a fraction (and precentage).
    where : string
        Defines where the extra axis should be placed.
    fraction : float, optional
        To display a line showing a specified fraction.
    """

    def forward(x):
        return 100 * (x / norm)

    def backward(x):
        return x * norm / 100

    if where not in ("top", "bottom", "left", "right"):
        if where == "horizontal":
            where = "top"
        else:
            where = "right"

    if where in ("top", "bottom"):
        ax2 = ax.secondary_xaxis(
            where,
            functions=(forward, backward),
        )
        ax2.set(xlabel="NaN (%)")
        if fraction is not None:
            ax.axvline(x=fraction * norm, ls=":", color=ALMOST_BLACK)
    else:  # right or left
        ax2 = ax.secondary_yaxis(
            where,
            functions=(forward, backward),
        )
        ax2.set(ylabel="NaN (%)")
        if fraction is not None:
            ax.axhline(y=fraction * norm, ls=":", color=ALMOST_BLACK)


def missing_fraction(
    data: pd.DataFrame,
    fraction: float | None = 0.2,
    ax: plt.Axes | None = None,
    sort: str | None = None,
    orient: str = "horizontal",
    show_all_features: bool = True,
    use_feature_names: bool = True,
    rotation: str | int | None = None,
    add_bar_label: bool = True,
) -> tuple[pd.DataFrame, plt.Figure | None]:
    """Show the fraction of missing numbers.

    Parameters
    ----------
    data : object like pd.DataFrame
        Our raw data.
    fraction : float, optional
        The cut-off for dropping variables. Note: No dropping is
        done here, it is only used for visualization. If not given,
        we will not show the fractions.
    ax : object like matplotlib.pyplot.Axes, optional
        The axis to use for the figure. If None are give a new axis
        will be created.
    sort : string, optional
        If this string is different from None, the NaN counts will be
        sorted (ascending if the string is "ascending"; descending for
        all other strings).
    orient : string, optional
        Used to orient the plot (vertical or horizontal).
    show_all_features : bool, optional
        If False, plot only features with missing values.
    use_feature_names : bool, optional
        If False, use the index of the variables as labels.
    rotation : str or int, optional
        Can be used to rotate the feature labels in case they
        are too long. Typically used together with orient,
        e.g. rotation="vertical", orient="vertical"
    add_bar_label : bool, optional
        If True, we will add a bar label for each rectangle.
    """
    nan_data = count_nan(data)

    if sort is not None:
        ascending = sort.lower() == "ascending"
        nan_data = nan_data.sort_values(by="nan-count", ascending=ascending)

    if len(nan_data[nan_data["nan-count"] > 0].index) < 1:
        LOGGER.debug("Will not plot missing bar figure - no missing data.")
        return nan_data, None

    select = nan_data
    if not show_all_features:
        select = nan_data[nan_data["nan-count"] > 0]

    if ax is None:
        fig, ax = create_figure_and_axes()
    else:
        fig = ax.get_figure()

    if use_feature_names:
        feature = "feature"
        feature_label = "Feature"
    else:
        feature = select.index
        feature_label = "Feature index"

    (line,) = ax.plot([], [])
    color = line.get_color()

    if orient == "horizontal":
        sns.barplot(
            data=select,
            y=feature,
            x="nan-count",
            ax=ax,
            orient="horizontal",
            color=color,
        )
        ax.set(ylabel=feature_label, xlabel="NaN (count)")
    else:
        sns.barplot(
            data=select,
            x=feature,
            y="nan-count",
            ax=ax,
            orient="vertical",
            color=color,
        )
        ax.set(xlabel=feature_label, ylabel="NaN (count)")

    if add_bar_label:
        ax.bar_label(ax.containers[0])

    if rotation is not None:
        if orient == "horizontal":
            feature_labels = ax.get_yticklabels()
        else:
            feature_labels = ax.get_xticklabels()
        for item in feature_labels:
            item.set_rotation(rotation)

    if fraction is not None:
        _add_fraction_axis(
            ax, where=orient, norm=len(data.index), fraction=fraction
        )

    return nan_data, fig
