import numpy as np
import pandas as pd
from bokeh.core.enums import MarkerType
from bokeh.core.property.vectorization import Field
from bokeh.models import ColumnDataSource
from bokeh.palettes import all_palettes as bokeh_palettes
from bokeh.plotting import figure
from bokeh.transform import factor_cmap, factor_mark, linear_cmap
from pandas._typing import ArrayLike
from pandas.api.types import is_numeric_dtype
from sklearn.preprocessing import MinMaxScaler

SIZES = list(range(6, 22, 3))

BOKEH_MARKERS = [
    "circle",
    "square",
    "triangle",
    "plus",
    "hex",
    "star",
    "asterisk",
    "inverted_triangle",
    "square_pin",
    "triangle_pin",
]
for i in MarkerType:
    if i not in BOKEH_MARKERS:
        BOKEH_MARKERS.append(i)


def get_size_mapping(
    values: ArrayLike | list[str] | list[int], set_sizes: list = SIZES
) -> list:
    """Create a mapping for the size from the given values.

    The parameter `set_sizes` determines the possible sizes that can
    be returned. Note that this number of items in `set_sizes` will
    then also define the number of categories we can handle. If the
    number of categories is larger than `set_sizes`, this method will
    just create a linear mapping between the given values and the
    min/max in `set_sizes`.

    Args:
        values (ArrayLike): The values to create a mapping for.
        set_sizes (list): The possible sizes to map to.

    Returns:
        sizes (list): The sizes to use for plotting.

    """
    sizes = []
    categories = list(set(values))
    if len(categories) <= len(set_sizes):
        groups = pd.Categorical(values)
        sizes = [set_sizes[i] for i in groups.codes]
    else:
        if is_numeric_dtype(np.array(values)):
            # If we have numerical values, just do linear norm:
            scaler = MinMaxScaler(
                feature_range=(min(set_sizes), max(set_sizes))
            )
            sizes = scaler.fit_transform(np.array(values).reshape(-1, 1))
            sizes = list(sizes.flatten())
        else:
            # If we have not numerical, and not categorical, we
            # still can just create a linear mapping, but we
            # base it on the indexes of the unique items:
            index = np.array(range(len(categories)))
            scaler = MinMaxScaler(
                feature_range=(min(set_sizes), max(set_sizes))
            )
            sizes = scaler.fit_transform(index.reshape(-1, 1))
            sizes = list(sizes.flatten())
            mapping = {key: i for key, i in zip(categories, sizes)}
            sizes = [mapping[i] for i in values]
    return sizes


def find_bokeh_palettes(ncolors: int) -> list[str]:
    """Get possible bokeh palettes for the given number of colors.

    Args:
        ncolors (int): The required number of colors.

    Returns:
        palettes (list[str]): A list of possible palettes.

    """
    palettes = []
    for key, val in bokeh_palettes.items():
        if any(ncolors <= i for i in val):
            palettes.append(key)
    return palettes


def get_bokeh_palette(categories: int, palette: str = "Bright") -> str:
    """Get a bokeh palette with enough colors for the given categories.

    Args:
        categories (int): The number of categories to consider.
        palette (str): The preferred palette to use.

    Returns:
        (str): The palette, represented by a string.

    """
    candidates = []
    preference = [palette] + [
        "Colorblind",
        "Category10",
        "Category20",
        "TolRainbow",
        "Viridis",
    ]
    if palette not in bokeh_palettes:
        # Palette is not known by bokeh:
        candidates = find_bokeh_palettes(categories)
    else:
        if categories not in bokeh_palettes[palette]:
            # Palette does not have the correct number of colors:
            candidates = find_bokeh_palettes(categories)
        else:
            # Palette exist and have the correct number of colors
            candidates = [palette]

    if len(candidates) == 1:
        name = candidates[0]
    else:
        name = None
        for key in preference:
            if key in candidates:
                name = key
                break
        if name is None:
            raise ValueError("Could not find a suitable bokeh palette!")
    length = categories
    for key in sorted(bokeh_palettes[name]):
        if categories <= key:
            length = key
            break
    name = name + "_" if name[-1].isdigit() else name
    return f"{name}{length}"


def get_color_mapping(
    values: ArrayLike | list[str] | list[int],
    palette: str = "Bright",
    max_categories: int = 20,
) -> Field:
    """Get a suitable color mapper for the given values.

    Args:
        values: Values to create a color mapping for.
        palette (str, optional): The preferred color palette.
        max_categories (int, optional): The max number of categories
            to consider.

    Returns:
        :obj:`Field`: The color mapper.
    """
    color_mapper = None
    categories = sorted(list(set(values)))
    categorical = len(categories) <= max_categories
    if is_numeric_dtype(np.array(categories)):
        try:
            color_mapper = linear_cmap(
                field_name="color_by",
                palette=palette,
                low=min(values),
                high=max(values),
            )
        except ValueError:
            color_mapper = linear_cmap(
                field_name="color_by",
                palette="Viridis256",
                low=min(values),
                high=max(values),
            )
    else:
        if categorical:
            palette = get_bokeh_palette(len(categories), palette=palette)
            color_mapper = factor_cmap("color_by", palette, categories)
        else:
            raise ValueError("Too many categories for the color mapping.")
    return color_mapper


def get_marker_mapping(
    values: ArrayLike | list[str],
    markers: list[str] = BOKEH_MARKERS,
) -> Field | None:
    """Set up a mapping for markers for the given values.

    Note, currently only arrays/lists as strings are supported.

    Args:
        values (ArrayLike | list[str]): The values to create markers for.
        markers (list[str]): A list of strings with the markers to use.

    Returns:
        :obj:`Field`: A mapper for markers.

    """
    categories = sorted(list(set(values)))
    categorical = len(categories) <= len(markers)
    if is_numeric_dtype(np.array(categories)):
        raise ValueError("Marker mapper expects a list of strings!")
    else:
        if categorical:
            return factor_mark(
                field_name="markers",
                markers=[
                    markers[i % len(markers)] for i in range(len(categories))
                ],
                factors=categories,
            )
    return None


def scatterplot(
    data: pd.DataFrame | None = None,
    x: str | np.ndarray | None = None,
    y: str | np.ndarray | None = None,
    color: str | ArrayLike | None = None,
    marker: str | ArrayLike | None = None,
    size: str | ArrayLike | None = None,
    title: str | None = None,
    fig: None | figure = None,
) -> None | figure:
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

    plot_data = {
        "x": xdata,
        "y": ydata,
    }
    extra_kw = {}

    if size is not None:
        if isinstance(size, str):
            if data is not None:
                plot_data["sizes"] = get_size_mapping(data[size])
            else:
                raise ValueError(f"Missing data for size = {size}")
        else:
            plot_data["sizes"] = get_size_mapping(size)
        extra_kw["size"] = "sizes"

    if marker is not None:
        if isinstance(marker, str):
            if data is not None:
                extra_kw["marker"] = get_marker_mapping(data[marker])
                plot_data["markers"] = data[marker]
            else:
                raise ValueError(f"Missing data for marker = {marker}")
        else:
            extra_kw["marker"] = get_marker_mapping(marker)
            plot_data["markers"] = marker

    source = ColumnDataSource(data=plot_data)
    if fig is None:
        fig = figure(
            title=title,
            active_scroll="wheel_zoom",
            background_fill_color="#fafafa",
        )
    fig.scatter(
        x="x",
        y="y",
        source=source,
        **extra_kw,
    )
    return fig
