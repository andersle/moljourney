import numpy as np
import pandas as pd
from moljourney.plotting.bokehplotting import (
    get_size_mapping,
    find_bokeh_palettes,
    get_bokeh_palette,
    get_color_mapping,
    get_marker_mapping,
    scatterplot,
)
from bokeh.models.mappers import LinearColorMapper, CategoricalColorMapper, CategoricalMarkerMapper
from bokeh.plotting import figure
import pytest


def test_get_size_mapping():
    # Test that we understand different types of numbers:
    test1 = [1, 2, 3, 1, 2, 3, 4, 1, 1]
    test2 = np.array([1, 2, 3, 1, 2, 3, 4, 1, 1])
    test3 = pd.Series([1, 2, 3, 1, 2, 3, 4, 1, 1])
    test4 = pd.DataFrame(
        [1, 2, 3, 1, 2, 3, 4, 1, 1],
        columns=[
            "data",
        ],
    )
    correct = [6, 9, 12, 6, 9, 12, 15, 6, 6]
    for test in (test1, test2, test3, test4["data"]):
        sizes = get_size_mapping(test)
        assert sizes == correct

    # Test that we understand many numbers:
    test = range(1000)
    sizes = get_size_mapping(test, set_sizes=[10, 18])
    assert sizes[0] == 10
    assert sizes[-1] == 18

    # Test that we understand letters:
    test = ["a", "b", "a", "b", "c"]
    sizes = get_size_mapping(test, set_sizes=[1, 2, 3, 4, 5])
    assert sizes == [1, 2, 1, 2, 3]

    # Test that we understand letters, but we have too many categories
    test = ["a", "b", "c", "a", "d"]
    sizes = get_size_mapping(test, set_sizes=[0, 3,])
    assert set(sizes) == {0.0, 1.0, 2.0, 3.0}

    # print(test)
    # print(sizes)

    # Test that we understand a mixture of text and numbers:
    test = ["frog", "cat", "dog", "apple", 1, "a", "b", "c", "d", "frog"]
    sizes = get_size_mapping(test, set_sizes=[2, 4, 6, 8])
    assert min(sizes) == 2
    assert max(sizes) == 8


def test_find_bokeh_palettes():
    palettes = find_bokeh_palettes(20)
    correct = [
        "Purples",
        "Blues",
        "Greens",
        "Oranges",
        "Reds",
        "Greys",
        "Category20",
        "Category20b",
        "Category20c",
        "Iridescent",
        "TolRainbow",
        "Magma",
        "Inferno",
        "Plasma",
        "Viridis",
        "Cividis",
        "Turbo",
    ]
    assert len(palettes) == len(correct)
    for i in palettes:
        assert i in correct
    palettes = find_bokeh_palettes(256)
    assert "Viridis" in palettes


def test_get_bokeh_palette():
    palette = get_bokeh_palette(5)
    assert palette == "Bright5"
    palette = get_bokeh_palette(5, palette="Bokeh")
    assert palette == "Bokeh5"
    palette = get_bokeh_palette(7, palette="Colorblind")
    assert palette == "Colorblind7"
    palette = get_bokeh_palette(16, palette="Colorblind")
    assert palette == "Category20_16"
    palette = get_bokeh_palette(11, palette="Should-Not-Exist")
    assert palette == "Category20_11"
    with pytest.raises(ValueError):
        palette = get_bokeh_palette(257)

def test_get_color_mapping():
    values = [1, 2, 3, 4, 1, 2, 3, 4]
    mapping = get_color_mapping(values)
    assert isinstance(mapping.transform, LinearColorMapper)

    values = list(range(1000))
    mapping = get_color_mapping(values)
    assert isinstance(mapping.transform, LinearColorMapper)

    values = ["a", "a", "b", "c"]
    mapping = get_color_mapping(values)
    assert isinstance(mapping.transform, CategoricalColorMapper)

    values = ["a", "a", "b", "c", "d"]
    with pytest.raises(ValueError):
        mapping = get_color_mapping(values, max_categories=2)

def test_marker_mapping():
    values = ["a", "b", "z", "a"]
    markers = get_marker_mapping(values)
    if markers is not None:
        assert isinstance(markers.transform, CategoricalMarkerMapper)

    # Test too few markers:
    values = ["a", "b", "c", "d"] * 10
    markers = get_marker_mapping(values, markers=["circle", "square"])
    assert markers is None

    values = np.array([1, 2, 3, 4, 5])
    with pytest.raises(ValueError):
        markers = get_marker_mapping(values)


def test_scatter_plot():
    # Test nothing:
    fig = scatterplot()
    assert fig is None
    # Test only x:
    fig = scatterplot(x=np.arange(0, 10))
    assert isinstance(fig, figure)
    # Test only y:
    fig = scatterplot(y=np.arange(0, 10))
    assert isinstance(fig, figure)
    # Test that we can give a DataFrame:
    data = pd.DataFrame(
        np.random.normal(size=(10, 5)),
        columns=["a", "b", "c", "d", "e"]
    )
    fig = scatterplot(data=data, x="a")
    assert isinstance(fig, figure)
    fig = scatterplot(data=data, y="a")
    assert isinstance(fig, figure)
    # Missing field:
    with pytest.raises(KeyError):
        scatterplot(data=data, x="missing!")
    # Setting the size:
    fig = scatterplot(data=data, x="a", y="b", size="c")
    assert isinstance(fig, figure)
    fig = scatterplot(data=data, x="a", size=np.arange(10))
    assert isinstance(fig, figure)
    with pytest.raises(KeyError):
        scatterplot(data=data, x="a", size="missing!")
    with pytest.raises(ValueError):
        scatterplot(x=[1, 2, 3], size="missing!")
    with pytest.raises(KeyError):
        scatterplot(data=data, x="a", marker="missing!")
    with pytest.raises(ValueError):
        scatterplot(x=[1, 2, 3], marker="missing!")
    with pytest.raises(KeyError):
        scatterplot(data=data, x="a", marker="missing!")
    # Setting the marker:
    data["m"] = ["a", "b",] * 5
    fig = scatterplot(data=data, x="a", y="b", marker="m")
    assert isinstance(fig, figure)
    fig = scatterplot(data=data, x="a", y="b", marker=["frog", "cat"]*5)
    assert isinstance(fig, figure)

if __name__ == "__main__":
    #test_get_size_mapping()
    #test_find_bokeh_palettes()
    #test_get_bokeh_palette()
    #test_get_color_mapping()
    #test_marker_mapping()
    test_scatter_plot()
    # Setting the marker:
    data["m"] = ["a", "b",] * 5
    fig = scatterplot(data=data, x="a", y="b", marker="m")

if __name__ == "__main__":
    #test_get_size_mapping()
    #test_find_bokeh_palettes()
    #test_get_bokeh_palette()
    #test_get_color_mapping()
    #test_marker_mapping()
    test_scatter_plot()
