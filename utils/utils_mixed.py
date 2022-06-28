import argparse
from turtle import width
import yaml
import importlib
from sklearn.metrics import confusion_matrix
from typing import Optional, Any, Union
import itertools
from matplotlib.colors import ListedColormap
import pandas as pd
import numpy as np
import torch
from matplotlib import pyplot as plt
import os
from PIL import Image
from pathlib import Path
Image.MAX_IMAGE_PIXELS = 100000000000
from skimage import io
from .constants import MASK_VALUE_TO_TEXT, MASK_VALUE_TO_COLOR


def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path",
                        type=str,
                        required=True)
    parser.add_argument("--config",
                        type=str,
                        required=True)
    args = parser.parse_args()

    assert Path(args.base_path).exists(
    ), f"Base path does not exist: {args.base_path}"
    assert Path(args.config).exists(
    ), f"Config path does not exist: {args.config}"
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return Path(args.base_path), config


def get_metadata(constants):
    preprocess_directory = constants.PREPROCESS_PATH
    superpixel_directory = preprocess_directory / "superpixels"
    tissue_mask_directory = preprocess_directory / "tissue_masks"
    graph_directory = preprocess_directory / \
        "graphs" / ("partial_" + str(constants.PARTIAL))

    image_metadata = pd.read_pickle(constants.IMAGES_DF)
    annotation_metadata = pd.read_pickle(constants.ANNOTATIONS_DF)

    all_metadata = merge_metadata(
        image_metadata=image_metadata,
        annotation_metadata=annotation_metadata,
        graph_directory=graph_directory,
        superpixel_directory=superpixel_directory,
        tissue_mask_directory=tissue_mask_directory,
        add_image_sizes=False,
    )
    labels_metadata = pd.read_pickle(constants.LABELS_DF)
    label_mapper = to_mapper(labels_metadata)  # 所有的数据
    return all_metadata, label_mapper


def merge_metadata(
    image_metadata: pd.DataFrame,
    annotation_metadata: pd.DataFrame,
    graph_directory: Optional[Path] = None,
    superpixel_directory: Optional[Path] = None,
    tissue_mask_directory: Optional[Path] = None,
    add_image_sizes: bool = False,
):
    '''
    合并'image_path', 'annotation_mask_path', 'graph_path', 'superpixel_path', 'tissue_mask_path', 'height', 'width'到一个DataFrame中
    '''
    # Join with image metadata
    image_metadata = image_metadata.join(annotation_metadata)

    if graph_directory is not None:
        graph_metadata = pd.DataFrame(
            [
                (path.name.split(".")[0], path)
                for path in filter(
                    lambda x: x.name.endswith(
                        ".bin"), graph_directory.iterdir()
                )
            ],
            columns=["name", "graph_path"],
        )
        graph_metadata = graph_metadata.set_index("name")
        image_metadata = image_metadata.join(graph_metadata)

    if superpixel_directory is not None:
        superpixel_metadata = pd.DataFrame(
            [
                (path.name.split(".")[0], path)
                for path in filter(
                    lambda x: x.name.endswith(
                        ".h5"), superpixel_directory.iterdir()
                )
            ],
            columns=["name", "superpixel_path"],
        )
        superpixel_metadata = superpixel_metadata.set_index("name")
        image_metadata = image_metadata.join(superpixel_metadata)

    if tissue_mask_directory is not None:
        tissue_metadata = pd.DataFrame(
            [
                (path.name.split(".")[0], path)
                for path in filter(
                    lambda x: x.name.endswith(
                        ".png"), tissue_mask_directory.iterdir()
                )
            ],
            columns=["name", "tissue_mask_path"],
        )
        tissue_metadata = tissue_metadata.set_index("name")
        image_metadata = image_metadata.join(tissue_metadata)

    # Add image sizes
    if add_image_sizes:
        image_heights, image_widths = list(), list()
        for name, row in image_metadata.iterrows():
            # image = Image.open(row.annotation_mask_path)
            image = io.MultiImage(
                row.annotation_mask_path.as_posix())[-2][:, :, 0]
            height, width = image.shape
            image_heights.append(height)
            image_widths.append(width)
        image_metadata["height"] = image_heights
        image_metadata["width"] = image_widths

    return image_metadata


def to_mapper(df):
    """[summary]

    Args:
        df ([type]): [记录数据label的dataframe]

    Returns:
        [dict]: {'xxx': array([0,0,1,1]),...}
    """
    mapper = dict()
    for name, row in df.iterrows():
        mapper[name] = np.array(
            [row["benign"], row["grade3"], row["grade4"], row["grade5"]]
        )
    return mapper


def create_directory(path):
    if not os.path.isdir(path):
        os.mkdir(path)


def dynamic_import_from(source_file: str, class_name: str) -> Any:
    """Do a from source_file import class_name dynamically

    Args:
        source_file (str): Where to import from
        class_name (str): What to import

    Returns:
        Any: The class to be imported
    """
    module = importlib.import_module(source_file)
    return getattr(module, class_name)


def read_image(image_path: str) -> np.ndarray:
    """Reads an image from a path and converts it into a numpy array
    Args:
        image_path (str): Path to the image
    Returns:
        np.array: A numpy array representation of the image
    """
    assert image_path.exists()
    try:
        with Image.open(image_path) as img:
            image = np.array(img)
    except OSError as e:
        raise OSError(e)
    return image


def fast_histogram(input_array: np.ndarray, nr_values: int) -> np.ndarray:
    """Calculates a histogram of a matrix of the values from 0 up to (excluding) nr_values

    Args:
        x (np.array): Input tensor
        nr_values (int): Possible values. From 0 up to (exclusing) nr_values.

    Returns:
        np.array: Output tensor
    """
    output_array = np.empty(nr_values, dtype=int)
    for i in range(nr_values):
        output_array[i] = (input_array == i).sum()
    return output_array


def fast_confusion_matrix(y_true: Union[np.ndarray, torch.Tensor], y_pred: Union[np.ndarray, torch.Tensor], nr_classes: int):
    """Faster computation of confusion matrix according to https://stackoverflow.com/a/59089379

    Args:
        y_true (Union[np.ndarray, torch.Tensor]): Ground truth (1D)
        y_pred (Union[np.ndarray, torch.Tensor]): Prediction (1D)
        nr_classes (int): Number of classes

    Returns:
        np.ndarray: Confusion matrix of shape nr_classes x nr_classes
    """
    assert y_true.shape == y_pred.shape
    y_true = torch.as_tensor(y_true, dtype=torch.long)
    y_pred = torch.as_tensor(y_pred, dtype=torch.long)
    y = nr_classes * y_true + y_pred
    y = torch.bincount(y)
    if len(y) < nr_classes * nr_classes:
        y = torch.cat(
            (y, torch.zeros(nr_classes * nr_classes - len(y), dtype=torch.long)))
    y = y.reshape(nr_classes, nr_classes)
    return y.numpy()


def get_batched_segmentation_maps(
    instance_logits, superpixels, instance_associations, NR_CLASSES
):
    batch_instance_predictions = instance_logits.argmax(
        axis=1).detach().cpu().numpy()
    segmentation_maps = np.empty((superpixels.shape), dtype=np.uint8)
    start = 0
    for i, end in enumerate(instance_associations):
        instance_predictions = batch_instance_predictions[start: start + end]
        segmentation_maps[i] = get_segmentation_map(
            instance_predictions, superpixels[i], NR_CLASSES
        )
        start += end
    return segmentation_maps


def get_segmentation_map(instance_predictions, superpixels, NR_CLASSES):
    all_maps = list()
    for label in range(NR_CLASSES):
        (spx_indices,) = np.where(instance_predictions == label)
        spx_indices = spx_indices + 1
        map_l = np.isin(superpixels, spx_indices) * label
        all_maps.append(map_l)
    return np.stack(all_maps).sum(axis=0)


def save_confusion_matrix(prediction, ground_truth, classes, save_path):
    cm = confusion_matrix(
        y_true=ground_truth, y_pred=prediction, labels=np.arange(len(classes))
    )
    fig = plot_confusion_matrix(cm, classes, title=None, normalize=False)
    # fig = plot_confusion_matrix(cm, classes, figname=None, normalize=False)
    fig.savefig(str(save_path), dpi=300, bbox_inches="tight")


def plot_confusion_matrix(
    cm, classes, normalize=False, title=None, cmap=plt.cm.Blues
):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    (This function is copied from the scikit docs.)
    """
    fig, ax = plt.subplots(figsize=(7, 7))
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, fontsize=18)
    plt.yticks(tick_marks, classes, fontsize=18)
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(
                j,
                i,
                "%.2f" % cm[i, j],
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=16,
            )
        else:
            plt.text(
                j,
                i,
                cm[i, j],
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=16,
            )
    plt.tight_layout()
    plt.ylabel('True label', fontsize=20)
    plt.xlabel('Predicted label', fontsize=20)
    ax.imshow(cm, interpolation="nearest", cmap=cmap)
    if title is not None:
        ax.set_title(title)
    return fig


def show_class_activation(per_class_output):
    fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        im = ax.imshow(per_class_output[i], vmin=0, vmax=1, cmap="viridis")
        ax.set_axis_off()
        ax.set_title(MASK_VALUE_TO_TEXT[i])

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    return fig


def show_segmentation_masks(output, annotation=None, label=None, **kwargs):
    height = 4
    width = 5
    ncols = 1
    if annotation is not None:
        width += 5
        ncols += 1
    fig, ax = plt.subplots(nrows=1, ncols=ncols, figsize=(width, height))
    cmap = ListedColormap(MASK_VALUE_TO_COLOR.values())

    mask_ax = ax if annotation is None else ax[0]
    im = mask_ax.imshow(output, cmap=cmap, vmin=-0.5,
                        vmax=4.5, interpolation="nearest")
    mask_ax.axis("off")
    if annotation is not None:
        ax[1].imshow(
            annotation, cmap=cmap, vmin=-0.5, vmax=4.5, interpolation="nearest"
        )
        ax[1].axis("off")
        ax[0].set_title("Prediction")
        ax[1].set_title(f"Ground Truth: {label}")

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.016, 0.7])
    cbar = fig.colorbar(im, ticks=[0, 1, 2, 3, 4], cax=cbar_ax)
    cbar.ax.set_yticklabels(MASK_VALUE_TO_TEXT.values())
    return fig
