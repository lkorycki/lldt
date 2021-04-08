import matplotlib.pyplot as plt
import io
import tensorflow as tf
import itertools
import numpy as np
import matplotlib.colors as mcolors


class PlotUtils:

    @staticmethod
    def create_image_grid(images, labels, cls_names):
        figure = plt.figure(figsize=(20, 20))
        rows, cols = (10, 10) if len(images) == 100 else (5, 5)

        for i in range(min(rows * cols, len(images))):
            cls_idx = labels[i].item()
            plt.subplot(rows, cols, i + 1, title=cls_names[cls_idx] if len(cls_names) > cls_idx else cls_idx)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(images[i])

        return figure

    @staticmethod
    def fig_to_image(figure):
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(figure)

        buf.seek(0)
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        image = tf.expand_dims(image, 0)

        return image

    @staticmethod
    def create_confusion_matrix(cm, class_names, title=None):
        figure = plt.figure(figsize=(8, 8))

        colors = plt.cm.Blues(np.linspace(0, 1, 128))
        cmap = mcolors.LinearSegmentedColormap.from_list('colormap', colors)

        plt.imshow(cm, interpolation='nearest', cmap=cmap if len(cm) > 1 else plt.cm.Blues_r)
        if title: plt.title(title, fontsize=24, pad=16)

        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45, fontsize=12)
        plt.yticks(tick_marks, class_names, fontsize=12)

        labels = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

        threshold = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            color = 'white' if cm[i, j] > threshold else 'black'
            plt.text(j, i, labels[i, j], horizontalalignment='center', color=color)

        plt.ylabel('True label', fontsize=16, labelpad=20)
        plt.xlabel('Predicted label', fontsize=16, labelpad=20)

        return figure
