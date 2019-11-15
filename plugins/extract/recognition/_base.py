#!/usr/bin/env python3
""" Base class for Face Recognition plugins

Plugins should inherit from this class

See the override methods for which methods are required.

The plugin will receive a dict containing:

>>> {"filename": <filename of source frame>,
>>>  "image": <source image>,
>>>  "detected_faces": <list of DetectedFace classes from lib/faces_detect>}

For each source item, the plugin must pass a dict to finalize containing:

>>> {"filename": <filename of source frame>,
>>>  "image": <four channel source image>,
>>>  "detected_faces": <list of DetectedFace classes from lib/faces_detect>}
"""

import cv2
import numpy as np
from fastcluster import linkage

from plugins.extract._base import Extractor, logger


class Recognizer(Extractor):  # pylint:disable=abstract-method
    """ Recognizer plugin _base Object

    All Recognizer plugins must inherit from this class

    Parameters
    ----------
    git_model_id: int
        The second digit in the github tag that identifies this model. See
        https://github.com/deepfakes-models/faceswap-models for more information
    model_filename: str
        The name of the model file to be loaded
    image_is_aligned: bool, optional
        Indicates that the passed in image is an aligned face rather than a frame.
        Default: ``False``

    Other Parameters
    ----------------
    configfile: str, optional
        Path to a custom configuration ``ini`` file. Default: Use system configfile

    See Also
    --------
    plugins.extract.align : Aligner plugins
    plugins.extract._base : Parent class for all extraction plugins
    plugins.extract.detect._base : Detector parent class for extraction plugins.
    plugins.extract.align._base : Aligner parent class for extraction plugins.
    plugins.extract.mask._base : Masker parent class for extraction plugins.
    """

    def __init__(self, git_model_id=None, model_filename=None, configfile=None,
                 image_is_aligned=False):
        logger.debug("Initializing %s: (configfile: %s, )", self.__class__.__name__, configfile)
        super().__init__(git_model_id,
                         model_filename,
                         configfile=configfile)
        self.input_size = 224  # Override for model specific input_size
        self.coverage_ratio = 1.0  # Override for model specific coverage_ratio

        self._plugin_type = "recognizer"
        self._image_is_aligned = image_is_aligned
        self._faces_per_filename = dict()  # Tracking for recompiling face batches
        self._rollover = []  # Items that are rolled over from the previous batch in get_batch
        self._output_faces = []
        logger.debug("Initialized %s", self.__class__.__name__)

    def get_batch(self, queue):
        """ Get items for inputting into the masker from the queue in batches

        Items are returned from the ``queue`` in batches of
        :attr:`~plugins.extract._base.Extractor.batchsize`

        To ensure consistent batch sizes for masker the items are split into separate items for
        each :class:`lib.faces_detect.DetectedFace` object.

        Remember to put ``'EOF'`` to the out queue after processing
        the final batch

        Outputs items in the following format. All lists are of length
        :attr:`~plugins.extract._base.Extractor.batchsize`:

        >>> {'filename': [<filenames of source frames>],
        >>>  'image': [<source images>],
        >>>  'detected_faces': [[<lib.faces_detect.DetectedFace objects]]}

        Parameters
        ----------
        queue : queue.Queue()
            The ``queue`` that the plugin will be fed from.

        Returns
        -------
        exhausted, bool
            ``True`` if queue is exhausted, ``False`` if not
        batch, dict
            A dictionary of lists of :attr:`~plugins.extract._base.Extractor.batchsize`:
        """
        exhausted = False
        batch = dict()
        idx = 0
        while idx < self.batchsize:
            item = self._collect_item(queue)
            if item == "EOF":
                logger.trace("EOF received")
                exhausted = True
                break
            # Put frames with no faces into the out queue to keep TQDM consistent
            if not item["detected_faces"]:
                self._queues["out"].put(item)
                continue
            for f_idx, face in enumerate(item["detected_faces"]):
                face.image = self._convert_color(item["image"])
                face.load_feed_face(face.image,
                                    size=self.input_size,
                                    coverage_ratio=1.0,
                                    dtype="float32",
                                    is_aligned_face=self._image_is_aligned)
                batch.setdefault("detected_faces", []).append(face)
                batch.setdefault("filename", []).append(item["filename"])
                batch.setdefault("image", []).append(item["image"])
                idx += 1
                if idx == self.batchsize:
                    frame_faces = len(item["detected_faces"])
                    if f_idx + 1 != frame_faces:
                        self._rollover = {k: v[f_idx + 1:] if k == "detected_faces" else v
                                          for k, v in item.items()}
                        logger.trace("Rolled over %s faces of %s to next batch for '%s'",
                                     len(self._rollover["detected_faces"]),
                                     frame_faces, item["filename"])
                    break
        if batch:
            logger.trace("Returning batch: %s", {k: v.shape if isinstance(v, np.ndarray) else v
                                                 for k, v in batch.items()})
        else:
            logger.trace(item)
        return exhausted, batch

    def _collect_item(self, queue):
        """ Collect the item from the _rollover dict or from the queue
            Add face count per frame to self._faces_per_filename for joining
            batches back up in finalize """
        if self._rollover:
            logger.trace("Getting from _rollover: (filename: `%s`, faces: %s)",
                         self._rollover["filename"], len(self._rollover["detected_faces"]))
            item = self._rollover
            self._rollover = dict()
        else:
            item = self._get_item(queue)
            if item != "EOF":
                logger.trace("Getting from queue: (filename: %s, faces: %s)",
                             item["filename"], len(item["detected_faces"]))
                self._faces_per_filename[item["filename"]] = len(item["detected_faces"])
        return item

    def _predict(self, batch):
        """ Just return the recognizer's predict function """
        return self.predict(batch)

    def finalize(self, batch):
        """ Finalize the output from Masker

        This should be called as the final task of each `plugin`.

        It strips unneeded items from the :attr:`batch` ``dict`` and pairs the detected faces back
        up with their original frame before yielding each frame.

        Outputs items in the format:

        >>> {'image': [<original frame>],
        >>>  'filename': [<frame filename>),
        >>>  'detected_faces': [<lib.faces_detect.DetectedFace objects>]}

        Parameters
        ----------
        batch : dict
            The final ``dict`` from the `plugin` process. It must contain the `keys`:
            ``detected_faces``, ``filename``, ``image``

        Yields
        ------
        dict
            A ``dict`` for each frame containing the ``image``, ``filename`` and list of
            :class:`lib.faces_detect.DetectedFace` objects.

        """
        for mask, face in zip(batch["prediction"], batch["detected_faces"]):
            face.add_mask(self._storage_name,
                          mask,
                          face.feed_matrix,
                          face.feed_interpolators[1],
                          storage_size=self._storage_size)
            face.feed = dict()

        self._remove_invalid_keys(batch, ("detected_faces", "filename", "image"))
        logger.trace("Item out: %s", {key: val
                                      for key, val in batch.items()
                                      if key != "image"})
        for filename, image, face in zip(batch["filename"],
                                         batch["image"],
                                         batch["detected_faces"]):
            self._output_faces.append(face)
            if len(self._output_faces) != self._faces_per_filename[filename]:
                continue
            retval = dict(filename=filename, image=image, detected_faces=self._output_faces)

            self._output_faces = []
            logger.trace("Yielding: (filename: '%s', image: %s, detected_faces: %s)",
                         retval["filename"], retval["image"].shape, len(retval["detected_faces"]))
            yield retval

    # <<< PROTECTED ACCESS METHODS >>> #
    @staticmethod
    def find_cosine_similiarity(source_face, test_face, subtract_mean=False):
        """ Find the cosine similarity between a source face and a test face """
        # mean = np.mean(np.concatenate([embeddings1[train_set], embeddings2[train_set]]), axis=0)
        # dist = distance(embeddings1-mean, embeddings2-mean, distance_metric)
        var_a = np.matmul(np.transpose(source_face), test_face)
        var_b = np.sum(np.multiply(source_face, source_face))
        var_c = np.sum(np.multiply(test_face, test_face))
        cosine_similiarity = 1 - (var_a / (np.sqrt(var_b) * np.sqrt(var_c)))
        '''
        # Distance based on cosine similarity
        dot = np.sum(np.multiply(embeddings1, embeddings2), axis=1)
        norm = np.linalg.norm(embeddings1, axis=1) * np.linalg.norm(embeddings2, axis=1)
        similarity = dot / norm
        dist = np.arccos(similarity) / math.pi
        '''
        return cosine_similiarity

    def sorted_similarity(self, predictions, method="ward"):
        """ Sort a matrix of predictions by similarity Adapted from:
            https://gmarti.gitlab.io/ml/2017/09/07/how-to-sort-distance-matrix.html
        input:
            - predictions is a stacked matrix of vgg_face predictions shape: (x, 4096)
            - method = ["ward","single","average","complete"]
        output:
            - result_order is a list of indices with the order implied by the hierarhical tree

        sorted_similarity transforms a distance matrix into a sorted distance matrix according to
        the order implied by the hierarchical tree (dendrogram)
        """
        logger.info("Sorting face distances. Depending on your dataset this may take some time...")
        num_predictions = predictions.shape[0]
        result_linkage = linkage(predictions, method=method, preserve_input=False)
        result_order = self.seriation(result_linkage, num_predictions, num_predictions * 2 - 2)
        return result_order

    def seriation(self, tree, points, current_index):
        """ Seriation method for sorted similarity
            input:
                - tree is a hierarchical tree (dendrogram)
                - points is the number of points given to the clustering process
                - current_index is the position in the tree for the recursive traversal
            output:
                - order implied by the hierarchical tree

            seriation computes the order implied by a hierarchical tree (dendrogram)
        """
        if current_index < points:
            return [current_index]
        left = int(tree[current_index-points, 0])
        right = int(tree[current_index-points, 1])
        return self.seriation(tree, points, left) + self.seriation(tree, points, right)