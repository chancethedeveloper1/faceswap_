#!/usr/bin python3
""" Face Filterer for extraction in faceswap.py """

import logging

from lib.image import read_image_batch

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def avg(arr):
    """ Return an average """
    return sum(arr) * 1.0 / len(arr)


class FaceFilter():
    """ Face filter for extraction
        NB: we take only first face, so the reference file should only contain one face. """

    def __init__(self, reference_files, nreference_files, extractor, threshold=0.4):
        logger.debug("Initializing %s: (reference_file_paths: %s, nreference_file_paths: %s, "
                     "threshold: %s)",
                     self.__class__.__name__, reference_files, nreference_files, threshold)
        self.extractor = extractor
        self.recognizer = extractor._recognition
        self.filters = self.filter_encodings(reference_files, nreference_files)

        self.threshold = threshold
        logger.debug("Initialized %s", self.__class__.__name__)


    def filter_encodings(self, reference_files, nreference_files):
        """ Load the images """
        ref_images = read_image_batch(reference_files)
        nref_images = read_image_batch(nreference_files)
        ref_dictionary = {filename: {'image': img, 'type': 'filter'}
                          for filename, img in zip(reference_files, ref_images)}
        nref_dictionary = {filename: {'image': img, 'type': 'nfilter'}
                           for filename, img in zip(nreference_files, nref_images)}

        reference_dict = {**ref_dictionary, **nref_dictionary}
        logger.debug("Loaded filter images: %s", {k: v["type"] for k, v in reference_dict.items()})
        
        just_first_two_phases = range(min(2, self.extractor.passes))
        for _ in just_first_two_phases:
            self.queue_images(reference_dict)
            # self.extractor.launch()
            for faces in self.extractor.detected_faces():
                if faces["filename"] in reference_dict.keys():
                    filename = faces["filename"]
                    detected_faces = faces["detected_faces"]
                    if len(detected_faces) > 1:
                        logger.warning("Multiple faces found in %s file: '%s'. Using first "
                                       "detected face", reference_dict[filename]["type"], filename)
                    reference_dict[filename]["detected_face"] = detected_faces[0]

        for filename, face in reference_dict.items():
            logger.debug("Loading feed face: '%s'", filename)
            face["detected_face"].load_feed_face(face["image"],
                                                 size=self.recognizer.input_size,
                                                 coverage_ratio=1.0)
            input_batch = self.recognizer.process_input(face["detected_face"].feed_face[..., :3])
            face["encoding"] = self.recognizer.predict(input_batch)
            logger.debug("Feed face encoded: '%s'", filename)

        return reference_dict

    def queue_images(self, reference_dict):
        """ queue images for detection and alignment """
        in_queue = self.extractor.input_queue
        for fname, img in reference_dict.items():
            logger.debug("Adding to filter queue: '%s' (%s)", fname, img["type"])
            feed_dict = dict(filename=fname, image=img["image"])
            if img.get("detected_faces", None):
                feed_dict["detected_faces"] = img["detected_faces"]
            logger.debug("Queueing filename: '%s' items: %s", fname, list(feed_dict.keys()))
            in_queue.put(feed_dict)
        logger.debug("Sending EOF to filter queue")
        in_queue.put("EOF")

    def check(self, query_face):
        """ Check the extracted Face """
        logger.trace("Checking face with FaceFilter")
        distances = {"filter": list(), "nfilter": list()}
        input_batch = self.recognizer.process_input(query_face)
        query_encoding = self.recognizer.predict(input_batch)
        for filt in self.filters.values():
            similarity = self.recognizer.find_cosine_similiarity(filt["encoding"], query_encoding)
            distances[filt["type"]].append(similarity)

        avgs = {key: avg(val) if val else None for key, val in distances.items()}
        mins = {key: min(val) if val else None for key, val in distances.items()}
        # Filter
        if distances["filter"] and avgs["filter"] > self.threshold:
            msg = "Rejecting filter face: {} > {}".format(round(avgs["filter"], 2), self.threshold)
            retval = False
        # nFilter no Filter
        elif not distances["filter"] and avgs["nfilter"] < self.threshold:
            msg = "Rejecting nFilter face: {} < {}".format(round(avgs["nfilter"], 2),
                                                           self.threshold)
            retval = False
        # Filter with nFilter
        elif distances["filter"] and distances["nfilter"] and mins["filter"] > mins["nfilter"]:
            msg = ("Rejecting face as distance from nfilter sample is smaller: (filter: {}, "
                   "nfilter: {})".format(round(mins["filter"], 2), round(mins["nfilter"], 2)))
            retval = False
        elif distances["filter"] and distances["nfilter"] and avgs["filter"] > avgs["nfilter"]:
            msg = ("Rejecting face as average distance from nfilter sample is smaller: (filter: "
                   "{}, nfilter: {})".format(round(mins["filter"], 2), round(mins["nfilter"], 2)))
            retval = False
        elif distances["filter"] and distances["nfilter"]:
            # k-nn classifier
            var_k = min(5, min(len(distances["filter"]), len(distances["nfilter"])) + 1)
            var_n = sum(list(map(lambda x: x[0],
                                 list(sorted([(1, d) for d in distances["filter"]] +
                                             [(0, d) for d in distances["nfilter"]],
                                             key=lambda x: x[1]))[:var_k])))
            ratio = var_n/var_k
            if ratio < 0.5:
                msg = ("Rejecting face as k-nearest neighbors classification is less than "
                       "0.5: {}".format(round(ratio, 2)))
                retval = False
            else:
                msg = None
                retval = True
        else:
            msg = None
            retval = True
        if msg:
            logger.verbose(msg)
        else:
            logger.trace("Accepted face: (similarity: %s, threshold: %s)",
                         distances, self.threshold)
        return retval
