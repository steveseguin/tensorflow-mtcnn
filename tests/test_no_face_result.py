"""Control-flow regressions, without TensorFlow models or OpenCV inference.

Run with: python -m unittest discover -s tests -v
NumPy and six are required, as they are by the detector itself. The real detector
and CLI modules run against synthetic network outputs and inert image I/O.
"""
import contextlib
import importlib.util
import io
from pathlib import Path
import types
import unittest
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path, replacements):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict('sys.modules', replacements):
        spec.loader.exec_module(module)
    return module


class NoFaceResultTests(unittest.TestCase):
    def setUp(self):
        self.image = np.zeros((40, 40, 3), dtype=np.uint8)
        self.cv2 = types.SimpleNamespace(
            INTER_AREA=3,
            COLOR_BGR2RGB=4,
            resize=lambda image, size, interpolation: np.zeros(
                (size[1], size[0], 3), dtype=np.float32),
            imread=mock.Mock(side_effect=lambda filename: self.image.copy()),
            cvtColor=lambda image, mode: image[:, :, ::-1].copy(),
            imwrite=mock.Mock(return_value=True),
            rectangle=mock.Mock(),
            circle=mock.Mock(),
        )
        self.tf = types.SimpleNamespace(Session=mock.Mock(return_value=object()))
        self.detector = load_module(
            'detector_under_test', ROOT / 'detect_face.py',
            {'tensorflow': self.tf, 'cv2': self.cv2})

    def networks(self, proposal=True, refinement=True, output=True):
        pnet = mock.Mock(side_effect=lambda images: (
            np.zeros((1, 1, 1, 4)),
            np.array([[[[0.0, 0.99 if proposal else 0.0]]]])))

        def rnet_output(images):
            count = images.shape[0]
            scores = np.zeros((count, 2))
            scores[:, 1] = 0.99 if refinement else 0.0
            return np.zeros((count, 4)), scores

        def onet_output(images):
            count = images.shape[0]
            scores = np.zeros((count, 2))
            scores[:, 1] = 0.99 if output else 0.0
            return np.zeros((count, 4)), np.full((count, 10), 0.5), scores

        return pnet, mock.Mock(side_effect=rnet_output), mock.Mock(side_effect=onet_output)

    def detect(self, image, networks):
        return self.detector.detect_face(
            image, 40, *networks, [0.6, 0.7, 0.9], 0.709)

    def assert_empty_landmarks(self, boxes, points):
        self.assertEqual(boxes.shape[0], 0)
        self.assertIsInstance(points, np.ndarray)
        self.assertEqual(points.shape, (10, 0))
        self.assertEqual(points.T.shape, (0, 10))

    def test_image_smaller_than_minimum_face_size(self):
        networks = self.networks()
        boxes, points = self.detect(np.zeros((20, 20, 3)), networks)
        self.assert_empty_landmarks(boxes, points)
        for network in networks:
            network.assert_not_called()

    def test_no_pnet_proposals(self):
        networks = self.networks(proposal=False)
        boxes, points = self.detect(self.image, networks)
        self.assert_empty_landmarks(boxes, points)
        networks[0].assert_called_once()
        networks[1].assert_not_called()
        networks[2].assert_not_called()

    def test_all_proposals_rejected_by_rnet(self):
        networks = self.networks(refinement=False)
        boxes, points = self.detect(self.image, networks)
        self.assert_empty_landmarks(boxes, points)
        networks[1].assert_called_once()
        networks[2].assert_not_called()

    def test_all_candidates_rejected_by_onet(self):
        networks = self.networks(output=False)
        boxes, points = self.detect(self.image, networks)
        self.assert_empty_landmarks(boxes, points)
        networks[2].assert_called_once()

    def test_successful_detection_preserves_box_and_landmarks(self):
        boxes, points = self.detect(self.image, self.networks())
        self.assertEqual(boxes.shape, (1, 5))
        self.assertEqual(points.shape, (10, 1))
        np.testing.assert_allclose(boxes[0], [3, 3, 40, 40, 0.99])
        np.testing.assert_allclose(points, np.full((10, 1), 21.0))

    def run_empty_cli(self, image, networks):
        self.image = image
        cli = load_module(
            'cli_under_test', ROOT / 'facedetect_mtcnn.py',
            {'tensorflow': self.tf, 'cv2': self.cv2,
             'detect_face': self.detector})
        stdout = io.StringIO()
        with mock.patch.object(self.detector, 'create_mtcnn', return_value=networks):
            with contextlib.redirect_stdout(stdout):
                cli.main(types.SimpleNamespace(input='input.jpg', output='output.jpg'))
        self.cv2.imwrite.assert_called_once()
        filename, written = self.cv2.imwrite.call_args[0]
        self.assertEqual(filename, 'output.jpg')
        np.testing.assert_array_equal(written, image)
        self.cv2.rectangle.assert_not_called()
        self.cv2.circle.assert_not_called()
        self.assertIn('Total 0 face(s) detected, saved in output.jpg', stdout.getvalue())

    def test_cli_saves_tiny_image_with_zero_faces(self):
        self.run_empty_cli(np.zeros((20, 20, 3), dtype=np.uint8), self.networks())

    def test_cli_saves_image_with_no_proposals(self):
        self.run_empty_cli(self.image, self.networks(proposal=False))

    def test_cli_saves_image_with_all_proposals_rejected(self):
        self.run_empty_cli(self.image, self.networks(refinement=False))


if __name__ == '__main__':
    unittest.main()
