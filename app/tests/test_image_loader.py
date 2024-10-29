from unittest.mock import Mock

import numpy as np

from app.images import IFindFiles, ImageLoader, ILoader


def test_load():
    mock_loader = Mock(ILoader)
    mock_loader.load.return_value = np.array([1, 2, 3])

    mock_files_finder = Mock(IFindFiles)
    mock_files_finder.find.return_value = ["path/to/image1", "path/to/image2"]

    image_loader = ImageLoader("path", mock_files_finder, mock_loader)
    image_loader.load()

    assert len(image_loader.images) == 2
    assert all((image == np.array([1, 2, 3])).all() for image in image_loader.images)