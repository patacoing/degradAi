from unittest.mock import Mock

from app.images import ImageLoader, ILoader


def test_load_should_return_images_and_filenames():
    mock_loader = Mock(ILoader)
    image_loader = ImageLoader("tests/unit/images", mock_loader)

    def is_file(file_path: str) -> bool:
        return True

    def list_files(path: str) -> list[str]:
        return ["image1.jpg", "image2.jpg", ".DS_Store"]

    images, filenames = image_loader.load(is_file=is_file, list_files=list_files)

    assert len(images) == 2
    assert mock_loader.load.call_count == 2
