"""Read frozen partitions and numerical targets without extracting the archive."""
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile


def reference_file(name, root=None):
    root = Path(root) if root is not None else Path(__file__).resolve().parent
    with ZipFile(root / 'reference.zip') as archive:
        return BytesIO(archive.read(name))
