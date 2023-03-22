import logging
import os
import numpy as np
import warnings
from iopath.common.file_io import PathHandler
from detectron2.utils.file_io import PathManager
from zipfile import ZipFile
from detectron2.utils.logger import log_first_n
from io import BytesIO
import multiprocessing

__zip_file_pool__ = {
    # (path, mode): ZipFile
}


def find_zip_parent(path: str, mode: str = "r", is_dir=False):
    """Find the best match zipfile from the end"""
    if mode[-1] == "b":
        mode = mode[:-1]
    if is_dir:
        par_path = path
    else:
        par_path = os.path.dirname(path)
    visited = [par_path]
    # count = 0
    while par_path:
        # count += 1
        if ((par_path, mode) in __zip_file_pool__) and (
            __zip_file_pool__[(par_path, mode)].fp is not None
        ):
            # zip file is still open
            zip_file = __zip_file_pool__[(par_path, mode)]
            for path in visited[:-1]:
                __zip_file_pool__[(path, mode)] = zip_file
            return zip_file
        elif os.path.isfile(par_path + ".zip"):
            log_first_n(logging.INFO, "Open zip file {}.".format(par_path), n=1)
            zip_file = ZipFile(par_path + ".zip", mode=mode)
            for path in visited:
                __zip_file_pool__[(path, mode)] = zip_file
            # return par_path, zip_file, count
            return zip_file

        par_path = os.path.sep.join(par_path.split(os.path.sep)[:-1])
        visited.append(par_path)
    # return None, None, count
    return None


class ZipFileHandler(PathHandler):
    """
    Load data from zipfile and return a file-like object
    """

    PREFIX = "zip://"

    def _get_supported_prefixes(self):
        return [self.PREFIX]

    def _get_local_path(self, path, **kwargs):
        name = path[len(self.PREFIX) :]

        return name

    def _open(self, path: str, mode: str = "r", buffering=-1, **kwargs):
        """Open a file and return a file object.
        Args:
            path (str): _description_
            mode (str, optional): file open mode. Defaults to "r".

        Returns:
            ByteIO: file-like object
        """

        path = self._get_local_path(path)
        zip_file: ZipFile = find_zip_parent(path, mode)
        if zip_file is None:
            warnings.warn(
                "No zipfile contains {}, falling back to naive PathHandler".format(
                    path
                ),
            )
            return PathManager.open(path, mode, buffering, **kwargs)
        assert mode in [
            # "r",
            "rb",
        ], "Writing to ZipFile object is not thread safe. Only read mode is supported for now."  # Need to deal with write mode carefully, maybe we will change it later.
        filename = os.path.join(
            zip_file.filelist[0].filename,
            path[len(os.path.splitext(zip_file.filename)[0]) + 1 :],
        )
        log_first_n(
            logging.INFO,
            "[Example] load file {} from zip file {}.".format(
                filename, zip_file.filename
            ),
            n=1,
        )
        assert (
            buffering == -1
        ), f"{self.__class__.__name__} does not support the `buffering` argument"
        # Use zipfile.Path in Python 3.10

        if mode[-1] == "b":
            mode = mode[:-1]

        return BytesIO(
            zip_file.read(filename)
        )  # If any errors occur, check whether a ZipFile object is called by multiple threads/processes at the same time.

    def _ls(self, path: str, **kwargs):
        """
        List the contents of the directory at the provided URI.

        Args:
            path (str): A URI supported by this PathHandler

        Returns:
            List[str]: list of contents in given path
        """
        path = self._get_local_path(path)
        zip_file: ZipFile = find_zip_parent(path, is_dir=True)
        assert zip_file is not None, "No zipfile contains {}".format(path)
        file_names = zip_file.namelist()
        in_archive_path = os.path.join(
            file_names[0], path[len(os.path.splitext(zip_file.filename)[0]) + 1 :]
        ).rstrip(os.path.sep)
        file_names = [
            f[len(in_archive_path) + 1 :]
            for f in file_names
            if f.startswith(in_archive_path)
        ]
        # must be closed to avoid thread safety issues
        zip_file.close()
        return file_names


PathManager.register_handler(ZipFileHandler())
