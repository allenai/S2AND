import json
import logging
import os
import shutil
import tempfile
from hashlib import sha256
from typing import IO
from urllib.parse import urlparse

import requests  # type: ignore[import-untyped]

from s2and.consts import CACHE_ROOT

logger = logging.getLogger("s2and")

ARTIFACTS_CACHE = str(CACHE_ROOT / "artifacts")

# file largely taken from https://github.com/allenai/scispacy/blob/master/scispacy/file_cache.py


def cached_path(url_or_filename: str | os.PathLike[str], cache_dir: str | None = None) -> str:
    """
    Given something that might be a URL (or might be a local path),
    determine which. If it's a URL, download the file and cache it, and
    return the path to the cached file. If it's already a local path,
    make sure the file exists and then return the path.
    """
    if cache_dir is None:
        cache_dir = ARTIFACTS_CACHE
    if not isinstance(url_or_filename, str):
        url_or_filename = os.fspath(url_or_filename)

    parsed = urlparse(url_or_filename)

    if parsed.scheme in ("http", "https"):
        # URL, so get it from the cache (downloading if necessary)
        return get_from_cache(url_or_filename, cache_dir)
    elif os.path.exists(url_or_filename):
        # File, and it exists.
        return url_or_filename
    elif parsed.scheme == "":
        # File, but it doesn't exist.
        raise FileNotFoundError(f"file {url_or_filename} not found")
    else:
        # Something unknown
        raise ValueError(f"unable to parse {url_or_filename} as a URL or as a local path")


def url_to_filename(url: str, etag: str | None = None) -> str:
    """
    Convert `url` into a hashed filename in a repeatable way.
    If `etag` is specified, append its hash to the url's, delimited
    by a period.
    """

    last_part = url.split("/")[-1]
    url_bytes = url.encode("utf-8")
    url_hash = sha256(url_bytes)
    filename = url_hash.hexdigest()

    if etag:
        etag_bytes = etag.encode("utf-8")
        etag_hash = sha256(etag_bytes)
        filename += "." + etag_hash.hexdigest()

    filename += "." + last_part
    return filename


def http_get(url: str, temp_file: IO) -> None:
    req = requests.get(url, stream=True)
    for chunk in req.iter_content(chunk_size=1024):
        if chunk:  # filter out keep-alive new chunks
            temp_file.write(chunk)


def get_from_cache(url: str, cache_dir: str | None = None) -> str:
    """
    Given a URL, look for the corresponding dataset in the local cache.
    If it's not there, download it. Then return the path to the cached file.
    """
    if cache_dir is None:
        cache_dir = ARTIFACTS_CACHE

    os.makedirs(cache_dir, exist_ok=True)

    response = requests.head(url, allow_redirects=True)
    if response.status_code != 200:
        raise OSError(f"HEAD request failed for url {url} with status code {response.status_code}")
    etag = response.headers.get("ETag")

    filename = url_to_filename(url, etag)

    # get cache path to put the file
    cache_path = os.path.join(cache_dir, filename)

    if not os.path.exists(cache_path):
        # Download to temporary file, then copy to cache dir once finished.
        # Otherwise you get corrupt cache entries if the download gets interrupted.
        with tempfile.NamedTemporaryFile() as temp_file:  # type: IO
            logger.info("%s not found in cache; downloading to %s", url, temp_file.name)

            # GET file object
            http_get(url, temp_file)

            # we are copying the file before closing it, so flush to avoid truncation
            temp_file.flush()
            # shutil.copyfileobj() starts at the current position, so go to the start
            temp_file.seek(0)

            logger.info("Finished download; copying %s to cache at %s", temp_file.name, cache_path)
            with open(cache_path, "wb") as cache_file:
                shutil.copyfileobj(temp_file, cache_file)

            meta = {"url": url, "etag": etag}
            meta_path = cache_path + ".json"
            with open(meta_path, "w", encoding="utf-8") as meta_file:
                json.dump(meta, meta_file)

    return cache_path
