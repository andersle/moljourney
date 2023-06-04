"""Extra helper tools"""
import logging
import pathlib

import requests

LOGGER = logging.getLogger(__name__)


USER_AGENT = "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:102.0) Gecko/20100101 Firefox/102.0"  # noqa: E501


def download_file(
    url: str, output_file: str | pathlib.Path
) -> str | pathlib.Path | None:
    """Download a file if the file names does not exist locally."""
    if pathlib.Path(output_file).is_file():
        LOGGER.info("File %s exists - skipping download", output_file)
        return output_file
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    response = session.get(url, allow_redirects=True)
    if response:
        with open(output_file, "w") as output:
            output.write(response.text)
        LOGGER.info("Downloaded file to: %s", output_file)
        return output_file
    else:
        LOGGER.info(
            "Could not download file. Status code %i", response.status_code
        )
        return None
