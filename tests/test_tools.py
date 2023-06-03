import responses

from moljourney.tools import download_file


def test_download_file(tmp_path, caplog):
    # Test that we can download a file:
    fake_url = "https://example.com/does-not-exists.txt"
    contents = "this is a fake file"
    with responses.RequestsMock() as mock:
        mock.add(
            responses.GET,
            fake_url,
            body=contents,
            status=200,
        )
        output_file = download_file(
            fake_url, tmp_path / "does-not-exists.txt"
        )
        assert output_file == tmp_path / "does-not-exists.txt"
        with open(tmp_path / "does-not-exists.txt") as output:
            assert output.read() == contents
        # The file should now exists, so let us try again:
        txt = f"File {output_file} exists - skipping download"
        assert txt not in caplog.text
        download_file(fake_url, tmp_path / "does-not-exists.txt")
        assert txt in caplog.text

    # Test that we can fail:
    with responses.RequestsMock() as mock:
        mock.add(
            responses.GET,
            fake_url,
            body=None,
            status=418,
        )
        txt2 = "Could not download file. Status code 418"
        assert txt2 not in caplog.text
        output_file = download_file(fake_url, tmp_path / "should-fail.txt")
        assert output_file is None
        assert txt2 in caplog.text
