from importlib.metadata import PackageNotFoundError, version

import inflectsynth


def test_public_version_matches_distribution_metadata():
    try:
        expected = version("inflectsynth")
    except PackageNotFoundError:
        expected = "0+unknown"
    assert inflectsynth.__version__ == expected
