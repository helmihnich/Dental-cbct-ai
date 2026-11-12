import dentalcbct
from dentalcbct.cli import main


def test_version():
    assert dentalcbct.__version__ == "0.1.0"


def test_cli_runs():
    assert main([]) == 0
