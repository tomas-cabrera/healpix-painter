import argparse


class BasicParser(argparse.ArgumentParser):
    """
    A basic parser for command line arguments.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_argument(
            "-f",
            "--healpixfilename",
            type=str,
            help="HEALPix filename",
            default=None,
        )
        self.add_argument(
            "-e",
            "--lvkeventid",
            type=str,
            help="LVK event ID",
            default=None,
        )
