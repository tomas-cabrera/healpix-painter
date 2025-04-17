from ligo.skymap.io import read_sky_map


def parse_skymap_args(skymap_filename, lvk_eventname):
    if skymap_filename is None and lvk_eventname is None:
        raise ValueError("Either skymap_filename or lvk_eventname must be provided.")
    if skymap_filename is not None and lvk_eventname is not None:
        raise ValueError(
            "Only one of skymap_filename or lvk_eventname should be provided."
        )
    if skymap_filename is not None:
        return read_sky_map(skymap_filename)
    if lvk_eventname is not None:
        # TODO: download from GraceDB + flatten
        pass
