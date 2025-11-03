import typer
from typing_extensions import Annotated

from healpix_painter.basicpainter import basic_painter
from healpix_painter import footprints


def healpix_painter(
    routine: Annotated[
        str,
        typer.Option(
            help="The healpix_painter routine to use; 'basic_painter' by default"
        ),
    ] = "basic_painter",
    skymap_filename: Annotated[
        str,
        typer.Option(help="The path to the HEALPix skymap to tile"),
    ] = None,
    lvk_eventname: Annotated[
        str,
        typer.Option(help="The LVK event id to tile"),
    ] = None,
    footprint: Annotated[
        str,
        typer.Option(
            help="The telescope footprint to use for tiling; DECamConvexHull by default"
        ),
    ] = "DECamConvexHull",
    tiling_force_update: Annotated[
        bool,
        typer.Option(help="Whether to force update the tiling cache; False by default"),
    ] = False,
    scoring: Annotated[
        str,
        typer.Option(
            help="The scoring algorithm to use to rank pointings; 'probadd' by default"
        ),
    ] = "probadd",
    output_dir: Annotated[
        str,
        typer.Option(
            help="The output directory to save results; if not provided, uses the skymap directory"
        ),
    ] = None,
):
    """Function governing CLI interface.

    Parameters
    ----------
    routine : str, optional
        The healpix_painter routine to use; 'basic_painter' by default.
    skymap_filename : str, optional
        The path to the HEALPix skymap to tile.
        Either skymap_filename or lvk_eventname must be provided.
    lvk_eventname : str, optional
        The LVK event id to tile
        Either skymap_filename or lvk_eventname must be provided.
    footprint : str, optional
        The telescope footprint to use for tiling; 'DECamConvexHull' by default.
    tiling_force_update : bool, optional
        Whether to force update the tiling cache; False by default.
    scoring : str, optional
        The scoring algorithm to use to rank pointings; 'probadd' by default.
        Possible options are:
        - 'probadd': Score by total probability added by each pointing, ignoring previously covered pixels.
        - 'probden_probadd': Score by maximum probability density in each pointing, breaking ties by total probability added.
    output_dir : str, optional
        The output directory to save results; if not provided, uses the directory the skymap is in.

    Raises
    ------
    ValueError
        If the routine specified is not implemented.
    """
    # Select routine
    if routine == "basic_painter":
        basic_painter(
            skymap_filename=skymap_filename,
            lvk_eventname=lvk_eventname,
            footprint=getattr(footprints, footprint),
            tiling_force_update=tiling_force_update,
            scoring=scoring,
            output_dir=output_dir,
        )
    else:
        raise ValueError(f"Routine '{routine}' not implemented.")


app = typer.Typer()
app.command()(healpix_painter)

if __name__ == "__main__":
    app()
