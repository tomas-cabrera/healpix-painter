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
    scoring: Annotated[
        str,
        typer.Option(
            help="The scoring algorithm to use to rank pointings; 'probsum' by default"
        ),
    ] = "probsum",
):
    """A wrapper CLI for healpix_painter routines.

    Parameters
    ----------
    routine : Annotated[ str, typer.Option, optional
        _description_, by default "The healpix_painter routine to use; 'basic_painter' by default" ), ]="basic_painter"
    healpixfilename : Annotated[ str, typer.Option, optional
        _description_, by default "The path to the HEALPix skymap to tile"), ]=None
    lvk_eventname : Annotated[ str, typer.Option, optional
        _description_, by default "The LVK event id to tile"), ]=None
    footprint : Annotated[ str, typer.Option, optional
        _description_, by default "The telescope footprint to use for tiling; DECamConvexHull by default" ), ]="DECamConvexHull"
    scoring : Annotated[ str, typer.Option, optional
        _description_, by default "The scoring algorithm to use to rank pointings; 'probsum' by default" ), ]="probsum"

    Raises
    ------
    ValueError
        _description_
    """
    # Select routine
    if routine == "basic_painter":
        basic_painter(
            skymap_filename=skymap_filename,
            lvk_eventname=lvk_eventname,
            footprint=getattr(footprints, footprint),
            scoring=scoring,
        )
    else:
        raise ValueError(f"Routine '{routine}' not implemented.")


app = typer.Typer()
app.command()(healpix_painter)

if __name__ == "__main__":
    app()
