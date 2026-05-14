import os
import os.path as pa

from astropy.table import Table
from astropy.coordinates import SkyCoord
import ligo.skymap.plot  # noqa: F401
import matplotlib.pyplot as plt

from healpix_painter.io import plotting
from healpix_painter.telescopes.decam import DECamFootprint
from healpix_painter.telescopes.ztf import ZTFFootprint

# Get test data directory
datadir = pa.join(
    pa.dirname(pa.dirname(os.path.abspath(__file__))),
    "test_data",
)

# File paths
S230922g_skymap_filename = pa.join(
    datadir,
    "S230922g.Bilby.multiorder.fits",
)
S251112cm_skymap_filename = pa.join(
    datadir,
    "S251112cm.Bilby.multiorder.fits",
)
S25111cm_pointings_decam_filename = pa.join(
    datadir,
    "S251112cm.pointings_decam.csv",
)
S25111cm_pointings_ztf_filename = pa.join(
    datadir,
    "S251112cm.pointings_ztf.csv",
)


def test_plot_skymap_gradient_mollweide():
    # Initialize axes
    ax = plt.axes(projection="astro hours mollweide")
    ax.grid()
    # Plot
    plotting.plot_skymap_gradient(ax, S230922g_skymap_filename)
    plt.savefig("test_plot_skymap_gradient_mollweide.png")
    plt.close()
    assert True


def test_plot_skymap_contours_mollweide():
    # Initialize axes
    ax = plt.axes(projection="astro hours mollweide")
    ax.grid()
    # Plot
    plotting.plot_skymap_contours(ax, S230922g_skymap_filename)
    plt.savefig("test_plot_skymap_contours_mollweide.png")
    plt.close()
    assert True


def test_plot_skymap_gradient_zoom():
    # Initialize axes
    ax = plt.axes(
        projection="astro hours zoom",
        center="22.5h -25d",
        radius="15 deg",
    )
    ax.grid()
    # Plot
    plotting.plot_skymap_gradient(ax, S230922g_skymap_filename)
    plt.savefig("test_plot_skymap_gradient_zoom.png")
    plt.close()
    assert True


def test_plot_skymap_contours_zoom():
    # Initialize axes
    ax = plt.axes(
        projection="astro hours zoom",
        center="22.5h -25d",
        radius="15 deg",
    )
    ax.grid()
    # Plot
    plotting.plot_skymap_contours(ax, S230922g_skymap_filename)
    plt.savefig("test_plot_skymap_contours_zoom.png")
    plt.close()
    assert True


def test_plot_skymap_gradient_mollweide_coverage():
    # Initialize axes
    ax = plt.axes(projection="astro hours mollweide")
    ax.grid()
    # Plot skymap
    plotting.plot_skymap_gradient(ax, S251112cm_skymap_filename)
    # Plot DECam coverage
    tab_decam = Table.read(S25111cm_pointings_decam_filename)
    scs_decam = SkyCoord(
        ra=tab_decam["ra"],
        dec=tab_decam["dec"],
        unit="deg",
    )
    plotting.plot_footprints(
        ax,
        DECamFootprint,
        scs_decam,
    )
    # Plot ZTF coverage
    tab_ztf = Table.read(S25111cm_pointings_ztf_filename)
    scs_ztf = SkyCoord(
        ra=tab_ztf["ra"],
        dec=tab_ztf["dec"],
        unit="deg",
    )
    plotting.plot_footprints(
        ax,
        ZTFFootprint,
        scs_ztf,
    )
    plt.savefig("test_plot_skymap_gradient_mollweide_coverage.png")
    plt.close()
    assert True


def test_plot_skymap_gradient_zoom_coverage():
    # Initialize axes
    ax = plt.axes(
        projection="astro hours zoom",
        center="0h -30d",
        radius="25 deg",
    )
    ax.grid()
    # Plot skymap
    plotting.plot_skymap_gradient(ax, S251112cm_skymap_filename)
    # Plot DECam coverage
    tab_decam = Table.read(S25111cm_pointings_decam_filename)
    scs_decam = SkyCoord(
        ra=tab_decam["ra"],
        dec=tab_decam["dec"],
        unit="deg",
    )
    plotting.plot_footprints(
        ax,
        DECamFootprint,
        scs_decam,
    )
    # Plot ZTF coverage
    tab_ztf = Table.read(S25111cm_pointings_ztf_filename)
    scs_ztf = SkyCoord(
        ra=tab_ztf["ra"],
        dec=tab_ztf["dec"],
        unit="deg",
    )
    plotting.plot_footprints(
        ax,
        ZTFFootprint,
        scs_ztf,
    )
    plt.savefig("test_plot_skymap_gradient_zoom_coverage.png")
    plt.close()
    assert True


if __name__ == "__main__":
    test_plot_skymap_gradient_mollweide()
    test_plot_skymap_contours_mollweide()
    test_plot_skymap_gradient_zoom()
    test_plot_skymap_contours_zoom()
    test_plot_skymap_gradient_mollweide_coverage()
    test_plot_skymap_gradient_zoom()
