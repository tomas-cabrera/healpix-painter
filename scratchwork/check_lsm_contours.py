from ligo.skymap.postprocess.crossmatch import crossmatch
from healpix_painter import healpix
from astropy.table import Table
import os.path as pa
import matplotlib.pyplot as plt
import jax.numpy as jnp
from astropy.coordinates import SkyCoord

# Load skymap
gdir = pa.dirname(pa.dirname(pa.abspath(__file__)))
sm_moc_path = f"{gdir}/.cache/test_skymap_flattened.fits"
sm_moc_path = "/home/tcabrera/data/academia/projects/decam_followup_O4/S241125n/Bilby.offline0.flattened.fits"
sm_moc_tab = Table.read(sm_moc_path)

# Calculate contours
cs_keys = [0.9, 90]
cs = healpix.calc_contours_for_skymap(sm_moc_tab, contours=cs_keys)


# Plot contours
for k, c in zip(cs_keys, cs):
    for cii, ci in enumerate(c):
        ci = jnp.array(ci)
        plt.plot(ci[:, 0], ci[:, 1], label=f"Contour {k}:{cii}")
plt.legend()
plt.show()
plt.close()
