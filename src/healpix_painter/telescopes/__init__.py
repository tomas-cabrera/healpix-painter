from . import decam, ztf

FOOTPRINT_REGISTRY = {
    "DECamFootprint": decam.DECamFootprint,
    "DECamConvexHull": decam.DECamConvexHull,
    "ZTFFootprint": ztf.ZTFFootprint,
    "ZTFConvexHull": ztf.ZTFConvexHull,
}
