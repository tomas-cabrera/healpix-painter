# healpix-painter

My attempt at writing a package to optimize coverage of HEALPix skymaps with wide-field telescopes.
The inputs are the skymap, a set of telescope pointings to choose from, and the footprint of the telescope.
At the moment this is focused on DECam, and so the list of pointings is automatically fetched from the NOIRLab AstroDataArchive.

The main output is a set of csv's, one for each telescope filter containing the pointings and the probability that pointing adds to the plan as ordered in the csv
(that is, the probability contained in a footprint centered on that pointing that has not already been covered by a previous pointing).
Additional products include plots of the pointings chosen with the given footprint; plotting the skymap has not been implemented yet.
The final product is a plot of cumulative probability covered versus the number of pointings, to allow the user to assess the diminishing returns offered by adding more pointings to the final observation plan.

This package is intended to be simplistic, ranking all available pointings as specified and not attempting to do any kind of scheduling/packing optimization.
This choice is motivated by the idea that the relationship between qualities of an observation plan such as filters, exposure times, and number of pointings, and the science yield is complex, and challenging to automate in a satisfactory manner.
This package aims to produce an easily-digested summary of the options available when considering how to cover the area of HEALPix skymap, and leaves the more concrete decision making up to the human user.

## Installation

`healpix-painter` may be installed with `pip`:

```bash
$ pip install healpix-painter
...
Successfully installed healpix-painter.
```

For the most current version (beware of experimental features!), install from source:

```bash
$ git clone https://github.com/tomas-cabrera/healpix-painter
Cloning into 'healpix-painter'...
remote: Enumerating objects: 135, done.
remote: Counting objects: 100% (135/135), done.
remote: Compressing objects: 100% (78/78), done.
remote: Total 135 (delta 54), reused 118 (delta 37), pack-reused 0 (from 0)
Receiving objects: 100% (135/135), 7.70 MiB | 12.99 MiB/s, done.
Resolving deltas: 100% (54/54), done.
$ pip install ./healpix_painter
...
Successfully installed healpix-painter.
```

## Getting started

The easiest way to use `healpix_painter` is via the command-line interface (CLI).
Basic usage and stdout looks like this:

```bash
$ healpix-painter --skymap-filename /path/to/skymap/skymap.fits.gz
Loading skymap...
Calculating 90% contour regions...
Loading DECam archival pointings...
Fetching DECam archival pointings from https://astroarchive.noirlab.edu...
Selecting pointings near 90% contour regions...
Found 33 clustered pointings near 90% contour regions:
  u: 1 pointings
  g: 28 pointings
  r: 14 pointings
  i: 21 pointings
  z: 15 pointings
  Y: 1 pointings
Evaluating healpix coverage of pointings with footprint...
Selecting obsplan by 'probadd' scoring...
Saving results in /path/to/skymap...
========================================
Coverage summary:
  u: 11.27% (1 pointings)
  g: 92.66% (24 pointings)
  r: 90.45% (14 pointings)
  i: 91.90% (21 pointings)
  z: 89.76% (15 pointings)
  Y: 9.33% (1 pointings)
========================================
```

This tool was originally made to plan observations with the Dark Energy Camera (DECam).
As such, the default configuration uses a telescope footprint representing the outline of the DECam footprint (roughly hexagonal), and obtains the list of available pointings from the [NOIRLab AstroDataArchive](https://astroarchive.noirlab.edu).
The sections below explain some of the dials you might want to twiddle when using this tool.

### I want to change the output directory

By default the output of the program is stored in the same directory as the skymap (for downloaded skymaps this is `healpix-painter/src/healpix_painter/data/skymaps/.cache/SYYMMDDaa`).
A preferred output directory may be specified, with the skymap copied to the output directory for ease of access:

```bash
--output-dir /path/to/output/dir
```

### I want to fetch a skymap from GraceDb instead using a local skymap file

Instead of specifying the path to a skymap file, `healpix-painter` may be prompted to fetch the most recent GraceDb skymap for a gravitational wave event with the following flag:

```bash
--lvk-eventname SYYMMDDaa
```

### I want to update the cached list of DECam pointings

If one wishes to force a refresh of the cached DECam tiling from NOIRLab, add the flag

```bash
--tiling-force-update
```

### I want to use a footprint other than the default DECam hexagon

An alternate DECam footprint is available which represents the footprint fully as an array of CCDs; this may be enabled with the flag

```bash
--footprint DECamFootprint
```

Other telescopes may be added in the future; if you wish to do so yourself, you will need to convert your telescope footprint into a `.crtf` file that can be read by `astropy.regions`.
See [healpix_painter.footprints.Footprint](https://github.com/tomas-cabrera/healpix-painter/blob/main/src/healpix_painter/footprints.py#L23) for more details.
