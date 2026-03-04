# healpix-painter

`healpix-painter` is package meant to assist in working with imaging coverage of [HEALPix](https://healpix.sourceforge.io/) skymaps.
The package originated from the desire to have a simpler tool to automate observation planning for follow-up of gravitational wave events; many of the necessary features for such analysis have additional utility in different contexts.

`healpix-painter` is meant to be fairly simplistic, taking a list of possible telescope pointings and ranking them based on their relationship to the HEALPix skymap.
Decisions such as the number of pointings to be observed and the time of observation are not considered, following the philosophy that such decisions are often entangled in many complex factors, and so the human user is left to choose the best options for their science case.
`healpix-painter` does provide rankings on a filter-by-filter basis to facilitate decisions contingent on differences in coverage.

## Installation

`healpix-painter` may be installed with `pip` (`python` 3.12 recommended):

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

The easiest way to use `healpix-painter` is via the command-line interface (CLI).
Basic usage and stdout looks like this:

```bash
$ healpix-painter --skymap-filename /path/to/skymap_dir/skymap.fits.gz
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
Saving results in /path/to/skymap_dir...
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

In addition to this stdout, `healpix-painter` generates several output files.
Perhaps the most helpful diagnostic is the coverage vs. number of pointings plot `cumprob_npointings.png`, which displays the percent covered by the first $n$ pointings in the ranking, for each filter considered.
This plot can inform the decision on the number of pointings to observe, and to give a sense of the resources needed to cover the skymap.
A sample plot is shown here (presently from a different run than the CLI output provided above):

![alt text](https://github.com/tomas-cabrera/healpix-painter/blob/main/examples/example_plots/cumprob_npointings.png "cumprob_npointings.png")

A basic plot of the tiling coverage for each filter is provided for visualization, with the opaqueness of each footprint proportional to the amount of probabilty added by that pointing (beautification and addition of the HEALPix skymap to come in future versions).
A separate plot is generated for each filter available; here is a sample $g$-band plot (from the same run as the previous plot):

![alt text](https://github.com/tomas-cabrera/healpix-painter/blob/main/examples/example_plots/footprints_g.png "footprings_g.png")

The final file is a csv file with the pointings in the order of ranking; the columns included are `ra`, `dec`, and `probs_added`, the final of which includes the probability contributed by that observation that had not been covered by a higher-ranked pointing.

This tool was originally made to plan observations with the Dark Energy Camera (DECam).
As such, the default configuration uses a telescope footprint representing the outline of the DECam footprint (roughly hexagonal), and obtains the list of available pointings from the [NOIRLab AstroDataArchive](https://astroarchive.noirlab.edu).
The sections below explain some of the dials you might want to twiddle when using this tool.

### I want to change the output directory

By default the output of the program is stored in the same directory as the skymap (for downloaded skymaps this is `healpix-painter/src/healpix_painter/data/skymaps/.cache/SYYMMDDaa`).
A preferred output directory may be specified, with the skymap copied to the output directory for ease of access:

```bash
healpix-painter --output-dir /path/to/output/dir
```

### I want to fetch a skymap from GraceDb instead using a local skymap file

Instead of specifying the path to a skymap file, `healpix-painter` may be prompted to fetch the most recent GraceDb skymap for a gravitational wave event with the following flag:

```bash
healpix-painter --lvk-eventname SYYMMDDaa
```

### I want to use a different ranking algorithm for selecting pointings

The default ranking algorithm scores the available pointings based on how much *new* probability they cover, that is, how much probability they cover that hasn't been covered by a previous pointing.
The only other algorithm available at this time ranks pointings first by the maximum probability density contained in the footprint, breaking ties by the total new probabilty in the footprint; this mode can be enabled with

```bash
healpix-painter --scoring probden_probadd
```

As a note, while this does minimize gaps in coverage, it also can lead to quite inefficient observation plans when there are many overlapping pointings available for the field.

### I want to update the cached list of DECam pointings

If one wishes to force a refresh of the cached DECam tiling from NOIRLab, add the flag

```bash
healpix-painter --tiling-force-update
```

### I want to use a footprint other than the default DECam hexagon

An alternate DECam footprint is available which represents the footprint fully as an array of CCDs; this may be enabled with the flag

```bash
healpix-painter --footprint DECamFootprint
```

This version includes the implementation of `ZTFFootprint` and `ZTFConvexHull`, which are the fully-detailed and outline versions of the ZTF footprint; please note that no tiling has been implemented for ZTF yet, so this facility is not quite ready for full `healpix-painter` use.
That said, the footprint may be useful when using some of the other features of the package.

Other telescopes may be added in the future; if you wish to do so yourself, you will need to convert your telescope footprint into a `.crtf` file that can be read by `astropy.regions`.
The [healpix-painter.footprints.make_footprint_crtf](https://github.com/tomas-cabrera/healpix-painter/blob/main/src/healpix_painter/footprints.py#L256) function has been included to facilitate the generation of the `.crtf` files from a list of coordinates; [examples/get_footprint_coords_decam.py](https://github.com/tomas-cabrera/healpix-painter/blob/main/examples/get_footprint_coords_decam.py) and [examples/get_footprint_coords_ztf.py](https://github.com/tomas-cabrera/healpix-painter/blob/main/examples/get_footprint_coords_ztf.py) are the scripts used to generate the `.crtf` files for the respective telescopes, see them for more information on how to do this for your own instrument.
