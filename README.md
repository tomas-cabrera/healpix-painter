# healpix_painter

My attempt at writing a package to optimize coverage of HEALPix skymaps with wide-field telescopes.
At the moment I'm using this readme to plan the package.

## Package planning

- Read in HEALPix map
  - In general: read in map from file
  - GW speciality: get map from GraceDB
    - will need to flatten skymap
- Define telescope being used
  - Read in from file?
    - Focal plane: define a format that describes the focal plane as a set of polygons, with the whole set centered on the pointing of the telescope
      - astropy regions can be written as file; choose format (CASA, DS9, or FITS) that is the easiest to define a custom focal plane with
  - Might need to resample HEALPix map s.t. pixels are sufficiently smaller than the focal plane
- Define telescope tiling to use
  - In general: define a uniform tiling of the sky appropriate for the telescope focal plane
    - Ideally will need some intelligence to choose good tilings for square vs. round focal planes
  - Preset tiling: read in a list of pointings to choose from
    - Extra special (for DECam): generate tiling from NOIRLab Astro Data Archive
- Select pointings
  - Get coords for HEALPix tiles
  - Trim HEALPix tiles to those covering desired upper limit (optimization, to avoid working with entire map later)
    - Can choose either % upper limit upper limit?
    - Note that a number of pointings/time upper limit can be converted to % upper limit (add pixels until respective area is met and calculate probability, ignoring focal plane shape)
  - Trim telescope tiling to upper limits (as previous)
  - Pre-calculate coverage of each tile
    - Need function that takes focal plane+pointing and HEALPixels and provides as mask identifying the covered pixels
  - Select tiles
    - Loop until threshold is met (%, # of pointings/time, etc.)
      - Take current selection of pointings, add each of the next available choices, and calculate the coverage of the result
        - The pre-calculated masks should make it easy to select covered pointings (np.any(..., axis=...))
        - Could also sum the pixels that would be added, i.e. not in the current selection but in the next option
      - Choose the pointing that adds the most probability and add it to the selection
    - Might also want to implement an interactive option, where a prob covered vs. # of pointings/time plot is shown and the user can select the value appropriately
- Output
  - .csv or whatever with pointings
  - plot of healpix map with overlayed pointings and % covered
  - Special: convert pointings to DECam json (would need filters, propid, exptimes, etc.; perhaps provide template json?)