[![Build Status](https://travis-ci.org/spencerahill/aospy-plot.svg?branch=develop)](https://travis-ci.org/spencerahill/aospy-plot) [![Coverage Status](https://coveralls.io/repos/github/spencerahill/aospy-plot/badge.svg?branch=develop)](https://coveralls.io/github/spencerahill/aospy-plot?branch=develop)

Plotting of data generated using [aospy](https://www.github.com/spencerahill/aospy).  Useful for generating plots comprising multiple aospy-outputted data across multiple panels.  For example, create a six-panel map-style plot of the convective precipitation fraction change in six different perturbation simulations, relative to a control simulation, with contours of the control simulation values overlayed in each.

For the time being, documentation is currently sparse and the codebase far from easy to understand.  Questions are best posed by opening an Issue in the Github repo.  A cleanup/refactor is intended but not necessarily imminent.

*Warning*
The code does not yet actually work...in porting it from aospy to here, I started to refactor it but didn't finish.  For a version that does work, see an aospy commit October 2016 or earlier.