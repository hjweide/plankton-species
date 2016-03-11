# Plankton species labeling interface

Disclaimer: this is my first time writing more than a few lines of JavaScript,
and as such there is probably too much copy-and-pasting and other bad practice
in this project.  Pull requests (and helpful comments) are welcome.

## Introduction
This is a small __work-in-progress__ project aimed at helping some ecologist colleagues
assign species labels to a collection of plankton images.  The idea is that the
collection of images is stored on a central server, to which collaborators can
connect and label images by selecting groups of images and choosing the appropriate
species.

## Design philosophy
With such a large number of images and the potential to collect many more, the
design philosophy is centered around *bulk* annotation.  It should be easy for
the user to select a large number of images at once (such as by drag-to-select)
and assign the same label to all of them.  This means that loading and
selecting images should be fast, and only minimal screen space should be used
for purposes other than displaying images.

## Features under development
Some features currently still under development include:

1. User authorization. Only authorized users should be allowed to change
   the database.  Except where intellectual property rights are concerned, any
user should be able to view the images in the database.
2. Performance. It seems as though the jQuery ```selectable```'s performance
degrades significantly when working with more than a few tens of thousands of
elements.  Eventually we want to scale to approximately a million images (and
perhaps even more), so this needs to be addressed.
3. Database design. The current database schema was designed with a simple
proof-of-concept in mind.  To be practically useful, we need to consult with
the intended users, i.e., plankton ecologists.

## Screenshots
The home screen shows as many images as we can realistically fit on the screen.
<img src="screenshots/home.png" alt="Home screen" style="width: 100%"/>

It is possible to select images by dragging or clicking.  A right-click presents
the user with a context-menu to choose the appropriate species.
<img src="screenshots/selection.png" alt="Home screen" style="width: 100%"/>

Often the visual cues necessary to distinguish between two species are very subtle.
With a double-click on the image, the user is presented with the image in its native
resolution along with additional information.
<img src="screenshots/overlay.png" alt="Home screen" style="width: 100%"/>
