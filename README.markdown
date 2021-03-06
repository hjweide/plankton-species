# Plankton species labeling interface

Disclaimer: this is my first time writing more than a few lines of JavaScript
(and for a purpose other than my own amusement), and as such there is probably
too much copy-and-pasting and other bad practice in this project.  Pull
requests (and helpful comments) are welcome.

## Contents
1. [Introduction](#introduction)
2. [Design philosophy](#design-philosophy)
3. [Features under development](#features-under-development)
4. [Screenshots](#screenshots)
5. [Development](#development)

## <a name="introduction"></a>Introduction
This is a small __work-in-progress__ project aimed at helping some ecologist colleagues
assign species labels to a collection of plankton images.  The idea is that the
collection of images is stored on a central server, to which collaborators can
connect and label images by selecting groups of images and choosing the appropriate
species.

## <a name="design-philosophy"></a>Design philosophy
With such a large number of images and the potential to collect many more, the
design philosophy is centered around *bulk* annotation.  It should be easy for
the user to select a large number of images at once (such as by drag-to-select)
and assign the same label to all of them.  This means that loading and
selecting images should be fast, and only minimal screen space should be used
for purposes other than displaying images.

## <a name="features-under-development"></a>Features under development
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

## <a name="screenshots"></a>Screenshots
The home screen shows as many images as we can realistically fit on the screen.
<img src="screenshots/home.png" alt="Home screen" style="width: 100%"/>

It is possible to select images by dragging or clicking.  A right-click presents
the user with a context-menu to choose the appropriate species.
<img src="screenshots/selection.png" alt="Home screen" style="width: 100%"/>

Often the visual cues necessary to distinguish between two species are very subtle.
With a double-click on the image, the user is presented with the image in its native
resolution along with additional information.
<img src="screenshots/overlay.png" alt="Home screen" style="width: 100%"/>

## <a name="development"></a>Development
This project is still very much under development.  The ```init_dummy_db.py```
script is included to generate dummy data and populate a dummy database to
allow quick testing and development.  Simply run
```python
python init_dummy_db.py
python pyplankton.py
```
and then open ```http://127.0.0.1:5000/``` in your browser.
