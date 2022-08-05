# Projects from Digital Image class

This repository contains projects I made for the Digital Image class at Texas
A&M University. I made all using Python. You can find more about the class 
and the projects in this link:

[http://people.tamu.edu/~ergun/courses/viza654/20fall/projects/index.php](http://people.tamu.edu/~ergun/courses/viza654/20fall/projects/index.php)

## Project 1 - PPM Image format and image from a function

![function](docs/function.jpg)

Image created from a function

## Project 2 - Image creation of half-plane, circle and polynomial. Anti-aliasing.

![circle](docs/circle.jpg)

![half plane](docs/half_plane.jpg)

![polynomial](docs/polynomial.jpg)

## Project 3 - Line, star and convex polygon

![line](docs/line.jpg)

![star](docs/star.jpg)

![convex](docs/convex.jpg)

## Project 5 - Filters: Box blur, motion blur, dilation, erosion and edge

![blur](docs/blur.jpg)

Box blur

![motion](docs/motion.jpg)

Motion blur

| ![](images/circle.jpg)  | ![](docs/dilation.jpg)  |
|:---:|:---:|
| Original | Dilated |

| ![](images/box.jpg)  | ![](docs/erosion.jpg)  |
|:---:|:---:|
| Original | Eroded |


![edge](docs/edge.jpg)

Edge

## Project 6 - Non-Stationary Filters (image instead of kernel): Motion blur and dilation

| Filter  | Input Image  | Image as Filter   | Result  |
|---|---|---|---|
| Motion Blur  | ![](images/mickey.jpg) | ![](images/move.jpg) | ![](docs/non_stationary_motion.jpg)  |
| Dilation  | ![](images/mickey.jpg) | ![](images/dilate.jpg) | ![](docs/non_stationary_dilation.jpg) |


## Project 7 - Transformations

## Project 9 - Dithering

See [dithering.py](dithering.py)

![dithering](docs/dithering.jpg)

## Project 10 - Seaming and carving

See [seam_carving.py](seam_carving.py)

| Left Image | Right Image | Seaming |
|---|---|---|
| ![](images/img1.jpg) | ![](images/img2.jpg) | ![](docs/seamed.jpg) |

| Original | Carved |
|---|---|
| ![](images/field.jpg) | ![](docs/carved.jpg) |

## Project 11 - Videos

See [video.py](video.py)

![](docs/ordered.gif)

Ordered Dithering Video

![](docs/specular.gif)

Phong Shader Video






## Dependencies

- numpy
- Pillow
- progress
