## Getting started with images in the digital humanities: What tools? Which data? And, most importantly: what questions?

Welcome to the workshop! If you haven't already filled out this form, I would appreciate if you would fill it out: [https://tinyurl.com/dhrx-survey](https://tinyurl.com/dhrx-survey)

## Rough agenda, but no worries if this changes!

- Go over survey results
- Discuss any experiences we have working with images (keep these in mind during the workshop today!)
- Goals of today
  - Check-in: who is everyone? have folks tackled problems from a statistical perspective before?
    - What types of data have folks used?
  - Understanding images as data
    - Loading images
    - Images as arrays (I'll talk a bit here)
    - Filters
    - Convolutions as filters (I'll talk a bit here)
    - Experimenting with image representations
    - Replicating ["Neural Neighbors: Capturing Image Similarity"](http://dhlab.yale.edu/projects/neural_neighbors.html)
      - Nearest neighbor search (I'll talk a bit here)
      - Convolutional neural networks (I'll talk a bit here)
  - A few other image APIs for your toolboxes
    - [Google Cloud Vision API, with web demo](https://cloud.google.com/vision/)
    - [face++](https://www.faceplusplus.com/)
    - [Microsoft Azure, with web demo](https://azure.microsoft.com/en-us/services/cognitive-services/computer-vision/)
  - Discussion: given the experience today, do you see any applications of computer vision to your own work?
    - Pixel-level studies versus concept level studies? Key question: what level of abstraction is appropriate for your work?
    - For pixel-level: my recommendation is to look for "segmentation"; "keypoint detection"; color-based methods
    - For concept-level: my recommendation is to use neural networks.
    - Large variety of tasks.
  - If time: diving deeper into convolutional neural networks

Coding instructions

## Part 1: Replicating Neural Neighbors

- Check ["Neural Neighbors: Capturing Image Similarity"](http://dhlab.yale.edu/projects/neural_neighbors.html)
- Download and unzip the files listed on the github page (5K coco images, 5K british library images)
- Open up terminal, and navigate using `cd` to this directory.
- run `jupyter notebook neural_neighbors.ipynb`
- show demo visualization of image space.

## Part 2: Diving Deeper

- run `jupyter notebook digging_deeper.ipynb`