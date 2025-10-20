# project-ochre

## Current training version

#### v1.1

- 6 frame input + yaw/axis data, single frame prediction.

Results: 
- Trained for ~14 epochs, ~90 hours of A100
- Decent video generation (inference.py) with 6 frame latent input. Some videos perform very well (ocean with islands, snow biome) whereas others tend to collapse (tree biomes turn to brownish-grey blur)
- Poor video generation (inference.py) with 1 frame duplicated x6 latent input. Very quickly collapses to some blurry grey-brown average.
- Horrible live inference (live_inference.py) performance. Completely nonresponsive to movements. Shows some change when fed 6-frame latent initial input, but over time collapses to grey-brown average.

Data
- Each child of preprocessed data corresponds to a random 7-frame subset of all the mp4s


#### v2.0
- 1 frame input + yaw/axis data, single frame prediction.
- Latent noise injection - gaussian noise in inputs to encourage recovery from previous error drift
- Self-sampling - occasionally using predicted frame 1 as input to predict frame 2 **(Applied after ___ epochs)**
- Removing 1s from start of each mp4 due to sudden teleporting

Data 
- Each child folder of preprocessed data now corresponds to a mp4
- Downsampling FPS - max pooling actions and sampling frames for 8 FPS

