# Detect and Read Handwritten Words

This is a **handwritten text recognition (HTR) pipeline** that operates on **scanned pages** and applies the following
operations:

* Detect words
* Read words

![example](./doc/example.png)

## Installation

* Download the zipped [model weights](https://www.dropbox.com/s/j1hl6bppecug0sz/models.zip?dl=0)
* Unzip
* Copy the files (reader.onnx, reader.json, detector.onnx) into the folder `htr_pipeline/models`
* Go to the root level of the repository (where `setup.py` is located)
* Execute `pip install .`

## Usage

### Run demo

* Additionally install matplotlib for plotting: `pip install matplotlib`
* Go to `scripts/`
* Run `python demo.py`
* The output should look like the plot shown above

### Run web demo (gradio)

* Additionally install gradio: `pip install gradio`
* Go to the root directory of the repository
* Run `python scripts/gradio_demo.py`
* Open the URL shown in the output

![example](./doc/gradio.png)

### Use Python package

Import the function `read_page` to detect and read text.

````python
import cv2
from htr_pipeline import read_page, DetectorConfig, LineClusteringConfig

# read image
img = cv2.imread('data/sample_1.png', cv2.IMREAD_GRAYSCALE)

# detect and read text
read_lines = read_page(img, 
                       DetectorConfig(height=200, enlarge=1), 
                       line_clustering_config=LineClusteringConfig(min_words_per_line=2))

# output text
for read_line in read_lines:
    print(' '.join(read_word.text for read_word in read_line))
````


If needed further configurations can be made by passing instances of these classes:

* `DetectorConfig`: configure the detector, the height should be roughly 50px per text-line
* `LineClusteringConfig`: configure the line clustering algorithm
* `ReaderConfig`: configure the reader


## Future work
* Better documentation of all the features (e.g., how to use a dictionary) - for now please have a look into the demo scripts to learn about the features of this package
* Add special characters like ".", "?", ...
* Optionally, read the whole line instead of single words
* Improve inference speed