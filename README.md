# Crater Detection Tool

### Installation Guide

By using the command: 
```
pip install ApolloCraterDetectionTool
```

### User instructions

### 📖Package
After installation, download the model(.pt file) using the download_models.py. 

First, in terminal, 
```
cd package_path + '/apollo'
```

Next, run the 

```
python download_models.py
```

Then it would download a models folder in the same path, which includes Mars_best.pt, Moon_Left_Model.pt, and Moon_Right_Model.pt files.

Then add the module by running in python:
```
from apollo import *
```

Congratulation, now you can use the models to detecte the crater in two ways.
## 1.UI.py
Use the commmand
```
python UI.py
```
to run software and the UI interface will show up.  ** Be care to install the yolov5 package first **
![Image](https://github.com/edsml-zw1622/33/raw/main/Img/Interface0.png)

To run the model, firstly, select the Model type, e.g. Yolov5, Mars model.

Next, click the `test folder` button and select the test images file and input a folder name in the `result directory`. 

Now, you can click the `Detect` to use the model to detect. Then the crater labels would be saved in the `user_directory/folder_name/detection`, the image with bounding boxes on the craters  and  the model's statistic would appear nearby.

Then, more results could be viewed by using the `Browse`.

Use the `Browse`, view one image in window, so you can view the image after detection.

## 2.CraterPredictor Class

To begin with, you can create an object named detection:
```
detection = CraterPredictor(mars_model, moon_model1, moon_model2, results_path, test_images_path, test_labels_path, img_size)
```
mars_model is string of path to the mars model weights, moon_model1, moon_model2: string of path to the moon model weights, results_path is string of path to the directory to store the results for the user, test_images_path is the str path to the directory containing the test images, test_labels_path is the string of path to the directory containing the test labels, optionally given by the user, and img_size is the size of the input images.



## Model 
This class is about the model detection and some funcitions you can use.

In detection, two models could be selected, one is moon model and another is mars.

```
detection.predict_lunar_large_image(images)
```
This function would Predict lunar craters in a large lunar image by slicing it into smaller sub-images and making predictions on each sub-image, and the predictions are then aggregated into a single result.

```
detection.predict_mars_craters(images)
```
Use this function could let function to detect the whole images and return a csv. file in user_directory/detections.file using the moon model.

```
detection.draw_boxes(label_path=None)
```
Use this result to D=draw bounding boxes on images, both ground truth (if provided) and detections and results are saved in 'results_path/images/detections' and 'results_path/images/detections_and_gt'.

```
detection.get_statistics(label_path=None):
```
Use this function would save the true positive (tp), false positive (fp), and false negative (fn) values to a csv file. label_path (str): The path to the directory containing the label csv files. If not specified, the function returns without performing any operations.


### Analysis and Visualization
```
analysis.convert2physical(labels, subimg_pix, indices)
```
The function is to return the location (lat, lot) based on the information igven by user. `labels` means the predicted location and size of craters in the picture. `subimg_pix` is the number of pixels in each sub-picture, and `indices`  is the indices of different sub-pictures.

```
size_frequency_compare_plot(folderpath, detection, real)
```
From this function, we can plot a separate plot of the cumulative crater size-frequency distribution of detected craters, if information the `crater size` is provided. `folderpath` means the user-specified input folder location. `detection` is the physical, real-world crater sizes in detection, and `real`  is the physical, real-world crater sizes in real world.

### Model Perform Metric
After training our CDM, we randomly selected two sub-regions [Fig1](https://github.com/edsml-zw1622/33/raw/main/Img/B-19.jpg) and [Fig2](https://github.com/edsml-zw1622/33/raw/main/Img/B-0.jpg)(B-65-19, B-64-0). We plotted the crater size frequency distribution of these two regions. The blue broken line in the figure below represents the prediction result of our detector algorithm in this area, and the black broken line represents the actual distribution of crater diameters in this area. In addition, we also calculated the difference between the model prediction and the ground truth, and the residual is indicated by the red broken line.

![Image](https://github.com/edsml-zw1622/33/raw/main/Img/B-65-19.jpg)

![Image](https://github.com/edsml-zw1622/33/raw/main/Img/B-64-0.jpg)

The following two pictures show the `x, y, w, h, latitude, longitude, diameter` of the B-64-0 area. The first picture is the prediction result of the detector algorithm, and the second picture is the real information of the crater in this area . You can see the data in [Detection Data](https://github.com/edsml-zw1622/33/raw/main/Img/Detection.jpg) and [Label Data](https://github.com/edsml-zw1622/33/raw/main/Img/labeldata.jpg).


### UI Example:
![Image](https://github.com/edsml-zw1622/33/raw/main/Img/Example0.png)

### Documentation

The code includes [Sphinx](https://www.sphinx-doc.org) documentation. On systems with Sphinx installed, this can be build by running

```
python -m sphinx docs html
```

then viewing the generated `index.html` file in the `html` directory in your browser.

For systems with [LaTeX](https://www.latex-project.org/get/) installed, a manual pdf can be generated by running

```bash
python -m sphinx  -b latex docs latex
```

Then following the instructions to process the `CraterDetectionTool.tex` file in the `latex` directory in your browser.

### Testing

The tool includes several tests, which you can use to check its operation on your system. With [pytest](https://doc.pytest.org/en/latest) installed, these can be run with

```bash
python -m pytest --doctest-modules apollo
```

Additionally, we write a test to check our analysis.py. You can use it first cd the folder *tests*, and run the **test_analysis.py**.

### Reading list

 - (Description of lunar impact crater database Robbins, 2019.)
[https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2018JE005592]

 - (Yolvo5 Model description)
[https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data]

 - (Equirectangular projection descripition)[https://assets.publishing.service.gov.uk/government/uploads/system/uploads/attachment_data/file/283357/ILRSpecification2013_14Appendix_C_Dec2012_v1.pdf]
 
 
 
 ### Lisence
 
 MIT[https://opensource.org/licenses/MIT]
 
 
