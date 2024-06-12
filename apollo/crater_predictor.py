from yolov5 import detect
import os
import csv
import shutil
import cv2
import numpy as np
import pandas as pd
import analysis


class CraterPredictor:
    """
    Class to generate user outputs from input images
    by running object detection on the images using YOLOv5 models.
    """

    def __init__(
        self,
        mars_model,
        moon_model1,
        moon_model2,
        results_path,
        test_images_path,
        test_labels_path=None,
        img_size=416,
    ):
        """
        Parameters
        ----------
        mars_model : str (.pt file)
            Path to the mars model weights
        moon_model : str (.pt file)
            Path to the moon model weights
        results_path : str
            Path to the directory to store the results for the user
        test_images_path : str
            Path to the directory containing the test images
        test_labels_path: str
            Path to the directory containing the test labels,
            optionally given by the user
        img_size: int
            Size of the input images, specified by the user
        """

        self.mars_model = mars_model
        self.moon_model1 = moon_model1
        self.moon_model2 = moon_model2
        self.results_path = results_path
        self.detections_path = results_path + "detections/"
        self.agg_detections_path = results_path + "agg_detections/"
        self.images_path = results_path + "images/"
        self.stats_path = results_path + "statistics/"
        self.test_images_path = test_images_path
        self.test_labels_path = test_labels_path
        self.img_size = img_size

    def crop_img(self, img, x, y, pixel_size=416):
        """
        Crop a sub-image from a larger image.

        Parameters
        ----------
        img (ndarray): the larger image to crop a sub-image from
        x (int): the x-coordinate of the top-left corner of the sub-image
        y (int): the y-coordinate of the top-left corner of the sub-image
        pixel_size (int): the size (h and w) of the sub-image (default: 416)

        Returns:
        ndarray: the cropped sub-image
        """
        h, w, _ = img.shape

        sub_img = img[
            y * pixel_size: (y * pixel_size) + pixel_size,
            x * pixel_size: (x * pixel_size) + pixel_size,
        ]

        if sub_img.shape[0] != pixel_size or sub_img.shape[1] != pixel_size:
            # Fill the empty part with black color
            sub_img = cv2.copyMakeBorder(
                sub_img,
                0,
                pixel_size - sub_img.shape[0],
                0,
                pixel_size - sub_img.shape[1],
                cv2.BORDER_CONSTANT,
                value=[0, 0, 0],
            )

        return sub_img

    def slice_lunar_large_image(self, img_file, folder_name, image_size):
        """
        Slice a large lunar image into smaller sub-images and save them

        Parameters
        ----------
        img_file (str): path to the large lunar image file
        folder_name (str): name of the folder to save the sub-images to
        image_size (int): the size (height and width) of each sub-image

        Returns
        ----------
        tuple: the shape of the large lunar image
        """

        img = cv2.imread(img_file)

        # figure out how many iterations
        x_range = img.shape[1] // image_size
        y_range = img.shape[0] // image_size

        for x in range(x_range + 1):
            for y in range(y_range + 1):
                sub_img = self.crop_img(img, x, y, image_size)
                cv2.imwrite(folder_name + "/{}-{}.jpg".format(x, y), sub_img)

        return img.shape

    def predict_lunar_large_image(self, images):
        """
        Predict lunar craters in a large lunar image by slicing it into
        smaller sub-images and making predictions on each sub-image.
        The predictions are then aggregated into a single result.

        Parameters
        ----------
        images (str): path to the directory containing the large lunar image(s)
        """
        image_size = self.img_size

        if os.path.exists(self.agg_detections_path):
            shutil.rmtree(
                self.agg_detections_path, ignore_errors=False, onerror=None)

        os.mkdir(self.agg_detections_path)

        for i, img_file in enumerate(os.listdir(images)):
            print(img_file)

            result_file_name = img_file.replace(".jpg", ".csv")

            img_file = os.path.join(images, img_file)

            folder_path = "sliced_lunar_images_" + str(i)

            if os.path.exists(folder_path):
                shutil.rmtree(folder_path, ignore_errors=False, onerror=None)

            os.mkdir(folder_path)

            large_img_size = self.slice_lunar_large_image(
                img_file, folder_path, image_size
            )

            self.predict_moon_craters(folder_path)

            self.aggregate_labels(result_file_name, large_img_size, image_size)

    def aggregate_labels(self, result_file, large_img_size, image_size):
        """
        Combine the predictions of lunar craters in multiple sub-images into
        a single result.

        Parameters
        ----------
        result_file (str): name of the file to save the aggregated result to
        large_img_size (tuple): the shape of the large lunar image
        image_size (int): the size (height and width) of each sub-image
        """
        rows = []
        # loop through the list of CSV files
        for filename in os.listdir(self.detections_path):
            # read each CSV file into a dataframe
            if filename.endswith(".csv"):
                with open(
                    os.path.join(
                        self.detections_path, filename), "r") as f:
                    reader = csv.reader(f)
                    # add each row from the csv file to the rows list
                    fs = filename.split('-')
                    x = int(fs[0])
                    y = int(fs[1][:-4])

                    for row in reader:
                        row = [float(i) for i in row[:4]]
                        row = self.convert_to_global(
                            row, x, y, large_img_size[1],
                            large_img_size[0], image_size
                        )
                        rows.append(row)

        res_file = os.path.join(self.agg_detections_path, result_file)

        with open(res_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(rows)

    def convert_to_global(
        self, detection, i_sub_img, j_sub_img, w_img, h_img, sub_img_px=416
    ):
        """
        Converts the detected bounding boxes of a sub image to the global
        coordinates in the larger image

        Parameters
        ----------
        detection (list): detected crater bounds in form of [x, y, w, h]
        i_sub_img (int): the sub image x index in the larger image
        j_sub_img (int): the sub image y index in the larger image
        w_img (int): the width of the large image
        h_img (int): the height of the large image
        sub_img_pix (int): the pixel dimensions of the sub image

        Return
        ----------
        list: [x, y, w, h] as values in the large image
        """
        x_local, y_local, w_local, h_local = detection

        x_global = (x_local + i_sub_img) * sub_img_px / w_img
        y_global = (y_local + j_sub_img) * sub_img_px / h_img
        w_global = w_local * sub_img_px / w_img
        h_global = h_local * sub_img_px / h_img
        return [x_global, y_global, w_global, h_global]

    def predict_moon_craters(self, images):
        """
        Predict moon craters and put results in user_directory/detections.
        Result is a csv file for each image containing the detected locations

        Parameters
        ----------
        images : str
            Path to the test images

        Example
        -------
        predictor.predict_moon_craters('test_images/')
        """
        self.yolo_detect(
            images, [self.moon_model1, self.moon_model2], name="detections"
        )

    def predict_mars_craters(self, images):
        """
        Predict mars craters and put results in user_directory/detections.
        Result is a csv file for each image containing the detected locations

        Parameters
        ----------
        images : str
            Path to the test images

        Example
        -------
        predictor.predict_mars_craters('test_images/')
        """
        self.yolo_detect(images, self.mars_model, name="detections")

    def yolo_detect(self, images, weights, name):
        """
        Run YOLOv5 detection

        Parameters
        ----------
        images : str
            Path to the test images
        weights : str
            Path to the model weights
        name : str
            Name of the output file. Detections
            are converted to to csv and moved here.

        Example
        -------
        predictor.yolo_detect('test_images/', 'mars.weights', 'detections')
        """
        detect.run(
            source=images,
            weights=weights,
            imgsz=self.img_size,
            save_txt=True,
            name=name,
        )

        self.convert_and_move_detections()

    def create_user_results(self):
        """
        Creates the directory specified in `results_path` if it does not exist.
        """
        if not os.path.exists(self.results_path):
            os.mkdir(self.results_path)

    def convert_and_move_detections(self):
        """
        Converts the YOLOv5 detections saved as txt files
        in the `runs/detect/detections/labels`
        directory to csv format and moves them to `results_path/detections'.
        """
        self.create_user_results()
        if os.path.exists(self.detections_path):
            shutil.rmtree(
                self.detections_path, ignore_errors=False, onerror=None)
        os.mkdir(self.detections_path)
        self.convert_all_detections_to_csv("runs/detect/detections/")

    def convert_yolo_txt_to_csv(self, txt_file, csv_file):
        """
        Converts a YOLOv5 txt file to a csv file.

        Parameters
        ----------
        txt_file : str
            Path to the txt file
        csv_file : str
            Path to the csv file

        """
        with open(txt_file, "r") as file:
            labels = file.readlines()

            csv_rows = []
            for line in [lb.strip() for lb in labels]:
                dims = line.split(" ")

                row = [
                    float(dims[1]), float(dims[2]),
                    float(dims[3]), float(dims[4])]

                csv_rows.append(row)

            with open(csv_file, "w", newline="") as file:
                writer = csv.writer(file)
                writer.writerows(csv_rows)

            os.remove(txt_file)

    def convert_all_detections_to_csv(self, txt_path):
        """
        Converts all txt files in the specified directory to csv format.

        Parameters
        ----------
        txt_path : str
            Path to the directory containing the txt files

        """
        txt_labels_path = txt_path + "labels"
        for filename in os.listdir(txt_labels_path):
            if filename.endswith(".txt"):
                # Get the file paths
                txt_file = os.path.join(txt_labels_path, filename)
                csv_file = os.path.join(
                    self.detections_path, filename.replace(".txt", ".csv")
                )

                # Call the conversion function
                self.convert_yolo_txt_to_csv(txt_file, csv_file)

        # remove the labels folder so the same name can
        # be used again for next run
        shutil.rmtree(txt_path, ignore_errors=False, onerror=None)

    def bounding_box(self, img, x, y, w, h, img_w, img_h, color):
        """
        Draw bounding boxes.

        Parameters
        ----------
        img : numpy.array
            The image read from cv2.imread()
        x : float
            x position of the crater centre
        y : float
            y position of the crater centre
        w : float
            Width of crater
        h : float
            Height of crater
        img_w: int
            Width of the image
        img_h: int
            Height of the image

        Returns
        -------
        numpy.array
            The image with bounding boxes drawn

        Examples
        ------
        >>> x = 0.5
        >>> y = 0.5
        >>> w = 0.1
        >>> h = 0.1
        >>> img_w = 416
        >>> img_h = 416
        >>> color = (0, 255, 0)
        >>> bounding_box(img, x, y, w, h, img_w, img_h, color)
        """
        x = x * img_w
        # w = w * img_w
        y = y * img_h
        # h = h * img_h
        cv2.rectangle(
            img,
            (int(x - w * img_w / 2), int(y - h * img_h / 2)),
            (int(x + w * img_w / 2), int(y + h * img_h / 2)),
            color,
            1,
        )
        # cv2.rectangle(img, (0, 0), (208, 208), (0, 255, 0), 1)

        return img

    def draw_boxes(self, label_path=None):
        """
        Draw bounding boxes on images,
        both ground truth (if provided) and detections.
        Results are saved in
          'results_path/images/detections' and
          'results_path/images/detections_and_gt'

        Parameters
        ----------
        label_path : str, optional
            The path to the folder containing ground truth bounding boxes.
            If not provided, only detections will be drawn.

        Example
        -------
        >>> draw_boxes()
        >>> draw_boxes(label_path='/path/to/ground_truth')
        """
        # set the pred labels path to where they were saved
        img_path = self.test_images_path

        if os.path.exists(self.images_path):
            shutil.rmtree(self.images_path, ignore_errors=False, onerror=None)

        os.mkdir(self.images_path)

        # make one folder for only detections
        os.mkdir(self.images_path + "detections/")

        # make one folder for both detections and ground truth
        os.mkdir(self.images_path + "detections_and_gt/")

        pred_label_path = self.detections_path

        # TODO: handle different image file types
        for file in os.listdir(img_path):
            if file.endswith(".png") or \
                file.endswith(".jpg") or \
                    file.endswith(".tif"):
                filetype = "png"

                if file.endswith(".jpg"):
                    filetype = "jpg"

                if file.endswith(".tif"):
                    filetype = "tif"

                file_path_img = f"{img_path}/{file}"
                img_single = cv2.imread(file_path_img)
                img_both = cv2.imread(file_path_img)
                img_w = self.img_size
                img_h = self.img_size

                if label_path:
                    label = pd.read_csv(
                        f"{label_path}/{file.replace(filetype, 'csv')}",
                        names=["x", "y", "w", "h"],
                    )

                    # draw the ground truth bounding boxes
                    for i in range(label.shape[0]):
                        x = label["x"][i]
                        y = label["y"][i]
                        w = label["w"][i]
                        h = label["h"][i]
                        img_both = self.bounding_box(
                            img_both, x, y, w, h, img_w, img_h, (0, 0, 255)
                        )

                try:
                    pred_label = pd.read_csv(
                        f"{pred_label_path}/{file.replace(filetype, 'csv')}",
                        names=["x", "y", "w", "h"],
                    )
                except Exception:
                    print(
                        "No detection found: \
                            there were no craters detected in this image"
                    )
                    # cv2.imshow(img)
                    # if no detection is found, only show the ground truth box
                    if label_path:
                        cv2.imwrite(
                            self.images_path + "detections_and_gt/" +
                            file, img_both
                        )
                        continue
                    else:
                        cv2.imwrite(
                            self.images_path + "detections/" +
                            file, img_single)
                        continue

                # print(label)

                # draw the detected bounding boxes
                for i in range(pred_label.shape[0]):
                    det_x = pred_label["x"][i]
                    det_y = pred_label["y"][i]
                    det_w = pred_label["w"][i]
                    det_h = pred_label["h"][i]

                    # print(det_x, det_y, det_w, det_h)

                    img_both = self.bounding_box(
                        img_both, det_x, det_y, det_w,
                        det_h, img_w, img_h, (255, 0, 0)
                    )
                    img_single = self.bounding_box(
                        img_single,
                        det_x,
                        det_y,
                        det_w,
                        det_h,
                        img_w,
                        img_h,
                        (255, 0, 0),
                    )
                # print(img)
                # cv2.imshow(img)
                if label_path:
                    cv2.imwrite(
                        self.images_path + "detections_and_gt/" +
                        file, img_both
                    )

                cv2.imwrite(
                    self.images_path + "detections/" + file, img_single)

    def get_statistics(self, label_path=None):
        """
        Calculate and write the true positive (tp),
        false positive (fp), and false negative (fn) values to a csv file.

        Parameters
        ----------
        label_path (str): The path to the directory containing
        the label csv files.
        If not specified, the function returns without
        performing any operations.
        """

        if os.path.exists(self.stats_path):
            shutil.rmtree(self.stats_path, ignore_errors=False, onerror=None)
        os.mkdir(self.stats_path)

        if not label_path:
            return

        ground_truth = self.csv_to_numpy(label_path)
        detections = self.csv_to_numpy(self.detections_path)

        iou_threshold = 0.5

        tp = 0
        fp = 0
        fn = 0

        for filename, labels in ground_truth.items():
            if filename not in detections:
                # no detections
                fn += len(labels)
                continue

            else:
                curr_tp, curr_fp, curr_fn = self.calculate_statistics(
                    detections[filename], labels, iou_threshold
                )
                tp += curr_tp
                fp += curr_fp
                fn += curr_fn

        # use the same filename
        filepath = os.path.join(self.stats_path, "statistics.csv")
        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["tp", "fp", "fn"])
            writer.writerow([tp, fp, fn])

    def csv_to_numpy(self, directory):
        """
        Convert all csv files in a directory into NumPy arrays.

        Parameters
        ----------
        directory (str): The path to the directory containing the csv files.

        Returns
        ----------
        arrays (dict): A dictionary where the keys are
        the filenames of the csv files and the values are
        the NumPy arrays of the contents of the csv files.
        """
        arrays = {}
        for filename in os.listdir(directory):
            if filename.endswith(".csv"):
                filepath = os.path.join(directory, filename)
                arrays[filename] = np.genfromtxt(filepath, delimiter=",")
        return arrays

    def calculate_iou(self, detected, ground_truth):
        """
        Calculate the Intersection over Union (IoU) between two bounding boxes.

        Parameters
        ----------
        detected (list): A list of 4 values representing
        the bounding box coordinates (x, y, width, height).
        ground_truth (list): A list of 4 values representing
        the bounding box coordinates (x, y, width, height).

        Returns
        ----------
        float: The calculated IoU value.

        """

        x1 = detected[0]
        y1 = detected[1]
        w1 = detected[2]
        h1 = detected[3]

        x2 = ground_truth[0]
        y2 = ground_truth[1]
        w2 = ground_truth[2]
        h2 = ground_truth[3]

        ax1 = x1 - w1 / 2
        ay1 = y1 + h1 / 2
        bx1 = x1 + w1 / 2
        by1 = y1 - h1 / 2
        ax2 = x2 - w2 / 2
        ay2 = y2 + h2 / 2
        bx2 = x2 + w2 / 2
        by2 = y2 - h2 / 2

        cox1 = max(ax1, ax2)
        cox2 = min(bx1, bx2)
        coy1 = min(ay1, ay2)
        coy2 = max(by1, by2)

        dx = abs(cox2 - cox1)
        dy = abs(coy2 - coy1)

        area_overlap = 0
        area_overlap = max(0, dx) * max(0, dy)

        area_a = w1 * h1
        area_b = w2 * h2

        area_all = area_a + area_b - area_overlap
        iou = area_overlap / area_all

        return iou

    def calculate_statistics(self, detections, ground_truths, threshold):
        """
        Calculate true positive, false positive,
        and false negative values using IoU.

        Parameters
        ----------
        detections (numpy.ndarray or list): A list or Numpy array of
        bounding box coordinates (x, y, width, height)
        for each detected object.
        ground_truths (numpy.ndarray or list): A list or Numpy array of
        ground truth bounding box coordinates (x, y, width, height)
        for each object.
        threshold (float): The IoU threshold value to consider
        a detection as a true positive.

        Returns
        ----------
        tuple: A tuple of 3 values representing true positive (int),
        false positive (int), and false negative (int) values.

        """
        tp = 0
        fp = 0
        fn = 0

        if detections.ndim == 1:
            detections = detections[np.newaxis, :]

        if ground_truths.ndim == 1:
            ground_truths = ground_truths[np.newaxis, :]

        detections = detections.tolist()
        ground_truths = ground_truths.tolist()

        detected = [False] * len(ground_truths)

        for d in detections:
            max_iou = 0
            best_match = -1

            for i, gt in enumerate(ground_truths):
                iou = self.calculate_iou(d, gt)
                if iou > max_iou:
                    max_iou = iou
                    best_match = i

            if max_iou >= threshold:
                if not detected[best_match]:
                    tp += 1
                    detected[best_match] = True
                else:
                    fp += 1
            else:
                fp += 1

        fn = np.sum(np.logical_not(detected))

        return tp, fp, fn

    def idx_labels(self, det_path):
        """
        Add crater size and physical locations of the craters
        to the label csv files

        Parameter
        ----------
        det_path: str
            Detection path containing information of detected craters
        """
        for file in os.listdir(det_path):
            info = file[:-4].split("-")
            img_area, img_a, img_b = info[0], int(info[1]), int(info[2])
            df = pd.read_csv(det_path + "/" + file, names=["x", "y", "w", "h"])
            # print(type(df))
            # print(type(df.iloc[0]))
            lon, lat, crater_size = analysis.convert2physical(
                [df["x"], df["y"], df["w"], df["h"]],
                [416.0, 416.0],
                [img_area, img_a, img_b],
            )
            print(lat)
            print(lon)
            print(crater_size)
            df["lon"] = lon
            df["lat"] = lat
            df["crater_size"] = crater_size

            df.to_csv(det_path + "/" + "calc" + file, index=False)
