import csv
import os
import shutil
from sklearn.model_selection import train_test_split


class DataProcessor():

    def __init__(self, image_data_path, label_data_path):
        """
        DataProcessor class with image and label data paths.

        Parameters
        ----------
        image_data_path (str): path to the folder containing the image data.
        label_data_path (str): path to the folder containing the label data.
        """
        self.image_data_path = image_data_path
        self.label_data_path = label_data_path

    def convert_csv_to_yolo_txt(self, csv_file):
        """
        Convert a CSV file to YOLO format text file.

        Parameters
        ----------
        csv_file (str): path to the CSV file to be converted.
        """

        # Open the CSV file for reading
        with open(csv_file, 'r') as file:
            reader = csv.reader(file)

            txt_file = csv_file.replace('.csv', '.txt')
            with open(txt_file, 'w') as out_file:
                for row in reader:
                    # add class 0 to the first column
                    out_file.write('0 ' + ' '.join(row[:4]) + '\n')

            os.remove(csv_file)

    def convert_all_labels_to_txt(self):
        """
        Convert a CSV file with header to YOLO format text file.

        Parameters
        ----------
        csv_file (str): path to the CSV file with header to be converted.
        """
        for filename in os.listdir(self.label_data_path):
            if filename.endswith('.csv'):
                self.convert_csv_to_yolo_txt(self.label_data_path + filename)

    def create_data_directories(self, destination_root):
        """
        Create directories for train, validation and test datasets.

        Parameters
        ----------
        destination_root (str): root directory where train,
        validation and test datasets will be stored.
        """
        os.mkdir(destination_root)

        image_folder = destination_root + "images/"
        labels_folder = destination_root + "labels/"

        train_images = destination_root + "images/train/"
        train_labels = destination_root + "labels/train/"

        val_images = destination_root + "images/val/"
        val_labels = destination_root + "labels/val/"

        test_images = destination_root + "images/test/"
        test_labels = destination_root + "labels/test/"

        os.mkdir(image_folder)
        os.mkdir(labels_folder)

        os.mkdir(train_images)
        os.mkdir(train_labels)

        os.mkdir(val_images)
        os.mkdir(val_labels)

        os.mkdir(test_images)
        os.mkdir(test_labels)

    def split_and_move_data(self, destination_root):
        """
        Split and move image and label data to train,
        validation and test directories.

        Parameters
        ----------
        destination_root (str): root directory where train,
        validation and test datasets will be stored.
        """
        images = [
            os.path.join(
                self.image_data_path, x
            ) for x in os.listdir(self.image_data_path)
        ]

        labels = [
            os.path.join(
                self.label_data_path, x
            ) for x in os.listdir(self.label_data_path) if x[-3:] == "txt"
        ]

        images.sort()
        labels.sort()

        train_images, val_images, train_labels, val_labels = train_test_split(
            images,
            labels,
            test_size=0.2,
            random_state=1
        )

        val_images, test_images, val_labels, test_labels = train_test_split(
            val_images,
            val_labels,
            test_size=0.5,
            random_state=1
        )

        self.move_files_to_folder(
            train_images, destination_root + 'images/train')
        self.move_files_to_folder(
            val_images, destination_root + 'images/val')
        self.move_files_to_folder(
            test_images, destination_root + 'images/test')
        self.move_files_to_folder(
            train_labels, destination_root + 'labels/train')
        self.move_files_to_folder(
            val_labels, destination_root + 'labels/val')
        self.move_files_to_folder(
            test_labels, destination_root + 'labels/test')

    def move_files_to_folder(self, files, dest):
        """
        Move files between two folders

        Parameters
        ----------
        files (str): from folder
        dest (str): to folder
        """
        print('moving files into ' + dest)
        for f in files:
            try:
                shutil.copy2(f, dest)
            except Exception:
                print(f + "could not be moved")
