from tkinter import (
    Frame,
    StringVar,
    OptionMenu,
    Button,
    LEFT,
    RIGHT,
    BOTTOM,
    Label,
    Entry,
    PhotoImage,
    Tk,
    filedialog,
)
from crater_predictor import CraterPredictor
import pandas as pd


class GUI(Frame):
    """
    This class creates a graphical user interface for a project,
allowing the user to interact with input options.

    Parameters:
    master (Tk, optional): The parent widget for the GUI. Default is None.

    Attributes:
    option_frame (Frame): The frame for user-defined hyperparameters.
    model_options (list): The list of available models.
    model_var (StringVar): The variable storing the selected model.
    model_menu (OptionMenu):
    The dropdown menu for selecting models.
    model_var_button (Button): The button for displaying
    the selected model value.
    model_var_label (Label): The label displaying the selected model value.
    data_options (list): The list of available data options.
    data_var (StringVar): The variable storing the selected data option.
    data_menu (OptionMenu): The dropdown menu for selecting data options.
    test_folder_frame (Frame): The frame for the test folder path.
    test_folder_path (Button): The button for choosing the test folder.
    res_dir_frame (Frame): The frame for the result directory.
    e1_label (Label): The label for the result directory.
    result_dir (StringVar): The variable storing the result directory path.
    e1 (Entry): The entry widget for the result directory path.
    detect_frame (Frame): The frame for the detect button.
    detect_btn (Button): The button for starting detection.
    file_frame (Frame): The frame for the file browse button.
    label (Label): The label for displaying the selected file.
    file (Button): The button for browsing and selecting a file.
    exit_frame (Frame): The frame for the exit button.
    exit (Button): The button for exiting the program.

    Methods:
    show_value: Displays the selected model value.
    choose_folder: Chooses the test folder path.
    detect: Starts detection.
    choose: Allows the user to browse and select a file.
    """
    def __init__(self, master=None):
        Frame.__init__(self, master)
        # w, h = 650, 650
        w = root.winfo_screenwidth()
        h = root.winfo_screenheight()
        master.minsize(width=w, height=h)
        master.maxsize(width=w, height=h)
        self.pack()

        # user defined hyperparameters frame
        self.option_frame = Frame(master)
        self.option_frame.pack()

        self.model_options = ["Yolov5", "Yolov8"]
        self.model_var = StringVar(self.option_frame)
        # self.model_var_label = Label(self.option_frame)
        # default value
        self.model_var.set(self.model_options[0])
        self.model_menu = OptionMenu(
            self.option_frame, self.model_var, *self.model_options
        )
        self.model_menu.pack(side=LEFT)
        self.model_var_button = Button(
            self.option_frame, text="Show selected values",
            command=self.show_value)
        self.model_var_button.pack(side=RIGHT)

        self.model_var_label = Label(self.option_frame)
        self.model_var_label.pack(side=RIGHT)

        self.data_options = ["Mars", "Moon"]
        self.data_var = StringVar(self.option_frame)
        # default value
        self.data_var.set(self.data_options[0])
        self.data_menu = OptionMenu(
            self.option_frame, self.data_var, *self.data_options
        )
        self.data_menu.pack(side=LEFT)

        # group test folder path to a frame
        self.test_folder_frame = Frame(master)
        self.test_folder_frame.pack()
        self.test_folder_path = Button(
            self.test_folder_frame, text="Test folder",
            command=self.choose_folder)
        self.test_folder_path.pack(side=LEFT)

        # pack to result directory frame
        self.res_dir_frame = Frame(master)
        self.res_dir_frame.pack()
        # user defined directory for result
        self.e1_label = Label(
            self.res_dir_frame, text="Result directory").pack(
            side=LEFT)
        self.result_dir = StringVar(master)
        self.e1 = Entry(self.res_dir_frame, textvariable=self.result_dir)
        self.e1.pack(side=RIGHT)

        # detect frame
        self.detect_frame = Frame(master)
        self.detect_frame.pack()
        self.detect_btn = Button(
            self.detect_frame, text="Detect", command=self.detect)
        self.detect_btn.pack(side=LEFT)

        # compute physical location and crater size
        self.analysis_frame = Frame(master)
        self.analysis_frame.pack()
        self.analysis_btn = Button(
            self.analysis_frame,
            text="Compute crater size and location",
            command=self.analysis,
        )
        self.analysis_btn.pack()

        # file frame
        self.file_frame = Frame(master)
        self.file_frame.pack()
        self.label = Label(self.file_frame)
        self.file = Button(self.file_frame, text="Browse", command=self.choose)
        self.file.pack()
        self.label.pack()

        # exit btn
        self.exit_frame = Frame(master)
        self.exit_frame.pack()
        self.exit = Button(self.exit_frame, text="Exit", command=root.destroy)
        self.exit.pack()

    def show_value(self):
        """
        This function is used to show what kinds of values users
        choose in the interface.
        """
        self.model_var_label.config(
            text="Selected model: "
            + self.model_var.get()
            + "\nSelected data: "
            + self.data_var.get()
        )

    def choose(self):
        """
        This function is used to choose the specified image file.
        """
        ifile = filedialog.askopenfile(
            parent=self, mode="rb", title="Choose a file")

        self.image = PhotoImage(file=ifile.name)
        self.label.configure(image=self.image)
        self.label.image = self.image

    def choose_folder(self):
        """
        This function is used to choose the specified folder.
        """
        path = filedialog.askdirectory()
        self.test_folder_var = path
        Label(
            self.test_folder_frame, text=self.test_folder_var).pack(side=RIGHT)

    def detect(self):
        """
        This function is used to generate results from user's inputs.
        """
        # get model to detect
        self.test_img_path = self.test_folder_var + "/images/"
        self.test_labels_path = self.test_folder_var + "/labels/"

        self.user_specified_directory = self.result_dir.get()
        # yolov5
        mars_model_path = "models/Mars_best.pt"
        moon_model_left_path = "models/Moon_Left_Model.pt"
        moon_model_right_path = "models/Moon_Right_Model.pt"

        if self.data_var.get() == "Mars":
            self.craterPredictor = CraterPredictor(
                mars_model=mars_model_path,
                moon_model1=moon_model_left_path,
                moon_model2=moon_model_right_path,
                results_path=self.user_specified_directory,
                test_images_path=self.test_img_path,
            )
            # predict with mars
            self.craterPredictor.predict_mars_craters(self.test_img_path)
        else:
            self.craterPredictor = CraterPredictor(
                mars_model=mars_model_path,
                moon_model1=moon_model_left_path,
                moon_model2=moon_model_right_path,
                results_path=self.user_specified_directory,
                test_images_path=self.test_img_path,
            )
            # predict with moons
            # crop_func()
            self.craterPredictor.predict_moon_craters(self.test_img_path)
        # craterPredictor.draw_boxes()
        self.craterPredictor.draw_boxes(self.test_labels_path)
        # craterPredictor.get_statistics()
        self.craterPredictor.get_statistics(self.test_labels_path)
        self.detect_btn_label = Label(
            self.detect_frame, text="Detection complete"
        ).pack(side=BOTTOM)

        stats_path = self.user_specified_directory + \
            "/statistics/statistics.csv"
        df = pd.read_csv(stats_path)
        self.stats = Label(
            self.detect_frame,
            text=f"Statistics\ntp: {df['tp'][0]}"
            + f"\nfp: {df['fp'][0]}"
            + f"\nfn: {df['fn'][0]}",
        ).pack()

    def analysis(self):
        self.craterPredictor.idx_labels(
            self.user_specified_directory + "detections/")
        Label(text="Analysis computed").pack()


root = Tk()
root.title("Crater prediction")
app = GUI(master=root)
app.mainloop()
