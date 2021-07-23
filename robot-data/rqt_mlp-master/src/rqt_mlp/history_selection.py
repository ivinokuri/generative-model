import TimeSeriesFeatures as TS
from python_qt_binding.QtCore import Qt, Signal
from python_qt_binding.QtWidgets import QHBoxLayout, QFormLayout, QMessageBox, QFileDialog, QLineEdit, QRadioButton, QButtonGroup, \
    QLabel, QWidget, QVBoxLayout, QCheckBox, QScrollArea, QPushButton
from MyQCheckBox import MyQCheckBox
import logging

# save choose of user
# logger_topic = logging.getLogger("logger_history")

class HistorySelection(QWidget):
    def __init__(self):
        super(HistorySelection, self).__init__()
        self.setWindowTitle("Extracting Window Features")
        self.resize(550, 440)

        per_title, pre = TS.get_time_series_pre_feature_options()
        glob_title, glob = TS.get_global_time_series_features_options()

        self.history_items = dict()
        self.history_items[per_title] = pre
        self.history_items[glob_title] = glob

        self.saved_dir = []

        self.group_selected_items = dict()
        self.group_areas = dict()
        self.group_main_widget = dict()
        self.group_selection_vlayout = dict()
        self.group_item_all = dict()
        self.main_vlayout = QVBoxLayout(self)
        self.group_label = dict()

        self.items_list = []
        self.selected_topics = []
        self.files = []

        # layout = QFormLayout()

        self.select_path = QLineEdit()
        self.save_path = QLineEdit()

        self.select_path.setEnabled(False)
        self.save_path.setEnabled(False)

        self.ok_button = QPushButton("Select CSV...", self)
        self.ok_button.clicked.connect(self.onButtonClicked)

        self.two_buttons1 = QHBoxLayout(self)

        self.two_buttons1.addWidget(self.ok_button)

        self.two_buttons1.addWidget(self.select_path)

        self.main_vlayout.addLayout(self.two_buttons1)

        # self.main_vlayout.addRow(self.ok_button, self.client_answers)

        # self.main_vlayout.addWidget(self.ok_button)

        # self.main_vlayout.addWidget(self.ok_button)

        for group_name in self.history_items:
            self.group_selected_items[group_name] = []
            self.group_areas[group_name] = QScrollArea(self)
            self.group_main_widget[group_name] = QWidget(self.group_areas[group_name])
            self.group_label[group_name] = QLabel(group_name, self)
            self.group_label[group_name].setAlignment(Qt.AlignCenter)
            self.main_vlayout.addWidget(self.group_label[group_name])
            self.main_vlayout.addWidget(self.group_areas[group_name])
            self.group_selection_vlayout[group_name] = QVBoxLayout(self)
            self.group_item_all[group_name] = MyQCheckBox("All", self, self.group_selection_vlayout[group_name], None)
            self.MakeCheckBoxList(self.group_selection_vlayout[group_name],
                                  self.group_selected_items[group_name],
                                  self.history_items[group_name],
                                  self.group_item_all[group_name])
            self.group_main_widget[group_name].setLayout(self.group_selection_vlayout[group_name])
            self.group_areas[group_name].setWidget(self.group_main_widget[group_name])

        self.clear_button = QPushButton("Clear Selection", self)
        self.clear_button.clicked.connect(self.onClearClicked)

        self.choose_button = QPushButton("Get Last Export Choose", self)
        self.choose_button.clicked.connect(self.onButtonChooseCliked)



        # self.main_vlayout.addWidget(self.choose_button)

        self.label4 = QLabel("Window time", self)
        self.label4.setAlignment(Qt.AlignCenter)

        # self.main_vlayout.addWidget(self.label4)

        self.window = QLineEdit(self)
        # self.main_vlayout.addWidget(self.window)
        self.window.setText("3")

        self.label5 = QLabel("Duration Time:", self)
        self.label5.setAlignment(Qt.AlignCenter)

        self.label6 = QLabel("Step:", self)
        self.label6.setAlignment(Qt.AlignCenter)

        self.step = QLineEdit(self)
        self.step.setText("1")

        self.windows_time_3 = QHBoxLayout(self)

        self.windows_time_3.addWidget(self.label4)

        self.windows_time_3.addWidget(self.window)

        self.windows_time_3.addWidget(self.label5)

        self.main_vlayout.addLayout(self.windows_time_3)

        self.step_2 = QHBoxLayout(self)

        self.step_2.addWidget(self.label6)

        self.step_2.addWidget(self.step)

        self.main_vlayout.addLayout(self.step_2)


        # self.main_vlayout.addRow(self.label4, self.window)

        self.two_buttons = QHBoxLayout(self)

        self.two_buttons.addWidget(self.choose_button)

        self.two_buttons.addWidget(self.clear_button)

        self.main_vlayout.addLayout(self.two_buttons)

        # self.main_vlayout.addRow(self.clear_button, self.choose_button)

        # self.label5 = QLabel("Load CSV", self)
        # self.label5.setAlignment(Qt.AlignCenter)
        #
        # self.main_vlayout.addWidget(self.label5)

        # self.label6 = QLabel("output", self)
        # self.label6.setAlignment(Qt.AlignCenter)
        #
        # self.main_vlayout.addWidget(self.label6)

        self.save_button = QPushButton("Save CSV...", self)
        self.save_button.clicked.connect(self.onSaveClicked)

        self.submit_button = QPushButton("Sumbit", self)
        self.submit_button.clicked.connect(self.onSumbitClicked)

        # self.main_vlayout.addWidget(self.save_button)

        self.two_buttons2 = QHBoxLayout(self)

        self.two_buttons2.addWidget(self.save_button)

        self.two_buttons2.addWidget(self.save_path)

        self.main_vlayout.addLayout(self.two_buttons2)


        # self.main_vlayout.addRow(self.save_button, self.client_answers1)

        self.main_vlayout.addWidget(self.submit_button)

        self.show()


    def get_min_rows_csv(self, csv_files):
        import pandas as pd
        dfs = map(lambda x: pd.read_csv(x, header=0), csv_files)
        number = dfs[0].shape[0]
        number = reduce(lambda acc,curr: min(acc,curr.shape[0]), dfs, number)
	self.label5.setText("Duration Time: %s" % number)
        
    def onClearClicked(self):
        self.clearTopicCheckState()

    def clearTopicCheckState(self):
        for item in self.items_list:
            item.setCheckState(False)
        for item in self.group_item_all.values():
            item.setCheckState(False)

    def MakeCheckBoxList(self, selection_vlayout, selected, topics_Keys, item_all):
        item_all.stateChanged.connect(lambda x: self.updateList(x, item_all, None))
        selection_vlayout.addWidget(item_all)
        topic_data_list = topics_Keys
        topic_data_list.sort()
        for topic in topic_data_list:
            self.addCheckBox(topic, selection_vlayout, selected)

    def addCheckBox(self, topic, selection_vlayout, selected_list):
        item = MyQCheckBox(topic, self, selection_vlayout, selected_list)
        item.stateChanged.connect(lambda x: self.updateList(x, item, topic))
        self.items_list.append(item)
        selection_vlayout.addWidget(item)

    def updateList(self, state, item_clicked, topic=None, force_update_state=False):
        if type(item_clicked) is str:
            item_clicked = self.get_item_by_name(item_clicked)
            if item_clicked is None:
                return
        if topic is None:  # The "All" checkbox was checked / unchecked
            # print "if topic is None"
            if state == Qt.Checked:
                # self.item_all.setTristate(False)
                for item in self.items_list:
                    if item.checkState() == Qt.Unchecked and \
                                    item.selection_vlayout == item_clicked.selection_vlayout:
                        item.setCheckState(Qt.Checked)
            elif state == Qt.Unchecked:
                # self.item_all.setTristate(False)
                for item in self.items_list:
                    if item.checkState() == Qt.Checked and \
                                    item.selection_vlayout == item_clicked.selection_vlayout:
                        item.setCheckState(Qt.Unchecked)
        else:
            if state == Qt.Checked:
                item_clicked.selected_list.append(topic)
                # print item_clicked.selected_list
            else:
                item_clicked.selected_list.remove(topic)
                # if self.item_all.checkState() == Qt.Checked:
                #    self.item_all.setCheckState(Qt.PartiallyChecked)

    def get_current_opened_directory(self, filepath):
        import os
        direc = "/"
        if os.path.isfile(filepath):
            with open(filepath, 'r') as f:
                direc = f.read()
        return direc

    def onButtonClicked(self):
        import inspect, os
        filepath = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) + "/log/select_history.log"
        fd = QFileDialog(self)
        wc = "Csv files {.csv} (*.csv)"
        self.files = []
        current_directory = self.get_current_opened_directory(filepath)
        # for ubuntu 14
        tmp_pathes, filter = fd.getOpenFileNamesAndFilter(filter=wc, initialFilter=('*.csv'),
                                                          directory=current_directory)
        # tmp_pathes, filter = fd.getOpenFileNames(filter=wc, initialFilter=('*.csv'),
        #                                                   directory=current_directory)
        for path in tmp_pathes:
            self.files.append(path.encode("utf-8"))
        # print self.files
        if len(self.files) != 0:
            self.get_min_rows_csv(tmp_pathes)
            self.select_path.setText(self.files[0])
            with open(filepath, "w") as f:
                f.write(self.files[0])

                # handler = logging.FileHandler(filepath, mode='w')
                # logger_topic.addHandler(handler)
                # logger_topic.info(self.files[0])
        else:
            self.select_path.setText("")

    def check_int(self, number, condition, title, message_body):
        try:
            # print type(number)
            val = int(number)
            if val <= condition:
                QMessageBox.about(self, title, "The number should > %s" % condition)
                return False
        except ValueError:
            QMessageBox.about(self, title, message_body)
            return False
        return True

    def check_choose(self):
        flag = True
        temp = []
        for item in self.group_selected_items.values():
            if item:
                for i in item:
                    temp.append(i)
        if len(temp) == 0:
            QMessageBox.about(self, "Features", "One feature at least should be chosen")
            flag = False
        return flag

    def check_files_amount(self):
        flag = True
        if len(self.files) <= 0:
            QMessageBox.about(self, "Load CSV", "One file at least should be chosen")
            flag = False
        return flag

    def check_validation(self):
        flag = self.check_files_amount()
        flag = flag & self.check_choose()
        flag = flag & self.check_int(self.window.text(), 2, "Error in Window Time", "That's not a number!")
        flag = flag & self.check_int(self.step.text(), 0, "Error in Step", "That's not a integer!")
        # TODO selected topic not empty

        if self.saved_dir == []:
            QMessageBox.about(self, "Save CSV", "Select path for saving")
            flag = False
        return flag

    def onSumbitClicked(self):
        if not self.check_validation():
            return
        self.selected_topics = []
        for item in self.group_selected_items.values():
            if item:
                for i in item:
                    self.selected_topics.append(i)
        # Defined Logging
        # handler = logging.FileHandler('/var/tmp/logger_history.log', mode='a')
        # logger_topic.addHandler(handler)
        topics = self.selected_topics
        with open(get_path() + 'logger_history.log', "w") as f:
            for topic in topics:
                f.write(topic + "\n")
        self.createTimeSeriesFeatures(self.files, self.saved_dir, int(self.window.text()),
                                      self.group_selected_items, int(self.step.text()))

    def onSaveClicked(self):
        import inspect, os
        filepath = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) + "/log/save_history.log"
        # print self.files
        # print self.window.text()
        # print self.group_selected_items.values()

        # print self.group_selected_items
        current_directory = self.get_current_opened_directory(filepath)
        # print current_directory
        # self._to_save_filename = QFileDialog.getSaveFileName(self, self.tr('csv File'), current_directory,
        #                                                      self.tr('csv (*.csv)'))
        self.saved_dir = str(QFileDialog.getExistingDirectory(self, "Select Directory", current_directory))
        # if self._to_save_filename[0] != "":
        #     with open(filepath, "w") as f:
        #         f.write(self._to_save_filename[0])
        if self.saved_dir != "":
            with open(filepath, "w") as f:
                f.write(self.saved_dir)

                # handler = logging.FileHandler(filepath, mode='w')
                # logger_topic.addHandler(handler)
                # print self._to_save_filename[0]
                # logger_topic.info()
        # self.save_path.setText(get_corrent_file_name(self._to_save_filename[0], ".csv"))
        self.save_path.setText(self.saved_dir)


    def createTimeSeriesFeatures(self, files, saved_dir, window, group_selected_items, step):
        import TimeSeriesFeatures as TS
        # saved_dir = saved_dir[0].encode('utf-8')
        for input_path in files:
            output_path = generate_csv_from_bag(saved_dir, input_path)
            print "++++++++++++++++++++++++++ ", output_path
            print "in = %s out = %s " % (input_path, output_path)
            print "step = %s" % step
            ts = TS.TimeSeries(input_path, output_path, window, group_selected_items, step)
            ts.generate_time_series_features()
        QMessageBox.about(self, "csv save", "csv was saved successfuly")

    def onButtonChooseCliked(self):
        import os
        for checkbox in self.items_list:
            checkbox.setCheckState(Qt.Unchecked)
        if os.path.isfile(get_path() + "logger_history.log"):
            with open(get_path() + "logger_history.log", 'r') as f:
                topics = f.read().splitlines()
            for checkbox in self.items_list:
                if checkbox.text() in topics:
                    checkbox.setCheckState(Qt.Checked)

# def generate_csv_from_bag(output_path, input_path):
#     bag_split = output_path.split('/')
#     csv_split = input_path.split('/')
#     bag_split[len(bag_split) - 1] = "ts_" + csv_split[len(csv_split) - 1]
#     tmp =  reduce(lambda acc,curr: acc + "/" + curr , bag_split, "")
#     tmp = tmp[:-4] + ".csv"
#     return tmp

def generate_csv_from_bag(saved_dir, input_path):
    bag_split = input_path.split('/')
    csv_path = saved_dir + "/ts_" + bag_split[len(bag_split) - 1]
    csv_path = csv_path[:-4] + ".csv"
    print saved_dir
    print input_path
    print csv_path
    return csv_path

# def get_corrent_file_name(filename, suffix, i=-1):
#     if filename == "":
#         return ""
#     file_suffix = filename[-len(suffix):]
#     if file_suffix == suffix and i >= 0:
#         ret = "%s_%s%s" % (filename[:-len(suffix)], i, suffix)
#     elif file_suffix != suffix and i >= 0:
#         ret = "%s_%s%s" % (filename, i, suffix)
#     elif file_suffix != suffix and i < 0:
#         ret = "%s%s" % (filename, suffix)
#     else:
#         ret = filename
#     return ret
  
def get_path():
  import inspect, os
  import logging
  logging_file = os.path.dirname(
      os.path.abspath(inspect.getfile(inspect.currentframe()))) + "/Scenarios/History_Choosing/"
  return logging_file
  