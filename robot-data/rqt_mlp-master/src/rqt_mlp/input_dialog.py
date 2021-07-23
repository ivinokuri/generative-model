import sys
from python_qt_binding.QtCore import Qt, Signal
from python_qt_binding.QtWidgets import QApplication, QMessageBox, QFormLayout, QLineEdit, QLabel, QWidget, QPushButton

NO_COND = -9999
class inputDialog(QWidget):

    ParamsSelected = Signal(list, list, int)

    def __init__(self, params, id_radio):
        super(inputDialog, self).__init__()

        self.id_radio = id_radio
        self.center()
        layout = QFormLayout()

        self.label_items = []
        self.label_comment = dict()
        self.client_answers_items = []
        self.greather_conditions = []
        self.button_clicked = False
        # print params
        for item in params:
            param = item[0]
            label = item[1]
            cond = item[2]
            if not label == "":
                label = " (%s)" % label
            self.greather_conditions.append((param, cond))
            label_item = QLabel(param + label, self)
            # self.label_items[param] = QLabel(param + label, self)
            label_item.setAlignment(Qt.AlignCenter)
            self.label_items.append(param)
            client_answers = QLineEdit()
            self.client_answers_items.append(client_answers)
            # self.label_comment[label] = QLabel(label, self)
            # self.label_comment[label].setAlignment(Qt.AlignCenter)
            # layout.addWidget(self.label_comment[label])
            layout.addRow(label_item, client_answers)

        self.ok_button = QPushButton("ok", self)
        self.ok_button.clicked.connect(self.onButtonClicked)

        layout.addWidget(self.ok_button)

        self.setLayout(layout)
        self.setWindowTitle("Update Parameters")

        self.show()

    def closeEvent(self, event):
        if not self.button_clicked:
            self.ParamsSelected.emit([], [], 0)
            print "User has clicked the red x on the main window"

    def center(self):
        frameGm = self.frameGeometry()
        screen = QApplication.desktop().screenNumber(QApplication.desktop().pos())
        centerPoint = QApplication.desktop().screenGeometry(screen).center()
        frameGm.moveCenter(centerPoint)
        self.move(frameGm.topLeft())

    def onButtonClicked(self):
        counter = 0
        for item in self.client_answers_items:
            try:
                val = float(item.text())
            except ValueError:
                QMessageBox.about(self, "Error in data", "That's not a number!")
                return
            if item.text() == "":
                QMessageBox.about(self, "Missing data", "Fill the missing data")
                return
            condition = self.greather_conditions[counter][1]
            param = self.greather_conditions[counter][0]
            print "condition = %s param = %s val = %s" % (condition, param, val)
            # print param

            if not condition == NO_COND:
                if val <= float(condition):
                    QMessageBox.about(self, "Missing match condition", "the condition must be satisfied: %s > %s" % (param, condition))
                    return
            counter = counter + 1
        self.button_clicked = True
        self.close()
        self.ParamsSelected.emit(self.client_answers_items, self.label_items, self.id_radio)
