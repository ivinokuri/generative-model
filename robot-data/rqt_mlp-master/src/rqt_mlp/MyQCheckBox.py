from python_qt_binding.QtWidgets import QCheckBox


class MyQCheckBox(QCheckBox):
    def __init__(self, title, bagParser, selection_vlayout, selected_list):
        QCheckBox.__init__(self, title, bagParser)
        self.selection_vlayout = selection_vlayout
        self.selected_list = selected_list