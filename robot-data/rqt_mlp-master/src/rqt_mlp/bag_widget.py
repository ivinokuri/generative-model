import os
import time

import rospy
import rospkg

from python_qt_binding import loadUi
from python_qt_binding.QtCore import Qt, qWarning, Signal
from python_qt_binding.QtGui import QIcon
from python_qt_binding.QtWidgets import QMessageBox, QTreeView, QPushButton, QFileDialog, QGraphicsView, QWidget, QPlainTextEdit

import rosbag
from rqt_bag import bag_helper
from .bag_timeline import BagTimeline
from .topic_selection import TopicSelection
from .history_selection import HistorySelection
from .bag_parser import BagParser
import RunScenarios as S

class BagGraphicsView(QGraphicsView):
    def __init__(self, parent=None):
        super(BagGraphicsView, self).__init__()


class BagWidget(QWidget):
    """
    Widget for use with Bag class to display and replay bag files
    Handles all widget callbacks and contains the instance of BagTimeline for storing visualizing bag data
    """

    set_status_text = Signal(str)

    def __init__(self, context, publish_clock):
        """
        :param context: plugin context hook to enable adding widgets as a ROS_GUI pane, ''PluginContext''
        """
        super(BagWidget, self).__init__()
        rp = rospkg.RosPack()
        ui_file = os.path.join(rp.get_path('rqt_mlp'), 'resource', 'bag_widget.ui')
        loadUi(ui_file, self, {'BagGraphicsView': BagGraphicsView})

        self.setObjectName('BagWidget')

        self._timeline = BagTimeline(context, publish_clock)
        self.graphics_view.setScene(self._timeline)

        self.graphics_view.resizeEvent = self._resizeEvent
        self.graphics_view.setMouseTracking(True)



        self.play_icon = QIcon.fromTheme('media-playback-start')
        self.pause_icon = QIcon.fromTheme('media-playback-pause')
        # self.play_button.setIcon(self.play_icon)
        # self.begin_button.setIcon(QIcon.fromTheme('media-skip-backward'))
        # self.end_button.setIcon(QIcon.fromTheme('media-skip-forward'))
        # self.slower_button.setIcon(QIcon.fromTheme('media-seek-backward'))
        # self.faster_button.setIcon(QIcon.fromTheme('media-seek-forward'))
        # self.previous_button.setIcon(QIcon.fromTheme('go-previous'))
        # self.next_button.setIcon(QIcon.fromTheme('go-next'))
        # self.zoom_in_button.setIcon(QIcon.fromTheme('zoom-in'))
        # self.zoom_out_button.setIcon(QIcon.fromTheme('zoom-out'))
        # self.zoom_all_button.setIcon(QIcon.fromTheme('zoom-original'))
        # self.thumbs_button.setIcon(QIcon.fromTheme('insert-image'))
        self.record_button.setIcon(QIcon.fromTheme('media-record'))
        # self.history_button.setIcon(QIcon.fromTheme('insert-image'))
        import inspect
        path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) + "/extract-features-icon.gif"
        icon = QIcon(path)
        self.history_button.setIcon(QIcon(icon))
        self.load_button.setIcon(QIcon.fromTheme('document-open'))
        self.restart_button.setIcon(QIcon.fromTheme('view-refresh'))
        self.restart_button.setEnabled(False)
        # self.save_button.setIcon(QIcon.fromTheme('document-save'))

        # self.play_button.clicked[bool].connect(self._handle_play_clicked)
        # self.thumbs_button.clicked[bool].connect(self._handle_thumbs_clicked)
        # self.zoom_in_button.clicked[bool].connect(self._handle_zoom_in_clicked)
        # self.zoom_out_button.clicked[bool].connect(self._handle_zoom_out_clicked)
        # self.zoom_all_button.clicked[bool].connect(self._handle_zoom_all_clicked)
        # self.previous_button.clicked[bool].connect(self._handle_previous_clicked)
        # self.next_button.clicked[bool].connect(self._handle_next_clicked)
        # self.faster_button.clicked[bool].connect(self._handle_faster_clicked)
        # self.slower_button.clicked[bool].connect(self._handle_slower_clicked)
        # self.begin_button.clicked[bool].connect(self._handle_begin_clicked)
        # self.end_button.clicked[bool].connect(self._handle_end_clicked)
        self.record_button.clicked[bool].connect(self._handle_record_clicked)
        self.history_button.clicked[bool].connect(self._handle_history_clicked)
        self.load_button.clicked[bool].connect(self._handle_load_clicked)
        self.restart_button.clicked[bool].connect(self._handle_restart_clicked)
        # self.save_button.clicked[bool].connect(self._handle_save_clicked)
        self.graphics_view.mousePressEvent = self._timeline.on_mouse_down
        self.graphics_view.mouseReleaseEvent = self._timeline.on_mouse_up
        self.graphics_view.mouseMoveEvent = self._timeline.on_mouse_move
        self.graphics_view.wheelEvent = self._timeline.on_mousewheel
        self.closeEvent = self.handle_close
        self.keyPressEvent = self.on_key_press
        # TODO when the closeEvent is properly called by ROS_GUI implement that event instead of destroyed
        self.destroyed.connect(self.handle_destroy)

        self.graphics_view.keyPressEvent = self.graphics_view_on_key_press
        # self.play_button.setEnabled(False)
        # self.thumbs_button.setEnabled(False)
        # self.zoom_in_button.setEnabled(False)
        # self.zoom_out_button.setEnabled(False)
        # self.zoom_all_button.setEnabled(False)
        # self.previous_button.setEnabled(False)
        # self.next_button.setEnabled(False)
        # self.faster_button.setEnabled(False)
        # self.slower_button.setEnabled(False)
        # self.begin_button.setEnabled(False)
        # self.end_button.setEnabled(False)
        # self.save_button.setEnabled(False)

        self._recording = False

        self._restarting = False

        self._timeline.status_bar_changed_signal.connect(self._update_status_bar)

        self._timeline.make_pop_up.connect(self._update_popup)

        self.set_status_text.connect(self._set_status_text)

        self.check_restarting_file_exist()

    def _update_popup(self):
        QMessageBox.about(self, "Record", "record successfuly")

    def check_restarting_file_exist(self):
        import inspect, os
        path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) + "/Scenarios/Extentions/tmp/scenarios_counter.tmp"
        try:
            with open(path, 'r') as f:
                info = f.read().splitlines()
                record_filename = info[0]
                selected_scenario = eval(info[2])
            if selected_scenario["params"]["number of simulations"] == 0:
                os.remove(path)
            else:
                if os.path.isfile(get_path() + "logger_topic.log"):
                    with open(get_path() + "logger_topic.log", 'r') as f:
                        topics = f.read().splitlines()
                        print topics
                    self.start_recording("", "", record_filename, selected_scenario, topics, 2, 0, 0, "", "")
                else:
                    self.start_recording("", "", record_filename, selected_scenario, [], 1, 0, 0, "", "")
        except IOError:
            return

    def graphics_view_on_key_press(self, event):
        key = event.key()
        if key in (Qt.Key_Left, Qt.Key_Right, Qt.Key_Up, Qt.Key_Down, Qt.Key_PageUp, Qt.Key_PageDown):
            # This causes the graphics view to ignore these keys so they can be caught by the bag_widget keyPressEvent
            event.ignore()
        else:
            # Maintains functionality for all other keys QGraphicsView implements
            QGraphicsView.keyPressEvent(self.graphics_view, event)

    # callbacks for ui events
    def on_key_press(self, event):
        key = event.key()
        if key == Qt.Key_Space:
            self._timeline.toggle_play()
        # elif key == Qt.Key_Home:
        #     self._timeline.navigate_start()
        elif key == Qt.Key_End:
            self._handle_end_clicked()
        elif key == Qt.Key_Plus or key == Qt.Key_Equal:
            self._handle_faster_clicked()
        elif key == Qt.Key_Minus:
            self._handle_slower_clicked()
        elif key == Qt.Key_Left:
            self._timeline.translate_timeline_left()
        elif key == Qt.Key_Right:
            self._timeline.translate_timeline_right()
        elif key == Qt.Key_Up or key == Qt.Key_PageUp:
            self._handle_zoom_in_clicked()
        elif key == Qt.Key_Down or key == Qt.Key_PageDown:
            self._handle_zoom_out_clicked()

    def handle_destroy(self):
        self._timeline.handle_close()

    def handle_close(self, event):
        self.shutdown_all()

        event.accept()

    def _resizeEvent(self, event):
        # TODO The -2 allows a buffer zone to make sure the scroll bars do not appear when not needed. On some systems (Lucid) this doesn't function properly
        # need to look at a method to determine the maximum size of the scene that will maintain a proper no scrollbar fit in the view.
        self.graphics_view.scene().setSceneRect(0, 0, self.graphics_view.width() - 2, max(self.graphics_view.height() - 2, self._timeline._timeline_frame._history_bottom))

    def _handle_publish_clicked(self, checked):
        self._timeline.set_publishing_state(checked)

    # def _handle_play_clicked(self, checked):
    #     if checked:
    #         self.play_button.setIcon(self.pause_icon)
    #         self._timeline.navigate_play()
    #     else:
    #         self.play_button.setIcon(self.play_icon)
    #         self._timeline.navigate_stop()

    # def _handle_next_clicked(self):
    #     self._timeline.navigate_next()
    #     self.play_button.setChecked(False)
    #     self.play_button.setIcon(self.play_icon)

    # def _handle_previous_clicked(self):
    #     self._timeline.navigate_previous()
    #     self.play_button.setChecked(False)
    #     self.play_button.setIcon(self.play_icon)

    # def _handle_faster_clicked(self):
    #     self._timeline.navigate_fastforward()
    #     self.play_button.setChecked(True)
    #     self.play_button.setIcon(self.pause_icon)

    # def _handle_slower_clicked(self):
    #     self._timeline.navigate_rewind()
    #     self.play_button.setChecked(True)
    #     self.play_button.setIcon(self.pause_icon)

    # def _handle_begin_clicked(self):
    #     self._timeline.navigate_start()

    # def _handle_end_clicked(self):
    #     self._timeline.navigate_end()

    # def _handle_thumbs_clicked(self, checked):
    #     self._timeline._timeline_frame.toggle_renderers()

    # def _handle_zoom_all_clicked(self):
    #     self._timeline.reset_zoom()

    # def _handle_zoom_out_clicked(self):
    #     self._timeline.zoom_out()

    # def _handle_zoom_in_clicked(self):
    #     self._timeline.zoom_in()

    def _handle_record_clicked(self):
        if self._restarting:
            self.apply_restart()

        if self._recording:
            self._timeline.toggle_recording()
            return

        #TODO Implement limiting by regex and by number of messages per topic
        self.topic_selection = TopicSelection()
        self.topic_selection.recordSettingsSelected.connect(self._on_record_settings_selected)

    def apply_restart(self, reindex_bag):
        import inspect, subprocess
        ## subprocess.Popen("rosbag reindex \"%s\" && rm \"%s\"" % (reindex_bag, reindex_bag[:-4] + ".orig.bag"), shell=True) # for reindexing
        subprocess.Popen("rosbag reindex \"%s\" && rm \"%s\" && mv \"%s\" \"%s\"" % (reindex_bag+".active", reindex_bag + ".orig.active", reindex_bag + ".active", reindex_bag),
                         shell=True)  # for reindexing

        restart_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) + "/Scenarios/Extentions/scripts/restart.sh"
        subprocess.Popen(restart_path, shell=True)

    def _handle_history_clicked(self):
        # if self._recording:
        #     self._timeline.toggle_recording()
        #     return

        #TODO Implement limiting by regex and by number of messages per topic
        self.history_selection = HistorySelection()
        # self.topic_selection.recordSettingsSelected.connect(self._on_record_settings_selected)

    def get_current_opened_directory(self, filepath):
        import os
        direc = "/"
        if os.path.isfile(filepath):
            with open(filepath, 'r') as f:
                pathes = f.read()
                direc = pathes.rsplit('/', 1)[0]
        return direc

    def _on_record_settings_selected(self, general_filename_read, specific_filename_read, selected_topics, selected_scenario, action_id, time_interval, threshold, rule_filename, plp_filename):
        record_filename = ""
        if action_id != 1:
            # TODO verify master is still running
            import inspect
            filepath = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) + "/log/save_record.log"
            current_directory = self.get_current_opened_directory(filepath)
            filename = QFileDialog.getSaveFileName(self, self.tr('Select prefix for new Bag File'), current_directory,
                                                   self.tr('Bag files {.bag} (*.bag)'))
            if filename[0] != '':
                with open(filepath, "w") as f:
                    f.write(filename[0])
                record_filename = filename[0].strip()
                if not record_filename.endswith('.bag'):
                    record_filename = record_filename + ".bag"

                rospy.loginfo('Recording to %s.' % record_filename)
                self.start_recording(general_filename_read, specific_filename_read, record_filename, selected_scenario, selected_topics, action_id, time_interval, threshold, rule_filename, plp_filename)
        else:
            self.start_recording(general_filename_read, specific_filename_read, record_filename, selected_scenario, selected_topics, action_id, time_interval, threshold, rule_filename, plp_filename)

    def start_recording(self,general_filename_read, specific_filename_read, record_filename, selected_scenario, selected_topics, action_id, time_interval, threshold, rule_filename, plp_filename):
        # if action_id != 1:
            self._timeline.setBagWidget(self)

            self._recording = True
            self.run_scen = S.RunScenario(self._timeline, record_filename, selected_scenario, selected_topics, action_id, time_interval, threshold, rule_filename, plp_filename, general_filename_read, specific_filename_read)
            self.run_scen.run_record_scenario()

            self.load_button.setEnabled(False)
            self.history_button.setEnabled(False)
            self.restart_button.setEnabled(True)

            self.record_button.setToolTip("Pause")
            self.record_button.setIcon(QIcon.fromTheme('media-playback-pause'))
        # else:
        #     self.run_scen = S.RunScenario(self._timeline, record_filename, selected_scenario, selected_topics, action_id, time_interval, threshold, rule_filename, plp_filename, general_filename_read, specific_filename_read)
        #     self.run_scen.run_record_scenario()

    def get_current_opened_directory(self, filepath):
        direc = "/"
        if os.path.isfile(filepath):
            with open(filepath, 'r') as f:
                pathes = f.read()
                direc = pathes.rsplit('/', 1)[0]
        return direc

    def _handle_restart_clicked(self):
        self.handle_destroy()
        self.apply_restart(self.run_scen.get_export_bag())

    def _handle_load_clicked(self):
        # path = QFileDialog.getOpenFileName(self, self.tr('Load from File'), '.', self.tr('Bag files {.bag} (*.bag)'))
        # path = str(QFileDialog.getExistingDirectory(self, "Select Directory"))
        import inspect
        filepath = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) + "/log/select_bag.log"
        current_directory = self.get_current_opened_directory(filepath)
        fd = QFileDialog(self)
        wc = "Bag files {.bag} (*.bag)"
        # for ubuntu 14
        path, filter = fd.getOpenFileNamesAndFilter(filter=wc, initialFilter=('*.bag'), directory=current_directory)
        # path, filter = fd.getOpenFileNames(filter=wc, initialFilter=('*.bag'), directory=current_directory)
        # print path
        if len(path) != 0:
            with open(filepath, "w") as f:
                f.write(path[0])
        # if path[0][-4:] == ".bag":   # file
        #     print "True"
        #     self.load_bag(path[0])
        #     files = [path[0]]
        # else: #dir
        #     print "Else " + path
        #     path = path + "/"
        #     files = self.get_bag_files(path)
        #     print files
        #
        # # files = self.get_bag_files("/home/lab/bagfiles/")
        if path:
            self.get_features_from_bags(path)

        #TODO files is list of files

    def get_bag_files(self, path):
        import glob
        files = glob.glob(path + "*.bag")
        return files

    def get_features_from_bags(self, bag_files):
        bag = rosbag.Bag(bag_files[0])
        duration = bag.get_end_time() - bag.get_start_time()
        topics = bag.get_type_and_topic_info().topics
        set_topics = set(topics.keys())
        for f in bag_files:
            bag = rosbag.Bag(f)
            duration = min(duration, bag.get_end_time()-bag.get_start_time())
            topics = bag.get_type_and_topic_info().topics
            topics = set(topics.keys())
            set_topics = set_topics.intersection(topics)
        list_topics = list(set_topics)
        BagParser(bag_files, list_topics, duration)

    def load_bag(self, filename):
        qWarning("Loading %s" % filename)

        # QProgressBar can EITHER: show text or show a bouncing loading bar,
        #  but apparently the text is hidden when the bounding loading bar is
        #  shown
        #self.progress_bar.setRange(0, 0)
        self.set_status_text.emit("Loading %s" % filename)
        #progress_format = self.progress_bar.format()
        #progress_text_visible = self.progress_bar.isTextVisible()
        #self.progress_bar.setFormat("Loading %s" % filename)
        #self.progress_bar.setTextVisible(True)

        bag = rosbag.Bag(filename)
        # self.play_button.setEnabled(True)
        # self.thumbs_button.setEnabled(True)
        # self.zoom_in_button.setEnabled(True)
        # self.zoom_out_button.setEnabled(True)
        # self.zoom_all_button.setEnabled(True)
        # self.next_button.setEnabled(True)
        # self.previous_button.setEnabled(True)
        # self.faster_button.setEnabled(True)
        # self.slower_button.setEnabled(True)
        # self.begin_button.setEnabled(True)
        # self.end_button.setEnabled(True)
        # self.save_button.setEnabled(True)h
        self.record_button.setEnabled(False)
        self._timeline.add_bag(bag)
        qWarning("Done loading %s" % filename )
        # put the progress bar back the way it was
        self.set_status_text.emit("")
        #self.progress_bar.setFormat(progress_format)
        #self.progress_bar.setTextVisible(progress_text_visible) # causes a segfault :(
        #self.progress_bar.setRange(0, 100)
        # self clear loading filename
        #for topic, msg, t in bag.read_messages(topics=['/cmd_vel']):
            #self.logText.appendPlainText("%s %s %s" % (msg, t.secs, topic))

        #self.logText.appendPlainText("matan")
            #print msg, str(t.secs), topic
        #self.logText.appendPlainText(str(bag.get_compression_info()))
        #self.logText.appendPlainText(str(bag.get_type_and_topic_info().topics.keys()))
        # topics = bag.get_type_and_topic_info().topics
        # listtopics = []
        # for key in topics.keys():
        #     listtopics.append(key)
        # #self.logText.appendPlainText(str(bag.get_type_and_topic_info()))
        # # self.logText.appendPlainText(str(bag.get_end_time()-bag.get_start_time()))
        #
        # duration = bag.get_end_time()-bag.get_start_time()
        #
        # bagfile = filename
        # BagParser(bagfile, listtopics, duration)


    # def _handle_save_clicked(self):
    #     filename = QFileDialog.getSaveFileName(self, self.tr('Save selected region to file...'), '.', self.tr('Bag files {.bag} (*.bag)'))
    #     if filename[0] != '':
    #         self._timeline.copy_region_to_bag(filename[0])

    def _set_status_text(self, text):
        if text:
            self.progress_bar.setFormat(text)
            self.progress_bar.setTextVisible(True)
        else:
            self.progress_bar.setTextVisible(False)

    def _update_status_bar(self):
        if self._timeline._timeline_frame.playhead is None or self._timeline._timeline_frame.start_stamp is None:
            return
        # TODO Figure out why this function is causing a "RuntimeError: wrapped C/C++ object of %S has been deleted" on close if the playhead is moving
        try:
            # Background Process Status
            self.progress_bar.setValue(self._timeline.background_progress)

            # Raw timestamp
            self.stamp_label.setText('%.3fs' % self._timeline._timeline_frame.playhead.to_sec())

            # Human-readable time
            self.date_label.setText(bag_helper.stamp_to_str(self._timeline._timeline_frame.playhead))

            # Elapsed time (in seconds)
            self.seconds_label.setText('%.3fs' % (self._timeline._timeline_frame.playhead - self._timeline._timeline_frame.start_stamp).to_sec())

            # File size
            self.filesize_label.setText(bag_helper.filesize_to_str(self._timeline.file_size()))

            # Play speed
            spd = self._timeline.play_speed
            if spd != 0.0:
                if spd > 1.0:
                    spd_str = '>> %.0fx' % spd
                elif spd == 1.0:
                    spd_str = '>'
                elif spd > 0.0:
                    spd_str = '> 1/%.0fx' % (1.0 / spd)
                elif spd > -1.0:
                    spd_str = '< 1/%.0fx' % (1.0 / -spd)
                elif spd == 1.0:
                    spd_str = '<'
                else:
                    spd_str = '<< %.0fx' % -spd
                self.playspeed_label.setText(spd_str)
            else:
                self.playspeed_label.setText('')
        except:
            return


    def remove_tmp_files(self):
        import glob, inspect
        tmp_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) + "/Scenarios/Extentions/tmp/"
        # print tmp_path
        files = glob.glob('%s*'% tmp_path)
        # print files
        for f in files:
            os.remove(f)

    # Shutdown all members
    def shutdown_all(self):
        import inspect, subprocess
        restart_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) + "/Scenarios/Extentions/scripts/clean.sh"
        self.remove_tmp_files()
        print "dead shot"
        subprocess.Popen(restart_path, shell=True)
        self._timeline.handle_close()
        
def get_path():
    import inspect, os
    import logging
    logging_file = os.path.dirname(
        os.path.abspath(inspect.getfile(inspect.currentframe()))) + "/Scenarios/History_Choosing/"
    return logging_file
  