from PyQt5.QtWidgets import QPushButton, QDialog, QVBoxLayout, QLabel, QLineEdit, QRadioButton, QButtonGroup, QMessageBox
from PyQt5.QtCore import pyqtSignal, QThread, Qt
from PyQt5 import QtCore
from PyQt5.QtGui import QPixmap, QPalette, QBrush
from PyQt5.QtMultimedia import QSound
from release_mcgs import AlphaZeroMCTS
from board import Board

import cProfile


class AI(QThread):
    finishSignal = pyqtSignal(int)

    def __init__(self, board: Board, search: AlphaZeroMCTS):
        super(AI, self).__init__()
        self.search = search
        self.board = board

    def run(self) -> None:
        action = self.search.get_action(self.board)
        self.finishSignal.emit(action)

class MatchDialog(QDialog):
    def __init__(self, parent=None):
        super(MatchDialog, self).__init__(parent)
        self.parent = parent

        self.setWindowTitle("比赛信息")
        self.resize(300, 200)
        self.setStyleSheet("background-color: white;")

        # 创建标签和文本框
        self.label_our_name = QLabel("我方名称:")
        self.line_edit_our_name = QLineEdit()
        self.line_edit_our_name.setText("悟空海克斯")

        self.label_opponent_name = QLabel("对方名称:")
        self.line_edit_opponent_name = QLineEdit()
        self.line_edit_opponent_name.setText("吉大")

        self.label_location = QLabel("比赛地点:")
        self.line_edit_location = QLineEdit()
        self.line_edit_location.setText("渤海大学")

        self.label_match_name = QLabel("比赛名称:")
        self.line_edit_match_name = QLineEdit()
        self.line_edit_match_name.setText("2023 CCGC")

        self.label_search_time_name = QLabel("搜索时间:")
        self.line_edit_search_time_name = QLineEdit()
        self.line_edit_search_time_name.setText("1")
        self.line_edit_search_time_name.setText("50")

        # 创建单选按钮和按钮组
        self.radio_btn_first_hand = QRadioButton("我方先手")
        self.radio_btn_second_hand = QRadioButton("对方先手")
        self.button_group = QButtonGroup()
        self.button_group.addButton(self.radio_btn_first_hand)
        self.button_group.addButton(self.radio_btn_second_hand)

        # 创建确认按钮
        self.ok_button = QPushButton("确定")
        self.ok_button.clicked.connect(self.on_ok_button_clicked)

        # 创建布局并添加控件
        layout = QVBoxLayout()
        layout.addWidget(self.label_our_name)
        layout.addWidget(self.line_edit_our_name)
        layout.addWidget(self.label_opponent_name)
        layout.addWidget(self.line_edit_opponent_name)
        layout.addWidget(self.label_location)
        layout.addWidget(self.line_edit_location)
        layout.addWidget(self.label_match_name)
        layout.addWidget(self.line_edit_match_name)
        layout.addWidget(self.label_search_time_name)
        layout.addWidget(self.line_edit_search_time_name)
        layout.addWidget(self.radio_btn_first_hand)
        layout.addWidget(self.radio_btn_second_hand)
        layout.addWidget(self.ok_button)

        self.setLayout(layout)

    def on_ok_button_clicked(self):
        our_name = self.line_edit_our_name.text()
        opponent_name = self.line_edit_opponent_name.text()
        location = self.line_edit_location.text()
        match_name = self.line_edit_match_name.text()
        search_time = self.line_edit_search_time_name.text()

        self.parent.myName = our_name
        self.parent.oppoName = opponent_name
        self.parent.location = location
        self.parent.raceName = match_name
        self.parent.searchTime = float(search_time)

        if self.radio_btn_first_hand.isChecked():
            self.parent.ai_first = True
        else:
            self.parent.ai_first = False

        self.close()

def get_setting_button(self):
    button = QPushButton(self)
    button.setGeometry(1280, 600, 160, 40)  # 设置按钮的位置和大小
    button.setText("游戏设置")
    button.setStyleSheet('''
                   /* 按钮样式 */
                    QPushButton {
                        background-color: #3498db; /* 按钮背景色 - 蓝色 */
                        color: black; /* 文本颜色 - 白色 */
                        border: 2px solid #2980b9; /* 边框颜色 - 深蓝色 */
                        border-radius: 8px; /* 圆角半径 */
                        padding: 10px 20px; /* 按钮内边距 */
                       font-size: 18px;
                    }
                   QPushButton:pressed {
                       background-color: #008B8B;  /* 按下时的背景色 */
                   }
               ''')
    button.setChecked(False)  # 设置开始时的状态为未选中
    button.clicked.connect(lambda: openOptionWindow(self))
    return button

from policy_value_net import PolicyValueNet
import torch
import os

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

model = PolicyValueNet(11, 2, 20, 128, False)
model_file = resource_path('model/20b128c_hex11.pth')
model_file = resource_path('D:/Code/game/HexU2/HexUI1/model/20b128c_hex11.pth')
model_dict = torch.load(model_file, map_location='cpu').state_dict()
model.load_state_dict(model_dict)
model.eval()


def openOptionWindow(self):
    match_dialog = MatchDialog(parent=self)
    match_dialog.exec_()

    self.search = AlphaZeroMCTS(net=model, c_puct=1, n_iters=self.searchTime, game_round=1)
    self.ai = AI(self.board, self.search)
    self.ai.finishSignal.connect(self.doAction, Qt.BlockingQueuedConnection)
