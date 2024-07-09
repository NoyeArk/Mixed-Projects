import sys
import os
import time
from PyQt5 import QtCore, QtGui
from PyQt5.QtCore import QThread, pyqtSignal, QEvent, Qt, QPoint
import torch
from PyQt5.QtGui import QPixmap, QPainter, QFont, QPen, QPolygon
from PyQt5.QtMultimedia import QSound
from PyQt5.QtWidgets import QLabel, QWidget, QApplication, QMessageBox, QPushButton
from board import Board
from search import AlphaZeroMCTS
from policy_value_net import PolicyValueNet
from setting_button import get_setting_button

WIDTH = 1500
HEIGHT = 800
PIECE = 50

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


class Label(QLabel):

    def __init__(self, parent):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setAttribute(Qt.WA_TranslucentBackground)

    def enterEvent(self, e: QEvent) -> None:
        e.ignore()

class HexUI(QWidget):

    def __init__(self):
        super().__init__()
        self.ai_first = None
        self.searchTime = 3
        self.myName = None
        self.oppoName = None
        self.location = None
        self.raceName = None
        self.board = Board(11, 2)
        self.setting_button = get_setting_button(self)
        self.initUI()

    def initAI(self):
        self.search.set_searchTime(self.searchTime)

    def initUI(self):
        self.setCursor(Qt.UpArrowCursor)  # 鼠标变成向上箭头形状
        self.sound_piece = QSound(resource_path("sound/move.wav"))  # 加载落子音效
        self.sound_win = QSound(resource_path("sound/win.wav"))  # 加载胜利音效
        self.sound_defeated = QSound(resource_path("sound/defeated.wav"))  # 加载失败音效
        self.resize(WIDTH, HEIGHT + 50)
        self.setMinimumSize(QtCore.QSize(WIDTH, HEIGHT + 50))
        self.setMaximumSize(QtCore.QSize(WIDTH, HEIGHT + 50))
        self.setStyleSheet('''QWidget{background-color:	#006400;}''')
        self.setWindowTitle('一次拓展一个结点')  # 窗口名称

        self.black = QPixmap(resource_path('img/black.png'))
        self.white = QPixmap(resource_path('img/white.png'))

        self.pieces = [Label(self) for i in range(121)]  # 新建棋子标签，准备在棋盘上绘制棋子

        for piece in self.pieces:
            piece.setVisible(True)  # 图片可视
            piece.setScaledContents(True)  # 图片大小根据标签大小可变

        self.mouse_point = Label(self)  # 将鼠标图片改为棋子
        self.mouse_point.raise_()  # 鼠标始终在最上层

        for i in range(11):
            # 画棋盘横坐标标签
            x_label = QLabel(chr(65 + i), self)
            x_label.setFont(QFont("Roman times", 10, QFont.Bold))
            x_label.move(
                int((1500 // (11 * 2)) + (1500 // (11 * 2)) // 2 + (
                        1500 // (11 * 2)) * i + (1500 // (11 * 2)) * 11 * 0.5),
                int((1500 // (11 * 2)) + ((1500 // 1.73) // (11 * 2)) * 1.5 * 11)
            )

            # 画棋盘纵坐标标签
            y_label = QLabel(str(11-i), self)
            y_label.setFont(QFont("Roman times", 10, QFont.Bold))
            y_label.move(
                        int((1500 // (11 * 2)) - (1500 // (11 * 2)) * 0.5 + (
                        1500 // (11 * 2)) * i * 0.5),
               int( (1500 / (11 * 2)) + ((1500 // 1.73) // (11 * 2)) * 0.2 + (
                        (1500 // 1.73) // (11 * 2)) * 1.5 * i)
            )


        # 此处坐标只是用于追踪AI的下棋位置
        self.x = 10000
        self.y = 10000

        # 悔棋按钮
        self.button_undo = QPushButton("", self)
        self.button_undo.setGeometry(1350, 700, 40, 40)    # 设置按钮的位置和大小
        self.button_undo.setStyleSheet("QPushButton{border-image: url(img/regret.png)}") # 绑定图片
        self.button_undo.setChecked(False)  # 设置开始时的状态为未选中
        self.button_undo.clicked.connect(self.undo)

        self.rate_label = QLabel("累计思考时间:", self)
        self.rate_label.setFont(QFont("Roman times", 14, QFont.Bold))
        self.rate_label.setGeometry(1100, 210, 250, 60)

        self.time_label = QLabel("0s", self)
        self.time_label.setFont(QFont("Roman times", 14, QFont.Bold))
        self.time_label.setGeometry(1250, 210, 100, 60)

        self.count_label = QLabel("搜索次数:", self)
        self.count_label.setFont(QFont("Roman times", 14, QFont.Bold))
        self.count_label.setGeometry(1100, 270, 250, 60)

        self.count_number = QLabel("0次", self)
        self.count_number.setFont(QFont("Roman times", 14, QFont.Bold))
        self.count_number.setGeometry(1250, 270, 100, 60)

        self.my_rate = QLabel("我方胜率:", self)
        self.my_rate.setFont(QFont("Roman times", 14, QFont.Bold))
        self.my_rate.setGeometry(1100, 330, 250, 60)

        self.my_rate_number = QLabel("0%", self)
        self.my_rate_number.setFont(QFont("Roman times", 14, QFont.Bold))
        self.my_rate_number.setGeometry(1250, 330, 100, 60)


        self.show()

    def undo(self):
        """
        棋盘悔两步棋。
        :return:
        """
        # 1
        self.pieces[len(self.board.state) - 1].hide()
        self.pieces[len(self.board.state) - 1] = Label(self)
        self.pieces[len(self.board.state) - 1].setVisible(True)  # 图片可视
        self.pieces[len(self.board.state) - 1].setScaledContents(True)  # 图片大小根据标签大小可变
        # 2
        self.pieces[len(self.board.state) - 2].hide()
        self.pieces[len(self.board.state) - 2] = Label(self)
        self.pieces[len(self.board.state) - 2].setVisible(True)  # 图片可视
        self.pieces[len(self.board.state) - 2].setScaledContents(True)  # 图片大小根据标签大小可变
        # 逻辑悔棋
        self.board.undo()


    def paintEvent(self, a0: QtGui.QPaintEvent) -> None:
        """
        画棋盘，画最终 AI落子的最终位置
        :param a0:
        :return:
        """
        painter = QPainter()
        painter.begin(self)

        pen = QPen(Qt.black, 3, Qt.SolidLine)
        painter.setPen(pen)

        for j in range(11):
            for i in range(11):
                point = QPolygon(
                    [
                        QPoint(
                            int((1500 // (11 * 2)) + (1500 // (11 * 2)) * i + (
                                    1500 // (11 * 2)) * j * 0.5),
                            int((1500 // (11 * 2)) + ((1500 // 1.73) / (11 * 2)) * 1.5 * j)
                        ),
                        QPoint(
                                    int((1500 / (11 * 2)) + (1500 / (11 * 2)) * i + (
                                    int(1500 / (11 * 2)) * j * 0.5)),
                                    int((1500 / (11 * 2)) + ((1500 / 1.73) / (11 * 2)) + (
                                    (1500 / 1.73) / (11 * 2)) * 1.5 * j)
                        ),
                        QPoint(
                            int((1500 / (11 * 2)) + (1500 / (11 * 2)) / 2 + (
                                    1500 / (11 * 2)) * i + (1500 / (11 * 2)) * j * 0.5),
                            int((1500 / (11 * 2)) + ((1500 / 1.73) / (11 * 2)) * 1.5 + (
                                    (1500 / 1.73) / (11 * 2)) * 1.5 * j)
                        ),
                        QPoint(
                            int((1500 / (11 * 2)) + (1500 / (11 * 2)) + (
                                    1500 / (11 * 2)) * i + (1500 / (11 * 2)) * j * 0.5),
                            int((1500 / (11 * 2)) + ((1500 / 1.73) / (11 * 2)) + (
                                    (1500 / 1.73) / (11 * 2)) * 1.5 * j)
                        ),
                        QPoint(
                            int((1500 / (11 * 2)) + (1500 / (11 * 2)) + (
                                    1500 / (11 * 2)) * i + (1500 / (11 * 2)) * j * 0.5),
                            int((1500 / (11 * 2)) + ((1500 / 1.73) / (11 * 2)) * 1.5 * j)
                        ),
                        QPoint(
                            int((1500 / (11 * 2)) + (1500 / (11 * 2)) / 2 + (
                                    1500 / (11 * 2)) * i + (1500 / (11 * 2)) * j * 0.5),
                            int((1500 / (11 * 2)) - ((1500 / 1.73) / (11 * 2)) / 2 + (
                                    (1500 / 1.73) / (11 * 2)) * 1.5 * j)
                        )
                    ]
                )
                painter.drawPolygon(point)

        p = QPainter()
        p.begin(self)

        pencil = QPen(Qt.red, 5, Qt.SolidLine)
        p.setPen(pencil)
        for i in range(11):
            # 画棋盘横线
            p.drawLine(
                int((1500 / (11 * 2)) + (1500 * i / (11 * 2))),
                int((1500 / (11 * 2))),
                int((1500 / (11 * 2)) + (1500 / (11 * 2)) / 2 + (1500 * i / (11 * 2))),
                int((1500 / (11 * 2)) - ((1500 / 1.73) / (11 * 2)) / 2)
            )
            # 画棋盘竖线
            p.drawLine(
                int((1500 / (11 * 2)) + (1500 / (11 * 2)) / 2 + (1500 * i / (11 * 2))),
                int((1500 / (11 * 2)) - ((1500 / 1.73) / (11 * 2)) / 2),
                int((1500 / (11 * 2)) + (1500 / (11 * 2)) + (1500 * i / (11 * 2))),
                int((1500 / (11 * 2)))
            )

            p.drawLine(
                int((1500 / (11 * 2)) + (
                        1500 / (11 * 2)) * 10 * 0.5 + (1500 * i / (11 * 2))),
                int((1500 / (11 * 2)) + ((1500 / 1.73) / (11 * 2)) + (
                        (1500 / 1.73) / (11 * 2)) * 1.5 * 10),
                int((1500 / (11 * 2)) + (1500 / (11 * 2)) / 2 + (
                        1500 / (11 * 2)) * 0 + (1500 / (11 * 2)) * 10 * 0.5 + (1500 * i / (11 * 2))),
                int((1500 / (11 * 2)) + ((1500 / 1.73) / (11 * 2)) * 1.5 + (
                        (1500 / 1.73) / (11 * 2)) * 1.5 * 10)
            )
            p.drawLine(
                int((1500 / (11 * 2)) + (1500 / (11 * 2)) / 2 + (
                        1500 / (11 * 2)) * 0 + (1500 / (11 * 2)) * 10 * 0.5 + (1500 * i / (11 * 2))),
                int((1500 / (11 * 2)) + ((1500 / 1.73) / (11 * 2)) * 1.5 + (
                        (1500 / 1.73) / (11 * 2)) * 1.5 * 10),
                int((1500 / (11 * 2)) + (1500 / (11 * 2)) + (
                        1500 / (11 * 2)) * 0 + (1500 / (11 * 2)) * 10 * 0.5 + (1500 * i / (11 * 2))),
                int(((1500 / (11 * 2)) + ((1500 / 1.73) / (11 * 2)) + (
                        (1500 / 1.73) / (11 * 2)) * 1.5 * 10))
            )

        p0 = QPainter()
        p0.begin(self)

        pencil0 = QPen(Qt.blue, 5, Qt.SolidLine)
        p0.setPen(pencil0)

        for j in range(11):
            p0.drawLine(
                int((1500 / (11 * 2)) + (1500 / (11 * 2)) * 0 + (
                        1500 / (11 * 2)) * j * 0.5),
                int((1500 / (11 * 2)) + ((1500 / 1.73) / (11 * 2)) * 1.5 * j),
                int((1500 / (11 * 2)) + (1500 / (11 * 2)) * 0 + (
                        1500 / (11 * 2)) * j * 0.5),
                int((1500 / (11 * 2)) + ((1500 / 1.73) / (11 * 2)) + (
                        (1500 / 1.73) / (11 * 2)) * 1.5 * j)
            )

            p0.drawLine(
                int((1500 / (11 * 2)) + (1500 / (11 * 2)) * 0 + (
                        1500 / (11 * 2)) * j * 0.5),
                int((1500 / (11 * 2)) + ((1500 / 1.73) / (11 * 2)) + (
                        (1500 / 1.73) / (11 * 2)) * 1.5 * j),
                int((1500 / (11 * 2)) + (1500 / (11 * 2)) / 2 + (
                        1500 / (11 * 2)) * 0 + (1500 / (11 * 2)) * j * 0.5),
                int((1500 / (11 * 2)) + ((1500 / 1.73) / (11 * 2)) * 1.5 + (
                        (1500 / 1.73) / (11 * 2)) * 1.5 * j)
            )

            p0.drawLine(
                int((1500 / (11 * 2)) + (1500 / (11 * 2)) / 2 + (
                        1500 / (11 * 2)) * 10 + (1500 / (11 * 2)) * j * 0.5),
                int((1500 / (11 * 2)) - ((1500 / 1.73) / (11 * 2)) / 2 + (
                        (1500 / 1.73) / (11 * 2)) * 1.5 * j),
                int((1500 / (11 * 2)) + (1500 / (11 * 2)) + (
                        1500 / (11 * 2)) * 10 + (1500 / (11 * 2)) * j * 0.5),
                int((1500 / (11 * 2)) + ((1500 / 1.73) / (11 * 2)) * 1.5 * j)
            )

            p0.drawLine(
                int((1500 / (11 * 2)) + (1500 / (11 * 2)) + (
                        1500 / (11 * 2)) * 10 + (1500 / (11 * 2)) * j * 0.5),
                int((1500 / (11 * 2)) + ((1500 / 1.73) / (11 * 2)) * 1.5 * j),
                int((1500 / (11 * 2)) + (1500 / (11 * 2)) + (
                        1500 / (11 * 2)) * 10 + (1500 / (11 * 2)) * j * 0.5),
                int((1500 / (11 * 2)) + ((1500 / 1.73) / (11 * 2)) + (
                        (1500 / 1.73) / (11 * 2)) * 1.5 * j)
            )

        # 画出AI行棋的提示
        qp = QPainter()
        qp.begin(self)
        self.drawLines(qp)
        qp.end()

    def drawLines(self, qp):  # 指示AI当前下的棋子
        """
        追踪AI最后下棋位置，进行标记
        :param qp:
        :return:
        """
        pen = QtGui.QPen(Qt.green, 15, QtCore.Qt.SolidLine)
        qp.setPen(pen)
        qp.drawPoint(QPoint(int(self.x + 1), int(self.y + 1)))


    def isAIFirst(self):
        """
        根据AI先后手判断何时开启AI线程
        :return:
        """
        if self.ai_first == True:
            return len(self.board.state) % 2 == 0
        else:
            return len(self.board.state) % 2 == 1

    def mousePressEvent(self, e: QtGui.QMouseEvent) -> None:
        if e.button() == Qt.LeftButton and self.ai_first == None:
            # 创建一个警告提示框
            warning_box = QMessageBox()
            warning_box.setWindowTitle("警告")
            warning_box.setText("请先初始化游戏设置")
            warning_box.setIcon(QMessageBox.Warning)
            warning_box.setStandardButtons(QMessageBox.Ok)

            # 显示警告提示框
            warning_box.exec_()
            return

        if e.button() == Qt.LeftButton and not self.isAIFirst():
            x, y = e.x(), e.y()
            i, j = self.coordinate_transform_pixel2map(x, y)

            if not i is None and not j is None:
                if self.board.board[i][j] == self.board.EMPTY:
                    act = i * 11 + j
                    self.doAction(act)
                    self.time_label.setText(f'{self.searchTime*(len(self.board.state) // 2)}s')
        if self.isAIFirst():
            self.ai.start()

    def doAction(self, action):
        """
        棋盘执行动作
        :param action:
        :return:
        """

        # 获取对应棋子的颜色
        color = self.black
        if self.board.current_player == self.board.WHITE:
            color = self.white

        # 绘制棋盘
        self.draw(action // 11, action % 11, color)

        # 更新棋盘
        self.board.do_action(action)

        # 更新搜索次数
        self.count_number.setText(f'{self.search.node_count}次')

        # 更新搜索胜率
        self.my_rate_number.setText(f'{self.search.search_rate: < 10.2f}%')

        # 判断游戏是否结束
        is_over, winner = self.board.is_game_over()
        if is_over:
            self.game_over(winner)


    def draw(self, i: int, j: int, color: QPixmap):
        """
        画棋子
        :param i: 横坐标
        :param j: 纵坐标
        :return:
        """
        # 棋子转化为画布坐标
        x, y = self.coordinate_transform_map2pixel(i, j)
        self.x, self.y = x, y
        # 放置棋子
        self.pieces[len(self.board.state)].setPixmap(color)
        self.pieces[len(self.board.state)].setGeometry(x, y, PIECE, PIECE)  # 画出棋子
        self.sound_piece.play()  # 落子音效
        self.update()


    def game_over(self, winner):
        """
        游戏结束
        :param winner: 胜利者
        :return:
        """
        black_flag = "后手"
        if winner == self.board.BLACK:
            # self.sound_win.play()
            self.sound_defeated.play()
            reply = QMessageBox.question(self, ' 红胜！', '保存棋谱?',
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            black_flag = "先手"
        else:
            self.sound_defeated.play()
            reply = QMessageBox.question(self, '蓝胜！', '保存棋谱?',
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:  # 复位
            # 以下是打印棋谱，无用
            if not os.path.exists('./data'):
                os.makedirs('./data')
            current_time = time.strftime('%Y.%m.%d %H:%M:%S', time.localtime())
            hex_data = ''
            name = ''
            oppo_name = self.oppoName
            if self.ai_first:
                hex_data += '{' + f'[HEX][{self.myName} R][' + oppo_name + ' B][' + black_flag + '胜][' + str(
                    current_time) + f' {self.location}][{self.raceName}]'
                name += f'./data/HEX-{self.myName} R vs ' + oppo_name + ' B-' + black_flag + f'胜-{self.location}-{self.raceName}.txt'
            else:
                hex_data += '([HEX][' + oppo_name + f' R][{self.myName} B][' + black_flag + '胜][' + str(
                    current_time) + f' {self.location}][{self.raceName}]'
                name += './data/HEX-' + oppo_name + f' R vs {self.myName} B-' + black_flag + f'胜-{self.location}-{self.raceName}.txt'
            hex_data = str(hex_data)
            for k, v in self.board.state.items():
                if v == self.board.BLACK:
                    hex_data += ';R(' + str(chr(65 + (k % self.board.board_len))) + ',' + str(
                        self.board.board_len - k // self.board.board_len) + ')'
                else:
                    hex_data += ';B(' + str(chr(65 + (k % self.board.board_len))) + ',' + str(
                        self.board.board_len - k // self.board.board_len) + ')'
            hex_data += '}'
            output = open(name, 'w', encoding='utf-8')
            output.write(hex_data)
            output.write('\n')
            output.close()
        else:
            self.close()


    def coordinate_transform_map2pixel(self, i, j):
        # 从 chessMap 里的逻辑坐标到 UI 上的绘制坐标的转换
        return 100 + 68 * j + 34 * i - 30 + 8, 90 + 59 * i - 27


    def coordinate_transform_pixel2map(self, x, y):
        for i in range(11):
            for j in range(11):
                if x <= 100 + 68 * j + 34 * i + 32 \
                        and x > 100 + 68 * j + 34 * i - 32 \
                        and y <= 90 + 59 * i + 32 \
                        and y > 90 + 59 * i - 32:
                    return i, j
        return None, None



if __name__ == '__main__':

    if time.time() - 1656475790.6089072 < 365 * 24 * 3600 * 1.5:
        app = QApplication(sys.argv)
        ex = HexUI()
        sys.exit(app.exec_())
    else:
        exit()

