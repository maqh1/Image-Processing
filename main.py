# This Python file uses the following encoding: utf-8

from MainWindow import MainWindow
from PyQt6 import QtCore, QtGui, QtWidgets
import sys

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow.get_instance()
    window.show()
    sys.exit(app.exec())