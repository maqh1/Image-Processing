from PyQt6.QtWidgets import QApplication, QDialog, QFormLayout, QLineEdit, QVBoxLayout, QPushButton, QLabel, \
    QMessageBox, QMainWindow, QWidget


class ParametersInputDialog(QDialog):
    def __init__(self, enum):
        super().__init__()
        if enum == 1:
            self.setWindowTitle("参数输入")
            self.layout = QVBoxLayout(self)

            self.form_layout = QFormLayout()

            self.frequency_edit = QLineEdit(self)
            self.form_layout.addRow("采样频率 (宽，1-65535 Hz):", self.frequency_edit)

            self.quantization_edit = QLineEdit(self)
            self.form_layout.addRow("量化等级 (1-8 位):", self.quantization_edit)

            self.layout.addLayout(self.form_layout)

            self.button = QPushButton("获取参数", self)
            self.button.clicked.connect(self.get_frequency_and_quantization)
            self.layout.addWidget(self.button)


    def get_frequency_and_quantization(self):
        frequency = self.frequency_edit.text()
        quantization = self.quantization_edit.text()
        # 检查输入是否为整数
        try:
            frequency = int(frequency)
            quantization = int(quantization)
        except ValueError:
            QMessageBox.critical(self, "输入错误", "请输入有效的整数。")
            return None
        # 检查范围
        if not (1 <= frequency <= 65535) or not (1 <= quantization <= 8):
            QMessageBox.critical(self, "输入错误", "请确保输入在有效的范围内。")
            return None
        self.accept()
        return frequency, quantization

    @staticmethod
    def get_frequency_and_quantization_static():
        dialog = ParametersInputDialog(1)
        result = dialog.exec()
        if result == QDialog.DialogCode.Accepted:
            return dialog.get_frequency_and_quantization()
        else:
            return None

class MainWindow(QMainWindow):
    # 测试类
    def __init__(self):
        super().__init__()

        self.setWindowTitle("主窗口")
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        self.central_layout = QVBoxLayout(self.central_widget)

        self.show_parameters_button = QPushButton("显示参数对话框", self)
        self.show_parameters_button.clicked.connect(self.show_parameters_dialog)
        self.central_layout.addWidget(self.show_parameters_button)

    def show_parameters_dialog(self):
        ans = ParametersInputDialog.get_frequency_and_quantization_static()
        if ans is not None:
            QMessageBox.information(self, "参数", "采样频率：{} Hz\n量化等级：{} 位".format(ans[0], ans[1]))


if __name__ == "__main__":
    app = QApplication([])

    window = MainWindow()
    window.show()

    app.exec()
