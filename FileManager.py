import cv2
import numpy as np
from PyQt6 import QtWidgets, QtGui
from PyQt6.QtCore import QObject, QDateTime, Qt
from PyQt6.QtGui import QImage, QColor, qRgb, qGray, QPixmap, QFont, QPainter
from PyQt6.QtWidgets import QFileDialog, QMessageBox, QWidget, QVBoxLayout, QLabel, QScrollArea, QLineEdit, QPushButton, \
    QHBoxLayout
from sympy import symbols, sympify, lambdify

from Dialog import ParametersInputDialog


class ImageViewer(QtWidgets.QWidget):
    _instance = None

    def __init__(self, parent=None):
        super(ImageViewer, self).__init__(parent)

        self.tab_widget = QtWidgets.QTabWidget(self)
        # 设置布局
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.tab_widget)
        self.setLayout(layout)
        # 创建一个默认的标签页

    def add_tab(self, image_info):
        new_tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(new_tab)

        graphics_view = QtWidgets.QGraphicsView(new_tab)
        layout.addWidget(graphics_view)

        scene = QtWidgets.QGraphicsScene()
        scene.addPixmap(QtGui.QPixmap.fromImage(image_info[0]))

        graphics_view.setScene(scene)
        self.tab_widget.addTab(new_tab, image_info[1])

    def close_tab(self):
        num = self.tab_widget.currentIndex()
        if num >= 0 and ImagesList.get_instance().remove_image(num):
            self.tab_widget.removeTab(num)

    def update_images(self, num, flag=True):
        image_info = ImagesList.get_instance().imagesList[num]
        image_info[2] = flag
        if not image_info[1] is None:
            name = image_info[1].split('/')[-1] + ('*' if image_info[2] else '')
            self.tab_widget.setTabText(num, name)
        scene = self.tab_widget.widget(num).layout().itemAt(0).widget().scene()
        scene.clear()
        scene.addPixmap(QtGui.QPixmap.fromImage(image_info[0]))

    @staticmethod
    def get_instance(parent=None):
        if ImageViewer._instance is None:
            ImageViewer._instance = ImageViewer(parent)
        return ImageViewer._instance

    def open_image(self):
        image_info = ImagesList.get_instance().open_image()
        if image_info[0] is not None:
            self.add_tab(image_info)

    def new_image(self):
        imagesList = ImagesList.get_instance().imagesList
        ImagesList.get_instance().new_image()
        num = 0  # 存储未命名图片的数量
        for i in range(len(imagesList)):
            if ImagesList.get_instance().imagesList[i][1] is None:
                num += 1
        self.add_tab((imagesList[-1][0], f"未命名{num}*"))

    def check_current_tab(self):
        num = self.tab_widget.currentIndex()
        if num < 0:
            QMessageBox.information(self, "提示", "当前未打开图片")
            return -1
        image_info = ImagesList.get_instance().imagesList[num]
        if image_info[0] is None:
            QMessageBox.information(self, "提示", "图片为空")
            return -1
        return num

    def save_image(self):
        num = self.check_current_tab()
        if num < 0:
            return
        before = ImagesList.get_instance().imagesList[num][1]
        flag = not ImagesList.get_instance().save_image(num)
        after = ImagesList.get_instance().imagesList[num][1]
        if before != after:
            self.tab_widget.setTabText(num, after.split('/')[-1])
        self.update_images(num, flag)

    def save_image_as(self):
        num = self.check_current_tab()
        if num < 0:
            return
        before = ImagesList.get_instance().imagesList[num][1]
        flag = not ImagesList.get_instance().save_image_as(num)
        after = ImagesList.get_instance().imagesList[num][1]
        if before != after:
            self.tab_widget.setTabText(num, after.split('/')[-1])
        self.update_images(num, flag)

    def sampling_and_quantization(self):
        num = self.check_current_tab()
        if num < 0:
            return
        image_info = ImagesList.get_instance().imagesList[num]
        image_info[0] = ImageOperator.sampling_and_quantization(image_info[0])
        self.update_images(num)

    def bit_plane_decomposition(self):
        num = self.check_current_tab()
        if num < 0:
            return
        image_info = ImagesList.get_instance().imagesList[num]
        try:
            bit_planes = ImageOperator.BitPlanesWidget.convert_to_bit_planes(image_info[0])
            self.bit_planes_widget = ImageOperator.BitPlanesWidget(bit_planes)
            self.bit_planes_widget.show()
        except Exception as e:
            print(e)

    def bmp2txt(self):
        num = self.check_current_tab()
        if num < 0:
            return
        image_info = ImagesList.get_instance().imagesList[num]
        try:
            text = ImageOperator.Bmp2TxtWidget.qimage_to_txt(image_info[0])
            self.bmp2txt_widget = ImageOperator.Bmp2TxtWidget(text)
            self.bmp2txt_widget.show()
        except Exception as e:
            print(e)

    def gray_histogram(self):
        num = self.check_current_tab()
        if num < 0:
            return
        try:
            image_info = ImagesList.get_instance().imagesList[num]
            self.gray_histogram_widget = ImageOperator.GrayHistogramWidget(image_info[0])
            self.gray_histogram_widget.show()
        except Exception as e:
            print(e)

    def image_processor(self):
        num = self.check_current_tab()
        if num < 0:
            return
        try:
            image_info = ImagesList.get_instance().imagesList[num]
            self.image_processor_widget = ImageOperator.ImageProcessorWidget((image_info[0], num))  # 传入编号，用于变换
            self.image_processor_widget.show()
        except Exception as e:
            print(e)

    def histogram_equalization(self):
        num = self.check_current_tab()
        if num < 0:
            return
        try:
            image_info = ImagesList.get_instance().imagesList[num]
            self.image_histogram_equalization = ImageOperator.ImageHistogramEqualization((image_info[0], num))
            self.image_histogram_equalization.show()
        except Exception as e:
            print(e)

    def object_detection(self):
        try:
            self.object_detection = ImageOperator.ObjectDetection()
            self.object_detection.show()
        except Exception as e:
            print(e)

    def speed_calculation(self):
        try:
            self.speed_calculation = ImageOperator.SpeedCalculator()
            self.speed_calculation.show()
        except Exception as e:
            print(e)


class ImageOperator:
    @staticmethod
    def sampling_and_quantization(image):
        try:
            parameters = ParametersInputDialog.get_frequency_and_quantization_static()
            if parameters is None:
                return image
            frequency, quantization = parameters
            image_array = ImageOperator.qimage_to_array(image)
            sampled_and_quantized_array = ImageOperator.sample_and_quantize(image_array, frequency, quantization)
            sampled_and_quantized_image = ImageOperator.array_to_qimage(sampled_and_quantized_array)
            return sampled_and_quantized_image
        except Exception as e:
            print(e)
            return image

    @staticmethod
    def qimage_to_array(qimage: QImage) -> np.ndarray:
        # 将QImage转换为NumPy数组
        width = qimage.width()
        height = qimage.height()
        array = np.zeros((height, width), dtype=np.uint8)
        for y in range(height):
            for x in range(width):
                pixel = qimage.pixel(x, y)
                array[y, x] = qGray(pixel)
        return array

    @staticmethod
    def sample_and_quantize(image_array: np.ndarray, sample_rate: float, quantization_bits: int) -> np.ndarray:
        height, width = image_array.shape
        height_rate = int(height / width * sample_rate)
        width_rate = sample_rate

        y_indices = np.arange(0, height, (height / height_rate))[:height_rate]
        x_indices = np.arange(0, width, (width / width_rate))[:width_rate]

        y_indices = y_indices.astype(np.int16)
        x_indices = x_indices.astype(np.int16)

        sampled_array = image_array[y_indices[:, None], x_indices]

        # 进行均匀量化
        sampled_array = (sampled_array // (2 ** (8 - quantization_bits))) * (2 ** (8 - quantization_bits)) + 2 ** (
                8 - quantization_bits) / 2

        return sampled_array.astype(np.uint8)

    @staticmethod
    def set_image(image_label, image, width=512, height=512):
        image = image.scaled(width, height, Qt.AspectRatioMode.KeepAspectRatio)
        image_label.setPixmap(QPixmap.fromImage(image))

    @staticmethod
    def array_to_qimage(image_array: np.ndarray) -> QImage:
        # 将NumPy数组转换为QImage
        height, width = image_array.shape
        qimage = QImage(width, height, QImage.Format.Format_Grayscale8)

        # 将NumPy数组的值赋给QImage
        for y in range(height):
            for x in range(width):
                qimage.setPixel(x, y, qRgb(image_array[y, x], image_array[y, x], image_array[y, x]))
        return qimage

    @staticmethod
    def normalize(image_array):
        image_array = image_array.astype(np.float64)
        mean = np.mean(image_array)
        std = np.std(image_array)
        image_array = (image_array - mean) / std
        return image_array

    class BitPlanesWidget(QWidget):
        size = 200

        def __init__(self, bit_planes):
            super().__init__()
            self.setWindowTitle("位平面分解")
            layout = QVBoxLayout(self)
            labels = []
            for i, bit_plane in enumerate(bit_planes):
                label = QLabel(f"Bit Plane {i}")
                pixmap = QPixmap.fromImage(bit_plane)
                # Resize the pixmap to fit within a square of a specific size
                square_size = self.size
                if pixmap.width() < pixmap.height():
                    scaled_pixmap = pixmap.scaledToHeight(square_size, Qt.TransformationMode.SmoothTransformation)
                else:
                    scaled_pixmap = pixmap.scaledToWidth(square_size, Qt.TransformationMode.SmoothTransformation)
                label.setPixmap(scaled_pixmap)
                labels.append(label)
            if pixmap.width() < pixmap.height():
                count = 0
                QHBoxLayout = QtWidgets.QHBoxLayout(self)
                for label in labels:
                    count += 1
                    # 两个一行
                    if count % 2 == 0:
                        layout.addLayout(QHBoxLayout)
                        QHBoxLayout.addWidget(label)
                        QHBoxLayout = QtWidgets.QHBoxLayout(self)
                    else:
                        QHBoxLayout.addWidget(label)
            else:
                count = 0
                QHBoxLayout = QtWidgets.QHBoxLayout(self)
                for label in labels:
                    count += 1
                    # 四个一行
                    if count % 4 == 0:
                        layout.addLayout(QHBoxLayout)
                        QHBoxLayout.addWidget(label)
                        QHBoxLayout = QtWidgets.QHBoxLayout(self)
                    else:
                        QHBoxLayout.addWidget(label)
            self.setLayout(layout)

        @staticmethod
        def convert_to_bit_planes(image):
            bit_planes = []
            image_array = ImageOperator.qimage_to_array(image)
            height, width = image_array.shape

            for i in range(8):
                # Create a bitmask with the i-th bit set
                bitmask = 1 << i
                # Apply the bitmask and normalize to 0-255
                bit_plane = ((image_array & bitmask) >> i) * 255
                bit_plane_image = QImage(bit_plane.data, width, height, QImage.Format.Format_Grayscale8)
                bit_planes.append(bit_plane_image)

            return bit_planes

        def set_size(self, size):
            self.size = size

    class Bmp2TxtWidget(QWidget):
        def __init__(self, text):
            super().__init__()
            self.setWindowTitle("BMP转TXT")
            # 创建一个滚动区域
            scroll_area = QScrollArea(self)
            scroll_area.setWidgetResizable(True)

            # 在滚动区域内创建一个 QLabel 用于显示文本
            label = QLabel(text)
            # label.setWordWrap(True)  # 自动换行

            # 将 QLabel 放入滚动区域
            scroll_area.setWidget(label)

            # 创建一个垂直布局来容纳滚动区域
            layout = QVBoxLayout(self)
            layout.addWidget(scroll_area)

            self.setLayout(layout)
            font = QFont("Courier New", 6)

            label.setFont(font)

        size = 100
        aspect_ratio = 0.48  # 宽高比，可能不准确

        @staticmethod
        def set_size(size):
            ImageOperator.Bmp2TxtWidget.size = size

        @staticmethod
        def qimage_to_txt(qimage):
            # 获取图像的宽度和高度
            width, height = qimage.width(), qimage.height()
            if width > height:
                small_width = ImageOperator.Bmp2TxtWidget.size
                small_height = int(height / width * small_width * ImageOperator.Bmp2TxtWidget.aspect_ratio)
            else:
                small_height = ImageOperator.Bmp2TxtWidget.size
                small_width = int(width / height * small_height / ImageOperator.Bmp2TxtWidget.aspect_ratio)
            small_image = qimage.scaled(small_width, small_height)
            # 将像素值映射到字符集中的字符
            char_map = [" ", ".", "`", "'", "-", ";", ",", ";", "\"", "_", "~", "!", "^", "i", "r", "|", "/", "I", "=",
                        "<", ">", "*", "l", "\\", "1", "t", "+", "j", "?", "v", ")", "(", "L", "f", "{", "7", "}", "J",
                        "T", "c", "x", "z", "]", "[", "u", "n", "s", "Y", "o", "F", "y", "e", "2", "a", "V", "k", "3",
                        "h", "Z", "C", "4", "P", "5", "A", "q", "X", "K", "6", "H", "Q", "m", "B", "&", "N", "W", "M",
                        "@"]

            # 将像素值映射到字符并创建文本字符串
            text = ""
            for y in range(small_height):
                for x in range(small_width):
                    # 获取像素值
                    pixel_value = small_image.pixelColor(x, y).lightness()
                    # 将像素值映射到字符集中的字符
                    char_index = int(pixel_value / 256 * len(char_map))
                    # 添加字符到文本字符串
                    text += char_map[char_index]
                # 添加换行符
                text += "\n"
            return text

    class GrayHistogramWidget(QWidget):
        def __init__(self, image, parent=None):
            super().__init__(parent)
            self.setWindowTitle("灰度直方图")

            self.image = image
            self.temporary_image = self.image

            self.threshold = -1  # 初始阈值
            self.image_array = ImageOperator.qimage_to_array(self.image)
            self.histogram = self.calculate_histogram(self.image_array)

            self.layout = QVBoxLayout()

            # 显示图像
            self.image_label = QLabel()
            self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.set_image()
            self.layout.addWidget(self.image_label)

            # 显示直方图
            self.histogram_label = QLabel()
            self.histogram_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.histogram_label.setMinimumHeight(100)
            self.layout.addWidget(self.histogram_label)

            # 显示信息
            self.info_label = QLabel()
            self.info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.layout.addWidget(self.info_label)

            # 阈值输入框
            self.threshold_input = QLineEdit(self)
            self.threshold_input.setPlaceholderText("其输入阈值(1-255)")
            self.threshold_input.returnPressed.connect(self.update_threshold)
            self.layout.addWidget(self.threshold_input)

            self.setLayout(self.layout)

            # 更新直方图和信息
            self.update_histogram_label(self.histogram_label)
            self.update_info_label(self.info_label)

        @staticmethod
        def calculate_histogram(image_array):
            hist, _ = np.histogram(image_array, bins=256, range=(0, 255))
            return hist

        def update_histogram_label(self, label):
            histogram_pixmap = QPixmap(256, 100)
            histogram_pixmap.fill(Qt.GlobalColor.white)
            painter = QPainter(histogram_pixmap)
            painter.setPen(QColor(Qt.GlobalColor.black))

            max_hist_value = max(self.histogram)
            scale_factor = 100.0 / max_hist_value if max_hist_value > 0 else 1.0
            for i in range(256):
                bar_height = int(self.histogram[i] * scale_factor)
                painter.drawLine(i, 100, i, 100 - bar_height)
            if 0 <= self.threshold <= 255:
                painter.setPen(QColor(Qt.GlobalColor.red))
                painter.drawLine(self.threshold, 0, self.threshold, 100)
            painter.end()

            histogram_pixmap = histogram_pixmap.scaled(512, 100, Qt.AspectRatioMode.IgnoreAspectRatio)
            label.setFixedSize(512, 100)
            label.setPixmap(histogram_pixmap)

        @staticmethod
        def draw_histogram(histogram, height, width):
            histogram_pixmap = QPixmap(256, height)
            histogram_pixmap.fill(Qt.GlobalColor.white)
            painter = QPainter(histogram_pixmap)
            painter.setPen(QColor(Qt.GlobalColor.black))
            max_hist_value = max(histogram)
            scale_factor = height / max_hist_value if max_hist_value > 0 else 1.0
            for i in range(256):
                bar_height = int(histogram[i] * scale_factor)
                painter.drawLine(i, height, i, height - bar_height)
            painter.end()
            histogram_pixmap = histogram_pixmap.scaled(width, height, Qt.AspectRatioMode.IgnoreAspectRatio)
            return histogram_pixmap

        def update_info_label(self, label):
            # 更新信息的显示
            mean_value = np.mean(self.image_array)
            median_value = np.median(self.image_array)
            std_deviation = np.std(self.image_array)
            total_pixels = np.sum(self.histogram)

            info_text = f"平均灰度: {mean_value:.2f}\n中值灰度: {median_value:.2f}\n标准差: {std_deviation:.2f}\n像素总数: {total_pixels}"
            label.setText(info_text)

        def update_threshold(self):
            # 更新阈值并重新绘制图像
            threshold_str = self.sender().text()
            try:
                self.threshold = int(threshold_str)
                if self.threshold < 0 or self.threshold > 255:
                    raise ValueError
            except ValueError:
                self.threshold = -1
                self.threshold_input.setText("")
                QMessageBox.information(self, "输入错误", "请输入有效的整数。(1-255)")
                return
            self.set_image()
            self.update_histogram_label(self.histogram_label)
            self.update()

        def set_image(self):
            if self.threshold < 0 or self.threshold > 255:
                self.temporary_image = self.image
            else:
                image_array = ImageOperator.qimage_to_array(self.image)
                image_array[image_array < self.threshold] = 0
                image_array[image_array >= self.threshold] = 255
                image = ImageOperator.array_to_qimage(image_array)
                self.temporary_image = image

            self.temporary_image = self.temporary_image.scaled(512, 512, Qt.AspectRatioMode.KeepAspectRatio)
            self.image_label.setPixmap(QPixmap.fromImage(self.temporary_image))

    class ImageProcessorWidget(QWidget):
        def __init__(self, image_info):  # 传入元组为(image,num)
            super().__init__()
            self.image_info = image_info
            self.image = image_info[0]

            self.image_label = QLabel(self)
            self.histogram_before_label = QLabel(self)
            self.histogram_after_label = QLabel(self)

            self.textbox = QLineEdit(self)

            self.checkbox = QtWidgets.QCheckBox("是否单位化，如不选择会将超出范围自动截断", self)
            self.checkbox.setChecked(True)

            self.button = QPushButton('变换', self)
            self.button_1 = QPushButton('应用变换', self)
            self.init_ui()

            self.display_histogram(self.histogram_before_label, self.image)
            self.set_image(self.image)

            self.result_image = None

        def init_ui(self):
            layout = QVBoxLayout(self)

            layout.addWidget(self.image_label)

            layout_H = QHBoxLayout(self)
            layout_H.addWidget(self.histogram_before_label)
            layout_H.addWidget(self.histogram_after_label)
            layout.addLayout(layout_H)

            layout.addWidget(self.textbox)
            layout.addWidget(self.checkbox)
            layout.addWidget(self.button)

            self.button.clicked.connect(self.transform)

            layout.addWidget(self.button_1)
            self.button_1.clicked.connect(self.apply_transform)

            self.setLayout(layout)

            self.setWindowTitle('变换')
            self.textbox.setPlaceholderText("输入表达式，自变量为x，如：x**2+x+1,sin(2*x)，具体语法可查询sympy文档")

        @staticmethod
        def display_histogram(label, image, height=100, width=256):
            hist = ImageOperator.GrayHistogramWidget.calculate_histogram(ImageOperator.qimage_to_array(image))
            pixmap = ImageOperator.GrayHistogramWidget.draw_histogram(hist, height, width)
            label.setPixmap(pixmap)

        # 归一化
        @staticmethod
        def normalize(image_array):
            image_array = image_array.astype(np.float64)
            image_array /= 255
            image_array -= image_array.min()
            image_array /= image_array.max()
            image_array *= 255
            return image_array.astype(np.uint8)

        def set_image(self, image):
            image = image.scaled(512, 512, Qt.AspectRatioMode.KeepAspectRatio)
            self.image_label.setPixmap(QPixmap.fromImage(image))

        def transform(self):
            expression = self.textbox.text()
            image_array = ImageOperator.qimage_to_array(self.image)
            try:
                expr = sympify(expression)
                print(expr)
                x = symbols('x')
                image_array = image_array.astype(np.float64)
                numpy_func = lambdify(x, expr, 'numpy')
                result_array = numpy_func(image_array)
                if self.checkbox.checkState() == Qt.CheckState.Checked:
                    result_array = ImageOperator.ImageProcessorWidget.normalize(result_array)
                else:
                    result_array = result_array.astype(np.int16)
                    result_array[result_array < 0] = 0
                    result_array[result_array > 255] = 255
                    result_array = result_array.astype(np.uint8)
                self.result_image = ImageOperator.array_to_qimage(result_array)
                self.set_image(self.result_image)
                self.display_histogram(self.histogram_after_label, self.result_image)
            except Exception as e:
                print(e)
                QMessageBox.information(self, "输入错误", "请输入有效的表达式。")
                self.textbox.setText("")
                return

        def apply_transform(self):
            if self.result_image is not None:
                reply = QMessageBox.question(self, '应用变换', '是否应用该变换？',
                                             QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
                if reply == QMessageBox.StandardButton.Yes:
                    ImagesList.get_instance().imagesList[self.image_info[1]][0] = self.result_image
                    ImageViewer.get_instance().update_images(self.image_info[1])
                    QMessageBox.information(self, "提示", "变换成功")
                    self.close()
            else:
                QMessageBox.information(self, "提示", "请先进行变换")
                return

        @staticmethod
        def transformer(image_array, expression, flag=True):
            expr = sympify(expression)
            x = symbols('x')
            image_array = image_array.astype(np.float64)
            numpy_func = lambdify(x, expr, 'numpy')
            result_array = numpy_func(image_array)
            if flag:
                result_array = ImageOperator.ImageProcessorWidget.normalize(result_array)
            else:
                result_array = result_array.astype(np.int16)
                result_array[result_array < 0] = 0
                result_array[result_array > 255] = 255
                result_array = result_array.astype(np.uint8)
            return result_array

    class ImageHistogramEqualization(QWidget):
        def __init__(self, image_info):
            super().__init__()
            self.setWindowTitle("直方图均衡化")

            self.image_info = image_info
            self.image = image_info[0]

            self.original_label = QLabel(self)
            self.equalized_label = QLabel(self)
            self.optimize_label = QLabel(self)

            self.original_histogram_label = QLabel("原始直方图", self)
            self.equalized_histogram_label = QLabel("均衡化直方图", self)
            self.optimize_histogram_label = QLabel("优化算法直方图", self)

            self.equalize_button = QPushButton("均衡化直方图", self)
            self.equalize_button.clicked.connect(self.equalize_histogram)

            self.optimize_button = QPushButton(
                "自适应均衡化直方图(先划分成小块，分别进行限制对比度的均衡化，再双线性插值合并)", self)
            self.optimize_button.clicked.connect(self.optimize_histogram)

            self.apply_button = QPushButton("应用均衡化变换", self)
            self.apply_button.clicked.connect(self.apply_transform)
            self.apply_button_1 = QPushButton("应用自适应均衡化变换", self)
            self.apply_button_1.clicked.connect(self.apply_transform_1)

            self.layout = QVBoxLayout(self)

            # 添加小部件到布局
            layoutH = QHBoxLayout(self)
            layoutH.addWidget(self.original_label)
            layoutH.addWidget(self.equalized_label)
            layoutH.addWidget(self.optimize_label)
            self.layout.addLayout(layoutH)

            self.layout.addWidget(self.original_histogram_label)
            self.layout.addWidget(self.equalized_histogram_label)
            self.layout.addWidget(self.optimize_histogram_label)

            self.layout.addWidget(self.equalize_button)
            self.layout.addWidget(self.optimize_button)

            layoutH = QHBoxLayout(self)
            layoutH.addWidget(self.apply_button)
            layoutH.addWidget(self.apply_button_1)
            self.layout.addLayout(layoutH)

            self.setLayout(self.layout)

            self.image_array = ImageOperator.qimage_to_array(self.image)
            self.original_histogram = ImageOperator.GrayHistogramWidget.calculate_histogram(
                ImageOperator.qimage_to_array(self.image))

            ImageOperator.set_image(self.original_label, self.image, 256, 256)
            ImageOperator.ImageProcessorWidget.display_histogram(self.original_histogram_label, self.image, 200, 768)

            self.equalized_image = None
            self.optimize_image = None
            self.equalized_array = None
            self.optimize_array = None
            self.equalized_histogram = None
            self.optimize_histogram = None

        def equalize_histogram(self):
            self.equalized_array = self.equalize_histogram_array()

            # 将NumPy数组转换回QImage
            self.equalized_image = ImageOperator.array_to_qimage(self.equalized_array)
            # 显示均衡化后的图像
            ImageOperator.set_image(self.equalized_label, self.equalized_image, 256, 256)

            self.equalized_histogram = ImageOperator.GrayHistogramWidget.calculate_histogram(self.equalized_array)
            ImageOperator.ImageProcessorWidget.display_histogram(self.equalized_histogram_label, self.equalized_image,
                                                                 200, 768)
            # 存储均衡化后的图像

        def optimize_histogram(self):
            self.optimize_array = self.optimize_histogram_array()
            # 将NumPy数组转换回QImage
            self.optimize_image = ImageOperator.array_to_qimage(self.optimize_array)
            # 显示均衡化后的图像
            ImageOperator.set_image(self.optimize_label, self.optimize_image, 256, 256)

            self.optimize_histogram = ImageOperator.GrayHistogramWidget.calculate_histogram(self.optimize_array)
            ImageOperator.ImageProcessorWidget.display_histogram(self.optimize_histogram_label, self.optimize_image,
                                                                 200, 768)

        def equalize_histogram_array(self):
            array = self.original_histogram.astype(np.float64)
            array = array / np.sum(array)
            array = np.cumsum(array)
            array = array * 255
            array = array.astype(np.uint8)
            equalized_array = self.image_array.copy()
            height, width = equalized_array.shape
            for y in range(height):
                for x in range(width):
                    equalized_array[y, x] = array[equalized_array[y, x]]
            return equalized_array

        def optimize_histogram_array(self, size=16, clip_limit=0.01):
            # 先划分成小块，分别进行均衡化，再双线性插值合并
            array = self.image_array.astype(np.float64)
            height, width = array.shape
            height_block = int(np.ceil(height / size))
            width_block = int(np.ceil(width / size))
            block_hists = np.zeros((height_block, width_block, 256), dtype=np.float64)
            for i in range(height_block):
                for j in range(width_block):
                    # 考虑边界条件
                    if i == height_block - 1:
                        height_end = height
                    else:
                        height_end = (i + 1) * size
                    if j == width_block - 1:
                        width_end = width
                    else:
                        width_end = (j + 1) * size
                    block = array[i * size:height_end, j * size:width_end]
                    block_hist = ImageOperator.GrayHistogramWidget.calculate_histogram(block)
                    block_hist = block_hist.astype(np.float64)
                    block_hist = block_hist / np.sum(block_hist)
                    over = 0.0
                    for k in range(0, 256):
                        if block_hist[k] > clip_limit:
                            over += block_hist[k] - clip_limit
                            block_hist[k] = clip_limit
                    block_hist += over / 256
                    block_hist = np.cumsum(block_hist)
                    block_hist = block_hist * 255
                    block_hists[i, j, :] = block_hist
            # 双线性插值
            array = array.astype(np.uint8)
            optimize_array = np.zeros((height, width), dtype=np.uint8)
            for i in range(height):
                for j in range(width):
                    # 处理四个角落
                    if (i <= size * 0.5 or i >= (height_block - 1) * size + (
                            (height - (height_block - 1) * size) * 0.5)) and (
                            j < size * 0.5 or j >= (width_block - 1) * size + (
                            (width - (width_block - 1) * size) * 0.5)):
                        gray_scale_value = array[i, j]
                        cdf = block_hists[i // size, j // size, :]
                        optimize_array[i, j] = int(cdf[gray_scale_value])
                    # 上下两条边
                    elif i <= size * 0.5 or i >= (height_block - 1) * size + (
                            (height - (height_block - 1) * size) * 0.5):
                        gray_scale_value = array[i, j]
                        cdf_1 = block_hists[i // size, int((j - size * 0.5) // size), :]
                        cdf_2 = block_hists[i // size, int((j - size * 0.5) // size) + 1, :]
                        cdf_1_abs = j - ((j - size * 0.5) // size * size + size * 0.5)
                        cdf_2_abs = (((j - size * 0.5) // size + 1) * size + size * 0.5) - j
                        optimize_array[i, j] = int(
                            cdf_1[gray_scale_value] * cdf_2_abs / size + cdf_2[gray_scale_value] * cdf_1_abs / size)
                    # 左右两条边
                    elif j <= size * 0.5 or j >= (width_block - 1) * size + ((width - (width_block - 1) * size) * 0.5):
                        gray_scale_value = array[i, j]
                        cdf_1 = block_hists[int((i - size * 0.5) // size), j // size, :]
                        cdf_2 = block_hists[int((i - size * 0.5) // size) + 1, j // size, :]
                        cdf_1_abs = i - ((i - size * 0.5) // size * size + size * 0.5)
                        cdf_2_abs = (((i - size * 0.5) // size + 1) * size + size * 0.5) - i
                        optimize_array[i, j] = int(
                            cdf_1[gray_scale_value] * cdf_2_abs / size + cdf_2[gray_scale_value] * cdf_1_abs / size)
                    # 处理中间一块
                    else:
                        gray_scale_value = array[i, j]
                        cdf_1 = block_hists[int((i - size * 0.5) // size), int((j - size * 0.5) // size), :]
                        cdf_2 = block_hists[int((i - size * 0.5) // size) + 1, int((j - size * 0.5) // size), :]
                        cdf_3 = block_hists[int((i - size * 0.5) // size), int((j - size * 0.5) // size) + 1, :]
                        cdf_4 = block_hists[int((i - size * 0.5) // size) + 1, int((j - size * 0.5) // size) + 1, :]
                        cdf_1_abs = i - ((i - size * 0.5) // size * size + size * 0.5)
                        cdf_2_abs = (((i - size * 0.5) // size + 1) * size + size * 0.5) - i
                        cdf_3_abs = j - ((j - size * 0.5) // size * size + size * 0.5)
                        cdf_4_abs = (((j - size * 0.5) // size + 1) * size + size * 0.5) - j
                        # 先左右插值，后上下插值
                        cdf_1_2 = cdf_1[gray_scale_value] * cdf_2_abs / size + cdf_2[
                            gray_scale_value] * cdf_1_abs / size
                        cdf_3_4 = cdf_3[gray_scale_value] * cdf_2_abs / size + cdf_4[
                            gray_scale_value] * cdf_1_abs / size
                        optimize_array[i, j] = int(cdf_1_2 * cdf_4_abs / size + cdf_3_4 * cdf_3_abs / size)
            return optimize_array

        @staticmethod
        def equalize_histogram_static(image_array, flag=True):  # flag为true时将0计入，否则设0的像素为0
            # 计算直方图
            hist = ImageOperator.GrayHistogramWidget.calculate_histogram(image_array)
            if not flag:
                hist[0] = 0
            # 将直方图归一化
            hist = hist.astype(np.float64)
            hist = hist / np.sum(hist)
            # 计算累积分布函数
            cdf = np.cumsum(hist)
            cdf = cdf * 255
            cdf = cdf.astype(np.uint8)
            # 将像素值映射到新像素值
            equalized_array = np.zeros(image_array.shape, dtype=np.uint8)
            height, width = image_array.shape
            for y in range(height):
                for x in range(width):
                    equalized_array[y, x] = cdf[image_array[y, x]]
            return equalized_array

        @staticmethod
        def optimize_histogram_static(image_array, clip_limit=2.0, tile_size=(8, 8)):  # 使用opencv库
            # Convert to grayscale if the image is in color
            if len(image_array.shape) == 3:
                image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)

            # Create a CLAHE object
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)

            # Apply CLAHE to the image
            optimized_image = clahe.apply(image_array)

            return optimized_image

        def apply_transform(self):
            if self.equalized_image is not None:
                reply = QMessageBox.question(self, '应用均衡化变换', '是否应用该变换？',
                                             QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
                if reply == QMessageBox.StandardButton.Yes:
                    ImagesList.get_instance().imagesList[self.image_info[1]][0] = self.equalized_image
                    ImageViewer.get_instance().update_images(self.image_info[1])
                    QMessageBox.information(self, "提示", "变换成功")
                    self.close()
            else:
                QMessageBox.information(self, "提示", "请先进行均衡化")
                return

        def apply_transform_1(self):
            if self.optimize_image is not None:
                reply = QMessageBox.question(self, '应用自适应均衡化变换', '是否应用该变换？',
                                             QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
                if reply == QMessageBox.StandardButton.Yes:
                    ImagesList.get_instance().imagesList[self.image_info[1]][0] = self.optimize_image
                    ImageViewer.get_instance().update_images(self.image_info[1])
                    QMessageBox.information(self, "提示", "变换成功")
                    self.close()
            else:
                QMessageBox.information(self, "提示", "请先进行自适应均衡化")
                return

    class ObjectDetection(QWidget):
        def __init__(self):
            super().__init__()
            self.threshold = 10

            self.setWindowTitle("物体检测")

            self.image_label_1 = QLabel(self)
            self.image_label_2 = QLabel(self)
            self.image_label_3 = QLabel(self)

            self.image_load_button_1 = QPushButton("加载空场景", self)
            self.image_load_button_1.clicked.connect(self.load_image_1)

            self.image_load_button_2 = QPushButton("加载带物体场景", self)
            self.image_load_button_2.clicked.connect(self.load_image_2)

            self.minus_button = QPushButton("物体检测(均衡化，将黑色像素个数设为0)", self)
            self.minus_button_1 = QPushButton("物体检测(自适应均衡化)", self)

            self.minus_button.clicked.connect(self.object_detection)
            self.minus_button_1.clicked.connect(self.object_detection_1)

            self.checkbox = QtWidgets.QCheckBox("是否先进行均衡化", self)
            self.checkbox_1 = QtWidgets.QCheckBox("是否先进行归一化", self)

            self.textbox = QLineEdit(self)
            self.textbox.setPlaceholderText("输入阈值（10-255）")

            self.layout = QVBoxLayout(self)

            # 添加小部件到布局
            layoutH = QHBoxLayout(self)
            layoutH.addWidget(self.image_label_1)
            layoutH.addWidget(self.image_label_2)
            layoutH.addWidget(self.image_label_3)
            self.layout.addLayout(layoutH)

            layoutH = QHBoxLayout(self)
            layoutH.addWidget(self.image_load_button_1)
            layoutH.addWidget(self.image_load_button_2)
            self.layout.addLayout(layoutH)

            self.layout.addWidget(self.textbox)

            layoutH = QHBoxLayout(self)
            layoutH.addWidget(self.checkbox)
            layoutH.addWidget(self.checkbox_1)
            self.layout.addLayout(layoutH)

            layoutH = QHBoxLayout(self)
            layoutH.addWidget(self.minus_button)
            layoutH.addWidget(self.minus_button_1)
            self.layout.addLayout(layoutH)

            self.setLayout(self.layout)

            self.image_1 = None
            self.image_2 = None
            self.image_3 = None

        def update_threshold(self):
            # 更新阈值并重新绘制图像
            threshold_str = self.textbox.text()
            try:
                if len(threshold_str) == 0:
                    raise ValueError
                threshold = int(threshold_str)
                if self.threshold < 10 or self.threshold > 255:
                    raise ValueError
                self.threshold = threshold
            except ValueError:
                self.threshold = 10
                self.textbox.setText("")
                return

        def load_image_1(self):
            image_info = FileManager.open_image()
            if image_info[0] is None:
                return
            self.image_1 = image_info[0]
            ImageOperator.set_image(self.image_label_1, self.image_1, 256, 256)

        def load_image_2(self):
            image_info = FileManager.open_image()
            if image_info[0] is None:
                return
            self.image_2 = image_info[0]
            ImageOperator.set_image(self.image_label_2, self.image_2, 256, 256)

        def object_detection(self):
            if self.image_1 is None or self.image_2 is None:
                QMessageBox.information(self, "提示", "请先加载图片")
                return
            self.update_threshold()
            image_1 = self.image_1
            image_2 = self.image_2
            image_1_array = ImageOperator.qimage_to_array(image_1)
            image_2_array = ImageOperator.qimage_to_array(image_2)
            if self.checkbox_1.checkState() == Qt.CheckState.Checked:
                image_1_array = ImageOperator.normalize(image_1_array)
                image_2_array = ImageOperator.normalize(image_2_array)
                image_1_array = ImageOperator.ImageProcessorWidget.normalize(image_1_array)
                image_2_array = ImageOperator.ImageProcessorWidget.normalize(image_2_array)
            if self.checkbox.checkState() == Qt.CheckState.Checked:
                image_1_array = ImageOperator.ImageHistogramEqualization.equalize_histogram_static(image_1_array)
                image_2_array = ImageOperator.ImageHistogramEqualization.equalize_histogram_static(image_2_array)
            diff = image_2_array - image_1_array
            # 将绝对值大于10的像素点置为1
            diff = np.abs(diff)
            diff[diff < self.threshold] = 0
            diff[diff >= self.threshold] = 1
            diff_image = ImageOperator.array_to_qimage(diff.astype(np.uint8))
            object_array = ImageOperator.ObjectDetection.mult(diff_image, self.image_2)
            object_array = ImageOperator.ImageHistogramEqualization.equalize_histogram_static(
                object_array.astype(np.uint8), False)
            object_image = ImageOperator.array_to_qimage(object_array)
            ImageOperator.set_image(self.image_label_3, object_image, 256, 256)

        def object_detection_1(self):
            if self.image_1 is None or self.image_2 is None:
                QMessageBox.information(self, "提示", "请先加载图片")
                return
            self.update_threshold()
            image_1 = self.image_1
            image_2 = self.image_2
            image_1_array = ImageOperator.qimage_to_array(image_1)
            image_2_array = ImageOperator.qimage_to_array(image_2)
            if self.checkbox_1.checkState() == Qt.CheckState.Checked:
                image_1_array = ImageOperator.normalize(image_1_array)
                image_2_array = ImageOperator.normalize(image_2_array)
                image_1_array = ImageOperator.ImageProcessorWidget.normalize(image_1_array)
                image_2_array = ImageOperator.ImageProcessorWidget.normalize(image_2_array)
            if self.checkbox.checkState() == Qt.CheckState.Checked:
                image_1_array = ImageOperator.ImageHistogramEqualization.optimize_histogram_static(image_1_array)
                image_2_array = ImageOperator.ImageHistogramEqualization.optimize_histogram_static(image_2_array)
            diff = image_2_array - image_1_array
            # 将绝对值大于10的像素点置为1
            diff = np.abs(diff)
            diff[diff < self.threshold] = 0
            diff[diff >= self.threshold] = 1
            diff_image = ImageOperator.array_to_qimage(diff.astype(np.uint8))
            object_array = ImageOperator.ObjectDetection.mult(diff_image, self.image_2)
            object_array = ImageOperator.ImageHistogramEqualization.optimize_histogram_static(
                object_array.astype(np.uint8))
            object_image = ImageOperator.array_to_qimage(object_array)
            ImageOperator.set_image(self.image_label_3, object_image, 256, 256)

        @staticmethod
        def minus(image1, image2):
            image1_array = ImageOperator.qimage_to_array(image1)
            image2_array = ImageOperator.qimage_to_array(image2)
            image1_array.astype(np.int16)
            image2_array.astype(np.int16)
            image1_array -= image2_array
            return image1_array

        @staticmethod
        def mult(image1, image2):
            image1_array = ImageOperator.qimage_to_array(image1)
            image2_array = ImageOperator.qimage_to_array(image2)
            image1_array.astype(np.int16)
            image1_array = image1_array.astype(np.float64)
            image2_array = image2_array.astype(np.float64)
            image1_array *= image2_array
            return image1_array

    class SpeedCalculator(QWidget):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("速度计算")

            self.image_label_1 = QLabel(self)
            self.image_label_2 = QLabel(self)
            self.image_label_3 = QLabel(self)

            self.image_load_button_1 = QPushButton("加载第一张图片", self)
            self.image_load_button_1.clicked.connect(self.load_image_1)

            self.image_load_button_2 = QPushButton("加载第二张图片", self)
            self.image_load_button_2.clicked.connect(self.load_image_2)

            self.image_load_button_3 = QPushButton("加载第三张图片", self)
            self.image_load_button_3.clicked.connect(self.load_image_3)

            self.calculate_button = QPushButton("计算", self)
            self.calculate_button.clicked.connect(self.calculate)

            self.textbox = QLineEdit(self)
            self.textbox.setPlaceholderText("输入时间差（s）")

            self.layout = QVBoxLayout(self)

            # 添加小部件到布局
            layoutH = QHBoxLayout(self)
            layoutH.addWidget(self.image_label_1)
            layoutH.addWidget(self.image_label_2)
            layoutH.addWidget(self.image_label_3)
            self.layout.addLayout(layoutH)

            layoutH = QHBoxLayout(self)
            layoutH.addWidget(self.image_load_button_1)
            layoutH.addWidget(self.image_load_button_2)
            layoutH.addWidget(self.image_load_button_3)
            self.layout.addLayout(layoutH)

            self.layout.addWidget(self.textbox)

            self.layout.addWidget(self.calculate_button)

            self.setLayout(self.layout)

            self.image_1 = None
            self.image_2 = None
            self.image_3 = None

            self.threshold = 10

        def load_image_1(self):
            image_info = FileManager.open_image()
            if image_info[0] is None:
                return
            self.image_1 = image_info[0]
            ImageOperator.set_image(self.image_label_1, self.image_1, 256, 256)

        def load_image_2(self):
            image_info = FileManager.open_image()
            if image_info[0] is None:
                return
            self.image_2 = image_info[0]
            ImageOperator.set_image(self.image_label_2, self.image_2, 256, 256)

        def load_image_3(self):
            image_info = FileManager.open_image()
            if image_info[0] is None:
                return
            self.image_3 = image_info[0]
            ImageOperator.set_image(self.image_label_3, self.image_3, 256, 256)

        def calculate(self):
            if self.image_1 is None or self.image_2 is None or self.image_3 is None:
                return
            delta_t_str = self.textbox.text()
            try:
                if len(delta_t_str) == 0:
                    raise ValueError
                delta_t = float(delta_t_str)
            except Exception as e:
                QMessageBox.information(self, "提示", "请输入正确的时间差")
                return
            diff = ImageOperator.ObjectDetection.minus(self.image_2, self.image_1)
            diff = np.abs(diff)
            diff[diff < 10] = 0
            diff[diff >= 10] = 1
            # 计算所有为1的点的中心
            nonzero_points = np.argwhere(diff == 1)
            if len(nonzero_points) == 0:
                QMessageBox.information(self, "提示", "未找到目标")
                return
            # 计算坐标平均值，即中心
            center_1 = np.mean(nonzero_points, axis=0)
            diff = ImageOperator.ObjectDetection.minus(self.image_3, self.image_2)
            diff = np.abs(diff)
            diff[diff < 10] = 0
            diff[diff >= 10] = 1
            # 计算所有为1的点的中心
            nonzero_points = np.argwhere(diff == 1)
            if len(nonzero_points) == 0:
                QMessageBox.information(self, "提示", "未找到目标")
                return
            # 计算坐标平均值，即中心
            center_2 = np.mean(nonzero_points, axis=0)
            delta_center = center_2 - center_1
            v = delta_center / delta_t
            v[0] = -v[0]
            v_formatted = ["%.2f" % coord for coord in v]
            message = f"速度为 [{v_formatted[1]}, {v_formatted[0]}] (像素/s)"
            QMessageBox.information(self, "速度", message)


class ImagesList(QObject):
    # 单例
    _instance = None

    def __init__(self):
        super().__init__()
        self.imagesList = []  # 存储列表[image，path,bool]，如果为空path为None，如果修改过bool为True

    @staticmethod
    def get_instance():
        if ImagesList._instance is None:
            ImagesList._instance = ImagesList()
        return ImagesList._instance

    def open_image(self):
        image, file_path, flag = FileManager.open_image()
        if image is not None:
            self.imagesList.append([image, file_path, flag])
            file_name = file_path.split('/')[-1]
            return [image, file_name, flag]
        else:
            return [None, None, False]

    def new_image(self):
        image = FileManager.get_blank_image()
        self.imagesList.append([image, None, True])
        return [image, None, True]

    def save_image(self, num):
        image_info = self.imagesList[num]
        return FileManager.save_image(image_info)

    def save_image_as(self, num):
        image_info = self.imagesList[num]
        return FileManager.save_image_as(image_info)

    def remove_image(self, num):
        if self.imagesList[num][2] is False:
            self.imagesList.pop(num)
            return True
        else:
            reply = QMessageBox.question(None, '保存', '是否保存该图片？',
                                         QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel)
            if reply == QMessageBox.StandardButton.Yes:
                if self.save_image(num):
                    self.imagesList.pop(num)
                    return True
                else:
                    return False
            elif reply == QMessageBox.StandardButton.No:
                self.imagesList.pop(num)
                return True
            else:
                return False


class FileManager:
    @staticmethod
    def open_image(parent=None):
        # 目前打开后会自动转化为灰度图
        file_dialog = QFileDialog(parent)
        file_dialog.setWindowTitle("打开图像")
        file_dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
        file_dialog.setNameFilter("图像文件 (*.png *.jpg *.bmp)")
        if file_dialog.exec() == QFileDialog.DialogCode.Accepted:
            file_path = file_dialog.selectedFiles()[0]
            image = QImage(file_path)
            image = image.convertToFormat(QImage.Format.Format_Grayscale8)
            return [image, file_path, False]
        else:
            return [None, None, False]

    @staticmethod
    def save_image(image_info, parent=None):
        if image_info[1] is None:
            return FileManager.save_image_as(image_info, parent)
        else:
            if image_info[0].save(image_info[1]):
                QMessageBox.information(parent, "保存成功", "保存图像成功")
                image_info[2] = False
                return True
            else:
                QMessageBox.critical(parent, "保存错误", "保存图像失败")
                return False

    @staticmethod
    def save_image_as(image_info, parent=None):
        file_dialog = QFileDialog(parent)
        file_dialog.setWindowTitle("保存图像")
        file_dialog.setFileMode(QFileDialog.FileMode.AnyFile)
        file_dialog.setNameFilter("图像文件 (*.png *.jpg *.bmp)")

        # 自动生成一个默认的文件名，使用当前日期和时间
        default_file_name = f"image_{QDateTime.currentDateTime().toString('yyyyMMdd_hhmmss')}"
        file_dialog.selectFile(default_file_name + ".jpg")

        if file_dialog.exec() == QFileDialog.DialogCode.Accepted:
            file_path = file_dialog.selectedFiles()[0]
            if image_info[0].save(file_path):
                QMessageBox.information(parent, "保存成功", "保存图像成功")
                if image_info[1] is None:
                    image_info[1] = file_path
                    image_info[2] = False
                return True
            else:
                QMessageBox.critical(parent, "保存错误", "保存图像失败")
                return False

    @staticmethod
    def get_blank_image():
        # 创建一个黑色的 QImage 对象
        image = QImage(200, 200, QImage.Format.Format_RGB32)
        image.fill(QColor(255, 255, 0))
        image = image.convertToFormat(QImage.Format.Format_Grayscale8)
        return image