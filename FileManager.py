import heapq
import pickle
import random

import cv2
import numpy as np
import pywt
from PIL import Image, ImageQt
from PyQt6 import QtWidgets, QtGui
from PyQt6.QtCore import QObject, QDateTime, Qt, QPointF
from PyQt6.QtGui import QImage, QColor, qRgb, qGray, QPixmap, QFont, QPainter, qRed, qGreen, qBlue, qAlpha
from PyQt6.QtWidgets import QFileDialog, QMessageBox, QWidget, QVBoxLayout, QLabel, QScrollArea, QLineEdit, QPushButton, \
    QHBoxLayout, QCheckBox, QComboBox
from matplotlib import pyplot as plt
from scipy.signal import convolve2d
from scipy.spatial import KDTree
from sympy import symbols, sympify, lambdify

from Dialog import ParametersInputDialog, AccuracyInputDialog


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

    def matrix_transform(self):
        num = self.check_current_tab()
        if num < 0:
            return
        try:
            image_info = ImagesList.get_instance().imagesList[num]
            self.matrix_transform = ImageOperator.MatrixTransformer((image_info[0], num))
            self.matrix_transform.show()
        except Exception as e:
            print(e)

    def trans_scale_rotate(self):
        num = self.check_current_tab()
        if num < 0:
            return
        try:
            image_info = ImagesList.get_instance().imagesList[num]
            self.matrix_transform = ImageOperator.MatrixTransform((image_info[0], num))
            self.matrix_transform.show()
        except Exception as e:
            print(e)

    def fourier_transform(self):
        num = self.check_current_tab()
        if num < 0:
            return
        try:
            image_info = ImagesList.get_instance().imagesList[num]
            self.fourier_transform = ImageOperator.FourierTransformer((image_info[0], num))
            self.fourier_transform.show()
        except Exception as e:
            print(e)

    def wavelet_transform(self):
        num = self.check_current_tab()
        if num < 0:
            return
        try:
            image_info = ImagesList.get_instance().imagesList[num]
            self.wavelet_transform = ImageOperator.WaveletTransformer((image_info[0], num))
            self.wavelet_transform.show()
        except Exception as e:
            print(e)

    def image_enhancement(self):
        num = self.check_current_tab()
        if num < 0:
            return
        try:
            image_info = ImagesList.get_instance().imagesList[num]
            self.image_enhancement = ImageOperator.Filter((image_info[0], num))
            self.image_enhancement.show()
        except Exception as e:
            print(e)

    def rgb_process(self):
        num = self.check_current_tab()
        if num < 0:
            return
        try:
            image_info = ImagesList.get_instance().imagesList[num]
            if image_info[0].format() != QImage.Format.Format_RGB32:
                QMessageBox.information(self, "提示", "当前图片不是RGB格式")
                return
            self.rgb_process = ImageOperator.RGBProcess((image_info[0], num))
            self.rgb_process.show()
        except Exception as e:
            print(e)

    def edge_detection(self):
        num = self.check_current_tab()
        if num < 0:
            return
        try:
            image_info = ImagesList.get_instance().imagesList[num]

            self.edge_detection = ImageOperator.EdgeDetection((image_info[0], num))
            self.edge_detection.show()
        except Exception as e:
            print(e)

    def huffman_compress(self):
        num = self.check_current_tab()
        if num < 0:
            return
        try:
            if ImagesList.get_instance().imagesList[num][2] is True:
                QMessageBox.information(self, "提示", "当前图片未保存")
                return
            image_info = ImagesList.get_instance().imagesList[num]
            image_array = ImageOperator.qimage_to_array(image_info[0])
            ImageOperator.ImageCompress.huffman(image_info)

        except Exception as e:
            print(e)

    def huffman_decompress(self):
        try:
            dict =FileManager.load_information('huffman')
            if dict is None:
                return
            image_array = ImageOperator.ImageCompress.huffman_decompress(dict)
            image = ImageOperator.array_to_qimage(image_array)
            ImagesList.get_instance().imagesList.append([image, None, True])
            image_list = ImagesList.get_instance().imagesList
            num = 0  # 存储未命名图片的数量
            for i in range(len(image_list)):
                if ImagesList.get_instance().imagesList[i][1] is None:
                    num += 1
            self.add_tab((image_list[-1][0], f"未命名{num}*"))
        except Exception as e:
            print(e)
            QMessageBox.information(self, "提示", "解压失败")

    def rlc_compress(self):
        num = self.check_current_tab()
        if num < 0:
            return
        try:
            if ImagesList.get_instance().imagesList[num][2] is True:
                QMessageBox.information(self, "提示", "当前图片未保存")
                return
            image_info = ImagesList.get_instance().imagesList[num]
            image_array = ImageOperator.qimage_to_array(image_info[0])
            ImageOperator.ImageCompress.rlc_compress(image_info)

        except Exception as e:
            print(e)

    def rlc_decompress(self):
        try:
            dict =FileManager.load_information('rlc')
            if dict is None:
                return
            image_array = ImageOperator.ImageCompress.rlc_decompress(dict)
            image = ImageOperator.array_to_qimage(image_array)
            ImagesList.get_instance().imagesList.append([image, None, True])
            image_list = ImagesList.get_instance().imagesList
            num = 0  # 存储未命名图片的数量
            for i in range(len(image_list)):
                if ImagesList.get_instance().imagesList[i][1] is None:
                    num += 1
            self.add_tab((image_list[-1][0], f"未命名{num}*"))
        except Exception as e:
            print(e)
            QMessageBox.information(self, "提示", "解压失败")

    def huffman_rlc_compress(self):
        num = self.check_current_tab()
        if num < 0:
            return
        try:
            if ImagesList.get_instance().imagesList[num][2] is True:
                QMessageBox.information(self, "提示", "当前图片未保存")
                return
            image_info = ImagesList.get_instance().imagesList[num]
            image_array = ImageOperator.qimage_to_array(image_info[0])
            ImageOperator.ImageCompress.huffman_rlc_compress(image_info)

        except Exception as e:
            print(e)

    def huffman_rlc_decompress(self):
        try:
            dict =FileManager.load_information('huffman_rlc')
            if dict is None:
                return
            image_array = ImageOperator.ImageCompress.huffman_rlc_decompress(dict)
            image = ImageOperator.array_to_qimage(image_array)
            ImagesList.get_instance().imagesList.append([image, None, True])
            image_list = ImagesList.get_instance().imagesList
            num = 0  # 存储未命名图片的数量
            for i in range(len(image_list)):
                if ImagesList.get_instance().imagesList[i][1] is None:
                    num += 1
            self.add_tab((image_list[-1][0], f"未命名{num}*"))
        except Exception as e:
            print(e)
            QMessageBox.information(self, "提示", "解压失败")

    def jpeg_compress(self):
        num = self.check_current_tab()
        if num < 0:
            return
        try:
            if ImagesList.get_instance().imagesList[num][2] is True:
                QMessageBox.information(self, "提示", "当前图片未保存")
                return
            accuracy = AccuracyInputDialog.get_accuracy_static()
            image_info = ImagesList.get_instance().imagesList[num]
            ImageOperator.ImageCompress.jpeg_compress(image_info, accuracy)
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
        image_array = image_array.astype(np.int8)
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
            if bit_planes[0].width() < bit_planes[0].height():
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
                # Convert the bit plane to a QImage
                bit_plane_image = ImageOperator.array_to_qimage(bit_plane)
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
            if image_array.max() != 0:
                image_array /= image_array.max()
            else:
                image_array = np.zeros(image_array.shape)
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

    @staticmethod
    def linear_interpolation(image_array, transform_matrix):
        if np.linalg.det(transform_matrix) != 0:
            # 应用变换
            height, width = image_array.shape
            site_array = np.zeros((height, width, 2), dtype=np.float64)
            # 计算每一个点的新坐标
            for y in range(height):
                for x in range(width):
                    site = np.dot(transform_matrix, np.array([x, y, 1]))
                    site_array[y, x, :] = site[:2] / site[2]
            # 计算新的图像的范围
            x_min, x_max = int(np.ceil(np.min(site_array[:, :, 0]))), int(np.floor(np.max(site_array[:, :, 0])))
            y_min, y_max = int(np.ceil(np.min(site_array[:, :, 1]))), int(np.floor(np.max(site_array[:, :, 1])))
            # 定义新矩阵
            new_array = np.zeros((y_max - y_min + 1, x_max - x_min + 1), dtype=np.float64)
            # 计算逆矩阵
            transform_matrix_inverse = np.linalg.inv(transform_matrix)
            # 计算每一个点的原坐标
            for y in range(y_min, y_max + 1):
                for x in range(x_min, x_max + 1):

                    original_site = np.dot(transform_matrix_inverse, np.array([x, y, 1]))
                    original_site = original_site[:2] / original_site[2]
                    # 判断是否在原图像范围内
                    if original_site[0] >= 0 and original_site[0] + 1 < width and original_site[1] >= 0 and \
                            original_site[1] + 1 < height:
                        # 双线性插值
                        x_1 = int(original_site[0])
                        y_1 = int(original_site[1])
                        x_2 = x_1 + 1
                        y_2 = y_1 + 1
                        x_1_weight = x_2 - original_site[0]
                        x_2_weight = original_site[0] - x_1
                        y_1_weight = y_2 - original_site[1]
                        y_2_weight = original_site[1] - y_1
                        new_array[y - y_min, x - x_min] = x_1_weight * y_1_weight * image_array[
                            y_1, x_1] + x_2_weight * y_1_weight * image_array[y_1, x_2] + x_1_weight * y_2_weight * \
                                                          image_array[y_2, x_1] + x_2_weight * y_2_weight * image_array[
                                                              y_2, x_2]
                    else:
                        new_array[y - y_min, x - x_min] = 255
            return new_array
        else:
            raise ValueError("变换矩阵不可逆")

    class MatrixTransformer(QWidget):
        def __init__(self, image_info):
            super().__init__()

            self.setWindowTitle("矩阵变换")
            self.resize(400, 100)

            self.image_label = QLabel(self)

            self.scroll_area = QScrollArea(self)
            self.scroll_area.setWidgetResizable(True)

            self.scroll_area.setWidget(self.image_label)

            self.matrix_input = QLineEdit(self)
            self.matrix_input.setPlaceholderText("输入变换矩阵，形如[[1, 0, 0], [0, 1, 0], [0, 0, 1]](齐次变换)")

            self.transform_button = QPushButton('矩阵变换', self)
            self.transform_button.clicked.connect(self.apply_transformation)

            self.apply_transform = QPushButton('应用变换', self)
            self.apply_transform.clicked.connect(self.apply_trans)

            layout = QVBoxLayout()
            layout.addWidget(self.scroll_area)

            matrix_layout = QHBoxLayout()
            matrix_layout.addWidget(self.matrix_input)
            matrix_layout.addWidget(self.transform_button)
            layout.addLayout(matrix_layout)

            layout.addWidget(self.apply_transform)
            self.setLayout(layout)

            self.qimage = image_info[0]
            self.image_info = image_info

            self.transformed_image = None

        def apply_trans(self):
            if self.transformed_image is not None:
                reply = QMessageBox.question(self, '应用变换', '是否应用该变换？',
                                             QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
                if reply == QMessageBox.StandardButton.Yes:
                    ImagesList.get_instance().imagesList[self.image_info[1]][0] = self.transformed_image
                    ImageViewer.get_instance().update_images(self.image_info[1])
                    QMessageBox.information(self, "提示", "变换成功")
                    self.close()
            else:
                QMessageBox.information(self, "提示", "请先进行变换")
                return

        def apply_transformation(self):
            matrix_text = self.matrix_input.text()
            try:
                transform_matrix = np.array(eval(matrix_text))
                if not isinstance(transform_matrix, np.ndarray) or transform_matrix.shape != (3, 3):
                    raise ValueError("不合法的矩阵")

                # 调用linear_interpolation方法
                image_array = ImageOperator.qimage_to_array(self.qimage)
                new_array = ImageOperator.linear_interpolation(image_array, transform_matrix)
                self.transformed_image = ImageOperator.array_to_qimage(new_array.astype(np.uint8))
                # 显示变换后的图像
                ImageOperator.set_image(self.image_label, self.transformed_image, self.transformed_image.width(),
                                        self.transformed_image.height())

            except Exception as e:
                print(f"应用矩阵失败: {e}")
                return
            height, width = self.transformed_image.height(), self.transformed_image.width()
            if height > 600:
                height = 600
            if width > 2000:
                width = 2000
            self.scroll_area.setMinimumSize(width + 20, height + 20)
            self.adjustSize()

        @staticmethod
        def rotation_matrix(angle_degrees):
            angle_radians = np.radians(angle_degrees)
            return np.array([
                [np.cos(angle_radians), -np.sin(angle_radians), 0],
                [np.sin(angle_radians), np.cos(angle_radians), 0],
                [0, 0, 1]
            ])

        @staticmethod
        def translation_matrix(tx, ty):
            return np.array([
                [1, 0, tx],
                [0, 1, ty],
                [0, 0, 1]
            ])

        @staticmethod
        def scale_matrix(sx, sy):
            return np.array([
                [sx, 0, 0],
                [0, sy, 0],
                [0, 0, 1]
            ])

    class MatrixTransform(QWidget):
        # 这个widget有旋转，缩放，平移的功能
        def __init__(self, image_info):
            super().__init__()

            self.image_info = image_info
            self.image = image_info[0]

            self.setWindowTitle("平移，旋转，缩放")
            self.resize(400, 100)

            self.check_box = QCheckBox("是否保持变换", self)

            self.text_edit_trans_height = QLineEdit(self)
            self.text_edit_trans_width = QLineEdit(self)
            self.button_trans = QPushButton("平移(自动裁剪，看不出效果)", self)
            self.button_trans.clicked.connect(self.trans)

            self.text_edit_scale_height = QLineEdit(self)
            self.text_edit_scale_width = QLineEdit(self)
            self.button_scale = QPushButton("缩放", self)
            self.button_scale.clicked.connect(self.scale)

            self.text_edit_rotate = QLineEdit(self)
            self.button_rotate = QPushButton("旋转", self)
            self.button_rotate.clicked.connect(self.rotate)

            self.apply_trans = QPushButton("应用变换", self)
            self.apply_trans.clicked.connect(self.apply)

            self.image_label = QLabel(self)
            self.scroll_area = QScrollArea(self)
            self.scroll_area.setWidgetResizable(True)
            self.scroll_area.setWidget(self.image_label)

            self.layout = QVBoxLayout(self)
            self.layout.addWidget(self.scroll_area)

            self.layout.addWidget(self.check_box)

            # 添加平移部分
            w_layout = QHBoxLayout(self)
            w_layout.addWidget(self.text_edit_trans_width)
            w_layout.addWidget(self.text_edit_trans_height)
            w_layout.addWidget(self.button_trans)
            self.layout.addLayout(w_layout)

            # 添加缩放部分
            w_layout = QHBoxLayout(self)
            w_layout.addWidget(self.text_edit_scale_width)
            w_layout.addWidget(self.text_edit_scale_height)
            w_layout.addWidget(self.button_scale)
            self.layout.addLayout(w_layout)

            # 添加旋转部分
            w_layout = QHBoxLayout(self)
            w_layout.addWidget(self.text_edit_rotate)
            w_layout.addWidget(self.button_rotate)
            self.layout.addLayout(w_layout)

            self.layout.addWidget(self.apply_trans)

            self.setLayout(self.layout)

            self.transformed_image = self.image

            self.text_edit_trans_width.setPlaceholderText("输入平移距离(x)")
            self.text_edit_trans_height.setPlaceholderText("输入平移距离(y)")
            self.text_edit_scale_width.setPlaceholderText("输入缩放倍数(x)")
            self.text_edit_scale_height.setPlaceholderText("输入缩放倍数(y)")
            self.text_edit_rotate.setPlaceholderText("输入旋转角度")

            self.is_changed = False

        @staticmethod
        def set_image_in_scroll_area(image, scroll_area, label):
            ImageOperator.set_image(label, image, image.width(),
                                    image.height())
            height, width = image.height(), image.width()
            if height > 600:
                height = 600
            if width > 2000:
                width = 2000
            scroll_area.setMinimumSize(width + 20, height + 20)

        def trans(self):
            try:
                tx = float(self.text_edit_trans_width.text())
                ty = float(self.text_edit_trans_height.text())
                transform_matrix = ImageOperator.MatrixTransformer.translation_matrix(tx, ty)
                if self.check_box.isChecked() is True:
                    image_array = ImageOperator.qimage_to_array(self.transformed_image)
                else:
                    image_array = ImageOperator.qimage_to_array(self.image)
                new_array = ImageOperator.linear_interpolation(image_array, transform_matrix)
                self.transformed_image = ImageOperator.array_to_qimage(new_array.astype(np.uint8))
            except ValueError:
                QMessageBox.information(self, "提示", "请输入正确的平移参数")
                return
            ImageOperator.MatrixTransform.set_image_in_scroll_area(self.transformed_image, self.scroll_area,
                                                                   self.image_label)
            self.adjustSize()
            self.is_changed = True

        def scale(self):
            try:
                sx = float(self.text_edit_scale_width.text())
                sy = float(self.text_edit_scale_height.text())
                transform_matrix = ImageOperator.MatrixTransformer.scale_matrix(sx, sy)
                if self.check_box.isChecked() is True:
                    image_array = ImageOperator.qimage_to_array(self.transformed_image)
                else:
                    image_array = ImageOperator.qimage_to_array(self.image)
                new_array = ImageOperator.linear_interpolation(image_array, transform_matrix)
                self.transformed_image = ImageOperator.array_to_qimage(new_array.astype(np.uint8))
            except ValueError:
                QMessageBox.information(self, "提示", "请输入正确的缩放参数")
                return
            ImageOperator.MatrixTransform.set_image_in_scroll_area(self.transformed_image, self.scroll_area,
                                                                   self.image_label)
            self.adjustSize()
            self.is_changed = True

        def rotate(self):
            try:
                angle = float(self.text_edit_rotate.text())
                transform_matrix = ImageOperator.MatrixTransformer.rotation_matrix(angle)
                if self.check_box.isChecked() is True:
                    image_array = ImageOperator.qimage_to_array(self.transformed_image)
                else:
                    image_array = ImageOperator.qimage_to_array(self.image)
                new_array = ImageOperator.linear_interpolation(image_array, transform_matrix)
                self.transformed_image = ImageOperator.array_to_qimage(new_array.astype(np.uint8))
            except ValueError:
                QMessageBox.information(self, "提示", "请输入正确的旋转参数")
                return
            ImageOperator.MatrixTransform.set_image_in_scroll_area(self.transformed_image, self.scroll_area,
                                                                   self.image_label)
            self.adjustSize()
            self.is_changed = True

        def apply(self):
            if self.is_changed:
                reply = QMessageBox.question(self, '应用变换', '是否应用该变换？',
                                             QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
                if reply == QMessageBox.StandardButton.Yes:
                    ImagesList.get_instance().imagesList[self.image_info[1]][0] = self.transformed_image
                    ImageViewer.get_instance().update_images(self.image_info[1])
                    QMessageBox.information(self, "提示", "变换成功")
                    self.close()
            else:
                QMessageBox.information(self, "提示", "请先进行变换")
                return

    @staticmethod
    def fourier_transform(image_array):
        # 使用傅里叶变换
        f_transform = np.fft.fft2(image_array)
        # 将零频率分量移到图像中心
        f_transform_shifted = np.fft.fftshift(f_transform)
        return f_transform_shifted

    @staticmethod
    def inverse_fourier_transform(f_transform):
        # 将零频率分量移到图像左上角
        f_transform_shifted = np.fft.ifftshift(f_transform)
        # 使用傅里叶逆变换
        img_reconstructed = np.fft.ifft2(f_transform_shifted)
        # 返回傅里叶逆变换后的图像数组
        return img_reconstructed

    @staticmethod
    def cosine_transform(image_array):
        # 使用余弦变换
        c_transform = cv2.dct(image_array.astype(np.float32))
        return c_transform

    @staticmethod
    def inverse_cosine_transform(c_transform):
        # 使用逆余弦变换
        img_reconstructed = cv2.idct(c_transform)
        return img_reconstructed

    class FourierTransformer(QWidget):
        def __init__(self, image_info):
            super().__init__()

            self.image_info = image_info
            self.image = image_info[0]

            self.setWindowTitle("傅里叶变换")
            self.resize(400, 100)

            self.fourier_transform_button = QPushButton("傅里叶变换", self)
            self.fourier_transform_button.clicked.connect(self.fourier_transform)

            self.cosine_transform_button = QPushButton("余弦变换", self)
            self.cosine_transform_button.clicked.connect(self.cosine_transform)

            self.textedit_low = QLineEdit(self)
            self.button_low = QPushButton("低通滤波", self)
            self.button_low.clicked.connect(self.low_pass_filter)

            self.textedit_high = QLineEdit(self)
            self.button_high = (QPushButton("高通滤波", self))
            self.button_high.clicked.connect(self.high_pass_filter)

            self.textedit_mid_low = QLineEdit(self)
            self.textedit_mid_high = QLineEdit(self)
            self.button_mid = (QPushButton("带通滤波", self))
            self.button_mid.clicked.connect(self.mid_pass_filter)

            self.image_label = QLabel(self)

            self.layout = QVBoxLayout(self)
            self.layout.addWidget(self.image_label)
            self.layout.addWidget(self.fourier_transform_button)
            self.layout.addWidget(self.cosine_transform_button)

            h_layout = QHBoxLayout(self)
            h_layout.addWidget(self.textedit_low)
            h_layout.addWidget(self.button_low)
            self.layout.addLayout(h_layout)

            h_layout = QHBoxLayout(self)
            h_layout.addWidget(self.textedit_high)
            h_layout.addWidget(self.button_high)
            self.layout.addLayout(h_layout)

            h_layout = QHBoxLayout(self)
            h_layout.addWidget(self.textedit_mid_low)
            h_layout.addWidget(self.textedit_mid_high)
            h_layout.addWidget(self.button_mid)
            self.layout.addLayout(h_layout)

            self.setLayout(self.layout)

            self.f_transform_shifted = None
            self.trans_type = None

            self.textedit_low.setPlaceholderText("请输入整数")
            self.textedit_high.setPlaceholderText("请输入整数")
            self.textedit_mid_low.setPlaceholderText("请输入整数")
            self.textedit_mid_high.setPlaceholderText("请输入整数")

        def fourier_transform(self):
            image_array = ImageOperator.qimage_to_array(self.image)
            self.f_transform_shifted = ImageOperator.fourier_transform(image_array)
            norm = ImageOperator.ImageProcessorWidget.normalize(np.log(np.abs(self.f_transform_shifted)))
            norm[norm >= 255] = 255
            transformed_image = ImageOperator.array_to_qimage(norm.astype(np.uint8))
            ImageOperator.set_image(self.image_label, transformed_image, 512, 512)
            self.trans_type = "fourier"
            self.adjustSize()

        def array2image(self, array):
            if self.trans_type == "fourier":
                return ImageOperator.inverse_fourier_transform(array)
            if self.trans_type == "cosine":
                return ImageOperator.inverse_cosine_transform(array)
            if self.trans_type == 'wavelet':
                return ImageOperator.inverse_wavelet(array)

        def cosine_transform(self):
            image_array = ImageOperator.qimage_to_array(self.image)
            self.f_transform_shifted = ImageOperator.cosine_transform(image_array)
            norm = ImageOperator.ImageProcessorWidget.normalize(np.log(np.abs(self.f_transform_shifted)))
            norm[norm >= 255] = 255
            transformed_image = ImageOperator.array_to_qimage(norm.astype(np.uint8))
            ImageOperator.set_image(self.image_label, transformed_image, 512, 512)
            self.trans_type = "cosine"
            self.adjustSize()

        def inverse_fourier_transform(self):
            if self.f_transform_shifted is None:
                QMessageBox.information(self, "提示", "请先进行变换")
                return
            inverse_fourier_transform_array = np.abs(ImageOperator.inverse_fourier_transform(self.f_transform_shifted))
            transformed_image = ImageOperator.array_to_qimage(inverse_fourier_transform_array.astype(np.uint8))
            ImageOperator.set_image(self.image_label, transformed_image, 512, 512)
            self.adjustSize()

        def low_pass_filter(self):
            if self.f_transform_shifted is None:
                QMessageBox.information(self, "提示", "请先进行变换")
                return
            # 在已有的 f_transform_shifted 上执行低通滤波操作
            try:
                num = self.textedit_low.text()
                cutoff_frequency = int(num)
                if cutoff_frequency < 0:
                    raise ValueError
            except ValueError as e:
                QMessageBox.information(self, "提示", "请输入正确的截止频率")
                return
            low_pass_f_transform = self.f_transform_shifted.copy()
            rows, cols = low_pass_f_transform.shape
            crow, ccol = int(rows / 2), int(cols / 2)
            if self.trans_type == "cosine":
                mask = np.zeros((rows, cols), np.uint8)
                if crow - cutoff_frequency >= 0 and ccol - cutoff_frequency >= 0 and crow + cutoff_frequency < rows and ccol + cutoff_frequency < cols:
                    mask[crow - cutoff_frequency:crow + cutoff_frequency,
                    ccol - cutoff_frequency:ccol + cutoff_frequency] = 1
                else:
                    QMessageBox.information(self, "提示", "截止频率过大")
                    return
                low_pass_f_transform = low_pass_f_transform * mask
            elif self.trans_type == "fourier":
                if crow - cutoff_frequency >= 0 and ccol - cutoff_frequency >= 0 and crow + cutoff_frequency < rows and ccol + cutoff_frequency < cols:
                    low_pass_f_transform[crow - cutoff_frequency:crow + cutoff_frequency,
                    ccol - cutoff_frequency:ccol + cutoff_frequency] = 0
                else:
                    QMessageBox.information(self, "提示", "截止频率过大")
                    return
            inverse_f_transform = self.array2image(low_pass_f_transform)
            inverse_f_transform_array = np.abs(inverse_f_transform)
            transformed_image = ImageOperator.array_to_qimage(inverse_f_transform_array.astype(np.uint8))
            ImageOperator.set_image(self.image_label, transformed_image, 512, 512)
            self.adjustSize()

        def high_pass_filter(self):
            if self.f_transform_shifted is None:
                QMessageBox.information(self, "提示", "请先进行变换")
                return
            # 在已有的 f_transform_shifted 上执行低通滤波操作
            try:
                num = self.textedit_high.text()
                cutoff_frequency = int(num)
                if cutoff_frequency < 0:
                    raise ValueError
            except ValueError as e:
                QMessageBox.information(self, "提示", "请输入正确的截止频率")
                return

            high_pass_f_transform = self.f_transform_shifted.copy()
            rows, cols = high_pass_f_transform.shape
            crow, ccol = int(rows / 2), int(cols / 2)
            if self.trans_type == "cosine":
                if crow - cutoff_frequency >= 0 and ccol - cutoff_frequency >= 0 and crow + cutoff_frequency < rows and ccol + cutoff_frequency < cols:
                    high_pass_f_transform[crow - cutoff_frequency:crow + cutoff_frequency,
                    ccol - cutoff_frequency:ccol + cutoff_frequency] = 0
                else:
                    QMessageBox.information(self, "提示", "截止频率过大")
                    return
            elif self.trans_type == "fourier":
                mask = np.zeros((rows, cols), np.uint8)
                if crow - cutoff_frequency >= 0 and ccol - cutoff_frequency >= 0 and crow + cutoff_frequency < rows and ccol + cutoff_frequency < cols:
                    mask[crow - cutoff_frequency:crow + cutoff_frequency,
                    ccol - cutoff_frequency:ccol + cutoff_frequency] = 1
                else:
                    QMessageBox.information(self, "提示", "截止频率过大")
                    return
                high_pass_f_transform = high_pass_f_transform * mask
            inverse_f_transform = self.array2image(high_pass_f_transform)
            inverse_f_transform_array = np.abs(inverse_f_transform)

            transformed_image = ImageOperator.array_to_qimage(inverse_f_transform_array.astype(np.uint8))
            ImageOperator.set_image(self.image_label, transformed_image, 512, 512)
            self.adjustSize()

        def mid_pass_filter(self):
            if self.f_transform_shifted is None:
                QMessageBox.information(self, "提示", "请先进行变换")
                return
            # 在已有的 f_transform_shifted 上执行低通滤波操作
            try:
                num = self.textedit_mid_low.text()
                cutoff_frequency_1 = int(num)
                if cutoff_frequency_1 < 0:
                    raise ValueError

                num = self.textedit_mid_high.text()
                cutoff_frequency_2 = int(num)
                if cutoff_frequency_2 < 0:
                    raise ValueError

                if cutoff_frequency_1 >= cutoff_frequency_2:
                    raise ValueError
            except ValueError as e:
                QMessageBox.information(self, "提示", "请输入正确的截止频率")
                return

            mid_pass_f_transform = self.f_transform_shifted.copy()
            rows, cols = mid_pass_f_transform.shape
            crow, ccol = int(rows / 2), int(cols / 2)

            mask = np.zeros((rows, cols), np.uint8)
            if crow - cutoff_frequency_2 >= 0 and ccol - cutoff_frequency_2 >= 0 and crow + cutoff_frequency_2 < rows and ccol + cutoff_frequency_2 < cols:
                mask[crow - cutoff_frequency_2:crow + cutoff_frequency_2,
                ccol - cutoff_frequency_2:ccol + cutoff_frequency_2] = 1
            else:
                QMessageBox.information(self, "提示", "截止频率过大")
                return
            mid_pass_f_transform = mid_pass_f_transform * mask

            if crow - cutoff_frequency_1 >= 0 and ccol - cutoff_frequency_1 >= 0 and crow + cutoff_frequency_1 < rows and ccol + cutoff_frequency_1 < cols:
                mid_pass_f_transform[crow - cutoff_frequency_1:crow + cutoff_frequency_1,
                ccol - cutoff_frequency_1:ccol + cutoff_frequency_1] = 0
            else:
                QMessageBox.information(self, "提示", "截止频率过大")
                return

            inverse_f_transform = self.array2image(mid_pass_f_transform)
            inverse_f_transform_array = np.abs(inverse_f_transform)

            transformed_image = ImageOperator.array_to_qimage(inverse_f_transform_array.astype(np.uint8))
            ImageOperator.set_image(self.image_label, transformed_image, 512, 512)
            self.adjustSize()

    class WaveletTransformer(QWidget):
        def __init__(self, image_info):
            super().__init__()

            self.image = image_info[0]
            self.image_array = ImageOperator.qimage_to_array(self.image)
            self.coeffs = None

            self.setWindowTitle("小波变换")
            self.resize(400, 400)

            self.transform_button = QPushButton("执行小波变换", self)
            self.transform_button.clicked.connect(self.perform_wavelet_transform)

            self.images = []
            self.widget = QWidget(self)

            for i in range(4):
                self.images.append(QLabel(self.widget))
            layout = QVBoxLayout(self.widget)
            h_layout = QHBoxLayout(self.widget)
            for i in range(4):
                h_layout.addWidget(self.images[i])
                if (i + 1) % 2 == 0:
                    layout.addLayout(h_layout)
                    h_layout = QHBoxLayout(self.widget)

            layout_1 = QVBoxLayout(self)
            layout_1.addWidget(self.widget)
            layout_1.addWidget(self.transform_button)
            self.setLayout(layout_1)

        def perform_wavelet_transform(self):
            LLY, (LHY, HLY, HHY) = pywt.dwt2(self.image_array, 'haar')

            # 显示小波分解的图像部分
            self.show_wavelet_image(LLY, 0)
            self.show_wavelet_image(LHY, 1)
            self.show_wavelet_image(HLY, 2)
            self.show_wavelet_image(HHY, 3)

        def show_wavelet_image(self, component, index):
            component = ImageOperator.ImageProcessorWidget.normalize(component)
            qimage = ImageOperator.array_to_qimage(np.array(component))
            ImageOperator.set_image(self.images[index], qimage, 256, 256)

    class Filter(QWidget):
        def __init__(self, image_info):
            super().__init__()

            self.result = None
            self.image = image_info[0]
            self.image_array = ImageOperator.qimage_to_array(self.image)

            self.setWindowTitle("图像增强")
            self.resize(400, 400)

            self.filter_type_combo = QComboBox(self)
            self.filter_type_combo.addItem("邻域平均")
            self.filter_type_combo.addItem("超限像素平滑")
            self.filter_type_combo.addItem("中值滤波")
            self.filter_type_combo.addItem("K近邻均值滤波")
            self.filter_type_combo.addItem("最小均方差滤波")
            self.filter_type_combo.addItem("理想低通滤波")
            self.filter_type_combo.addItem("Butterworth低通滤波")
            self.filter_type_combo.addItem("指数低通滤波")
            self.filter_type_combo.addItem("梯形低通滤波")
            self.filter_button = QPushButton("平滑", self)

            self.filter_button.clicked.connect(self.filter)

            self.sharpen_type_combo = QComboBox(self)
            self.sharpen_type_combo.addItem("梯度锐化(1)")
            self.sharpen_type_combo.addItem("梯度锐化(2)")
            self.sharpen_type_combo.addItem("梯度锐化(3)")
            self.sharpen_type_combo.addItem("梯度锐化(4)")
            self.sharpen_type_combo.addItem("梯度锐化(5)")
            self.sharpen_type_combo.addItem("拉普拉斯锐化")
            self.sharpen_type_combo.addItem("高通滤波")
            self.sharpen_type_combo.addItem("Sobel锐化")
            self.sharpen_type_combo.addItem("Prewitt锐化")
            self.sharpen_type_combo.addItem("Isotropic锐化")
            self.sharpen_type_combo.addItem("理想高通滤波")
            self.sharpen_type_combo.addItem("Butterworth高通滤波")
            self.sharpen_type_combo.addItem("指数高通滤波")
            self.sharpen_type_combo.addItem("梯形高通滤波")
            self.sharpen_button = QPushButton("锐化", self)

            self.sharpen_button.clicked.connect(self.sharpen)

            self.image_label = QLabel(self)

            self.layout = QVBoxLayout(self)
            self.layout.addWidget(self.image_label)

            layout = QHBoxLayout(self)
            layout.addWidget(self.filter_type_combo)
            layout.addWidget(self.filter_button)
            self.layout.addLayout(layout)

            layout = QHBoxLayout(self)
            layout.addWidget(self.sharpen_type_combo)
            layout.addWidget(self.sharpen_button)
            self.layout.addLayout(layout)

            self.setLayout(self.layout)

        def filter(self):
            type = self.filter_type_combo.currentText()
            if type == "邻域平均":
                self.result = self.neighborhood_average_filter(self.image_array, flag=True)
            elif type == "超限像素平滑":
                self.result = self.outlier_pixel_smoothing(self.image_array)
            elif type == "中值滤波":
                self.result = self.neighborhood_average_filter(self.image_array, flag=False)
            elif type == "K近邻均值滤波":
                self.result = self.k_nearest_neighbor_mean_filter(self.image_array)
            elif type == "最小均方差滤波":
                self.result = self.minimum_mean_square_filter(self.image_array)
            elif type == "理想低通滤波":
                self.result = self.ideal_low_pass_filter(self.image_array)
            elif type == "Butterworth低通滤波":
                self.result = self.butterworth_low_pass_filter(self.image_array)
            elif type == "指数低通滤波":
                self.result = self.exponential_low_pass_filter(self.image_array)
            elif type == "梯形低通滤波":
                self.result = self.trapezoidal_low_pass_filter(self.image_array)
            else:
                return
            qimage = ImageOperator.array_to_qimage(np.array(self.result))
            ImageOperator.set_image(self.image_label, qimage, 512, 512)

        def sharpen(self):
            type = self.sharpen_type_combo.currentText()
            if type == "梯度锐化(1)":
                self.result = self.gradient_sharpen(self.image_array, 0)
            elif type == "梯度锐化(2)":
                self.result = self.gradient_sharpen(self.image_array, 1)
            elif type == "梯度锐化(3)":
                self.result = self.gradient_sharpen(self.image_array, 2)
            elif type == "梯度锐化(4)":
                self.result = self.gradient_sharpen(self.image_array, 3)
            elif type == "梯度锐化(5)":
                self.result = self.gradient_sharpen(self.image_array, 4)
            elif type == "拉普拉斯锐化":
                self.result = self.universal_convolve(self.image_array)
            elif type == "高通滤波":
                self.result = self.universal_convolve(self.image_array,
                                                      np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]))
            elif type == "Sobel锐化":
                self.result_1 = self.universal_convolve(self.image_array,
                                                        np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]), flag=False)
                self.result_2 = self.universal_convolve(self.image_array,
                                                        np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]), flag=False)
                self.result = np.sqrt(self.result_1 ** 2 + self.result_2 ** 2)
            elif type == "Prewitt锐化":
                self.result_1 = self.universal_convolve(self.image_array,
                                                        np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]), flag=False)
                self.result_2 = self.universal_convolve(self.image_array,
                                                        np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]), flag=False)
                self.result = np.sqrt(self.result_1 ** 2 + self.result_2 ** 2)
            elif type == "Isotropic锐化":
                self.result_1 = self.universal_convolve(self.image_array,
                                                        np.array(
                                                            [[-1, -np.sqrt(2), -1], [0, 0, 0], [1, np.sqrt(2), 1]]),
                                                        flag=False)
                self.result_2 = self.universal_convolve(self.image_array, np.array(
                    [[-1, 0, 1], [-np.sqrt(2), 0, np.sqrt(2)], [-1, 0, 1]]), flag=False)
                self.result = np.sqrt(self.result_1 ** 2 + self.result_2 ** 2)
            elif type == "理想高通滤波":
                self.result = self.ideal_high_pass_filter(self.image_array)
            elif type == "Butterworth高通滤波":
                self.result = self.butterworth_high_pass_filter(self.image_array)
            elif type == "指数高通滤波":
                self.result = self.exponential_high_pass_filter(self.image_array)
            elif type == "梯形高通滤波":
                self.result = self.trapezoidal_high_pass_filter(self.image_array)
            else:
                return
            qimage = ImageOperator.array_to_qimage(np.array(self.result))
            ImageOperator.set_image(self.image_label, qimage, 512, 512)

        @staticmethod
        def neighborhood_average_filter(image_array, flag=True):
            height, width = image_array.shape
            result = image_array.copy()

            for i in range(1, height - 1):
                for j in range(1, width - 1):
                    neighborhood = image_array[i - 1:i + 2, j - 1:j + 2]
                    if flag:
                        result[i, j] = np.mean(neighborhood)
                    else:
                        result[i, j] = np.median(neighborhood)
            return result

        @staticmethod
        def outlier_pixel_smoothing(image_array, threshold=20,
                                    convolution_kernel=np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16):
            smoothed_image = np.copy(image_array)
            for i in range(1, image_array.shape[0] - 2):
                for j in range(1, image_array.shape[1] - 2):
                    neighborhood = image_array[i - 1:i + 2, j - 1:j + 2]
                    if np.abs(image_array[i, j] - np.sum(neighborhood * convolution_kernel)) > threshold:
                        smoothed_image[i, j] = np.sum(neighborhood * convolution_kernel)
            return smoothed_image

        @staticmethod
        def k_nearest_neighbor_mean_filter(image_array, k=5, flag=True):
            height, width = image_array.shape
            result = np.copy(image_array)
            flat_image = image_array.flatten()
            kdtree = KDTree(flat_image.reshape(-1, 1))
            for i in range(height):
                for j in range(width):
                    pixel_value = image_array[i, j]
                    _, indices = kdtree.query(pixel_value, k + 1)
                    neighbor_values = flat_image[indices[1:]]
                    if flag:
                        result[i, j] = np.mean(neighbor_values)
                    else:
                        result[i, j] = np.median(neighbor_values)
            return result

        @staticmethod
        def minimum_mean_square_filter(image_array):
            templates = [
                [[0, 1, 1, 1, 0], [0, 1, 1, 1, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
                [[0, 0, 0, 0, 0], [1, 1, 0, 0, 0], [1, 1, 1, 0, 0], [1, 1, 0, 0, 0], [0, 0, 0, 0, 0]],
                [[0, 0, 0, 0, 0], [0, 0, 0, 1, 1], [0, 0, 1, 1, 1], [0, 0, 0, 1, 1], [0, 0, 0, 0, 0]],
                [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 1, 1, 1, 0], [0, 1, 1, 1, 0]],
                [[1, 1, 0, 0, 0], [1, 1, 1, 0, 0], [0, 1, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
                [[0, 0, 0, 1, 1], [0, 0, 1, 1, 1], [0, 0, 1, 1, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
                [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 1, 1, 0, 0], [1, 1, 1, 0, 0], [1, 1, 0, 0, 0]],
                [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 1, 1, 0], [0, 0, 1, 1, 1], [0, 0, 0, 1, 1]],
                [[0, 0, 0, 0, 0], [0, 1, 1, 1, 0], [0, 1, 1, 1, 0], [0, 1, 1, 1, 0], [0, 0, 0, 0, 0]],
            ]
            templates = np.array(templates)
            height, width = image_array.shape
            result = image_array.copy()
            for i in range(2, height - 3):
                for j in range(2, width - 3):
                    neighborhood = image_array[i - 2:i + 3, j - 2:j + 3]
                    min_mean_square = np.inf
                    best_template = None
                    for template in templates:
                        template_values = neighborhood[template == 1]
                        mean_square = np.mean(np.square(neighborhood - np.mean(template_values)))
                        if mean_square < min_mean_square:
                            min_mean_square = mean_square
                            best_template = template
                    result[i, j] = np.mean(neighborhood[best_template == 1])
            return result

        @staticmethod
        def gradient_sharpen(image_array, type=0, threshold=50):
            height, width = image_array.shape
            array = image_array.astype(np.float64)
            result = image_array.copy().astype(np.float64)
            for i in range(1, height - 1):
                for j in range(1, width - 1):
                    dx = array[i, j] - array[i - 1, j]
                    dy = array[i, j] - array[i, j - 1]
                    gradient = np.sqrt(dx * dx + dy * dy)
                    if type == 0:
                        result[i, j] = int(gradient)
                    elif type == 1:
                        if gradient > threshold:
                            result[i, j] = int(gradient)
                        else:
                            result[i, j] = int(array[i, j])
                    elif type == 2:
                        if gradient > threshold:
                            result[i, j] = 0
                        else:
                            result[i, j] = int(array[i, j])
                    elif type == 3:
                        if gradient > threshold:
                            result[i, j] = int(gradient)
                        else:
                            result[i, j] = 0
                    elif type == 4:
                        if gradient > threshold:
                            result[i, j] = 0
                        else:
                            result[i, j] = 255
            result = np.clip(result, 0, 255)
            result = result.astype(np.uint8)
            return result

        @staticmethod
        def universal_convolve(image_array, operator=np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]), flag=True):
            result = convolve2d(image_array, operator, mode='same', boundary='wrap')
            # 如果需要将结果限制在0到255之间，可以使用以下代码
            if flag:
                result = np.clip(result, 0, 255).astype(np.uint8)
            return result

        @staticmethod
        def ideal_low_pass_filter(image_array, cutoff_frequency=0.2):
            f_transform_shifted = ImageOperator.fourier_transform(image_array)
            rows, cols = image_array.shape
            cutoff_frequency = int(cutoff_frequency * min(rows, cols))
            crow, ccol = int(rows / 2), int(cols / 2)
            mask = np.zeros((rows, cols), np.uint8)
            mask[crow - cutoff_frequency:crow + cutoff_frequency,
            ccol - cutoff_frequency:ccol + cutoff_frequency] = 1
            f_transform_shifted = f_transform_shifted * mask
            inverse_f_transform = ImageOperator.inverse_fourier_transform(f_transform_shifted)
            return np.abs(inverse_f_transform)

        @staticmethod
        def butterworth_low_pass_filter(image_array, cutoff_frequency=0.1, order=2):
            f_transform_shifted = ImageOperator.fourier_transform(image_array)
            rows, cols = image_array.shape
            cutoff_frequency = int(cutoff_frequency * min(rows, cols))
            crow, ccol = int(rows / 2), int(cols / 2)
            mask = np.zeros((rows, cols), np.float64)
            for i in range(rows):
                for j in range(cols):
                    mask[i, j] = 1 / (1 + np.sqrt((i - crow) ** 2 + (j - ccol) ** 2) / cutoff_frequency ** 2) ** (
                            2 * order)
            f_transform_shifted = f_transform_shifted * mask
            inverse_f_transform = ImageOperator.inverse_fourier_transform(f_transform_shifted)
            return np.abs(inverse_f_transform)

        @staticmethod
        def exponential_low_pass_filter(image_array, cutoff_frequency=0.2, order=2):
            f_transform_shifted = ImageOperator.fourier_transform(image_array)
            rows, cols = image_array.shape
            cutoff_frequency = int(cutoff_frequency * min(rows, cols))
            crow, ccol = int(rows / 2), int(cols / 2)
            mask = np.zeros((rows, cols), np.float64)
            for i in range(rows):
                for j in range(cols):
                    mask[i, j] = np.exp(-((i - crow) ** 2 + (j - ccol) ** 2) / cutoff_frequency ** 2) ** (order / 2)
            f_transform_shifted = f_transform_shifted * mask
            inverse_f_transform = ImageOperator.inverse_fourier_transform(f_transform_shifted)
            return np.abs(inverse_f_transform)

        @staticmethod
        def trapezoidal_low_pass_filter(image_array, cutoff_frequency_1=0.2, cutoff_frequency_2=0.6):
            f_transform_shifted = ImageOperator.fourier_transform(image_array)
            rows, cols = image_array.shape
            cutoff_frequency_1 = int(cutoff_frequency_1 * min(rows, cols))
            cutoff_frequency_2 = int(cutoff_frequency_2 * min(rows, cols))
            crow, ccol = int(rows / 2), int(cols / 2)
            mask = np.zeros((rows, cols), np.float64)
            for i in range(rows):
                for j in range(cols):
                    if (i - crow) ** 2 + (j - ccol) ** 2 < cutoff_frequency_1 ** 2:
                        mask[i, j] = 1
                    elif (i - crow) ** 2 + (j - ccol) ** 2 < cutoff_frequency_2 ** 2:
                        mask[i, j] = np.sqrt((cutoff_frequency_2 ** 2 - (i - crow) ** 2 - (j - ccol) ** 2) / (
                                cutoff_frequency_2 ** 2 - cutoff_frequency_1 ** 2))
                    else:
                        mask[i, j] = 0
            f_transform_shifted = f_transform_shifted * mask
            inverse_f_transform = ImageOperator.inverse_fourier_transform(f_transform_shifted)
            return np.abs(inverse_f_transform)

        @staticmethod
        def ideal_high_pass_filter(image_array, cutoff_frequency=0.2):
            f_transform_shifted = ImageOperator.fourier_transform(image_array)
            rows, cols = image_array.shape
            cutoff_frequency = int(cutoff_frequency * min(rows, cols))
            crow, ccol = int(rows / 2), int(cols / 2)
            mask = np.ones((rows, cols), np.uint8)
            mask[crow - cutoff_frequency:crow + cutoff_frequency,
            ccol - cutoff_frequency:ccol + cutoff_frequency] = 0
            f_transform_shifted = f_transform_shifted * mask
            inverse_f_transform = ImageOperator.inverse_fourier_transform(f_transform_shifted)
            return np.abs(inverse_f_transform)

        @staticmethod
        def butterworth_high_pass_filter(image_array, cutoff_frequency=0.1, order=2):
            f_transform_shifted = ImageOperator.fourier_transform(image_array)
            rows, cols = image_array.shape
            cutoff_frequency = int(cutoff_frequency * min(rows, cols))
            crow, ccol = int(rows / 2), int(cols / 2)
            mask = np.zeros((rows, cols), np.float64)
            for i in range(rows):
                for j in range(cols):
                    if (i - crow) ** 2 + (j - ccol) ** 2 == 0:
                        mask[i, j] = 0
                    else:
                        mask[i, j] = 1 / (1 + np.sqrt(cutoff_frequency ** 2 / ((i - crow) ** 2 + (j - ccol) ** 2))) ** (
                                2 * order)
            f_transform_shifted = f_transform_shifted * mask
            inverse_f_transform = ImageOperator.inverse_fourier_transform(f_transform_shifted)
            return np.abs(inverse_f_transform)

        @staticmethod
        def exponential_high_pass_filter(image_array, cutoff_frequency=0.2, order=2):
            f_transform_shifted = ImageOperator.fourier_transform(image_array)
            rows, cols = image_array.shape
            cutoff_frequency = int(cutoff_frequency * min(rows, cols))
            crow, ccol = int(rows / 2), int(cols / 2)
            mask = np.zeros((rows, cols), np.float64)
            for i in range(rows):
                for j in range(cols):
                    if (i - crow) ** 2 + (j - ccol) ** 2 == 0:
                        mask[i, j] = 0
                    else:
                        mask[i, j] = np.exp(
                            -(cutoff_frequency ** 2 / ((i - crow) ** 2 + (j - ccol) ** 2) ** (order / 2)))
            f_transform_shifted = f_transform_shifted * mask
            inverse_f_transform = ImageOperator.inverse_fourier_transform(f_transform_shifted)
            return np.abs(inverse_f_transform)

        @staticmethod
        def trapezoidal_high_pass_filter(image_array, cutoff_frequency_1=0.2, cutoff_frequency_2=0.6):
            f_transform_shifted = ImageOperator.fourier_transform(image_array)
            rows, cols = image_array.shape
            cutoff_frequency_1 = int(cutoff_frequency_1 * min(rows, cols))
            cutoff_frequency_2 = int(cutoff_frequency_2 * min(rows, cols))
            crow, ccol = int(rows / 2), int(cols / 2)
            mask = np.zeros((rows, cols), np.float64)
            for i in range(rows):
                for j in range(cols):
                    if (i - crow) ** 2 + (j - ccol) ** 2 < cutoff_frequency_1 ** 2:
                        mask[i, j] = 0
                    elif (i - crow) ** 2 + (j - ccol) ** 2 < cutoff_frequency_2 ** 2:
                        mask[i, j] = np.sqrt((cutoff_frequency_2 ** 2 - (i - crow) ** 2 - (j - ccol) ** 2) / (
                                cutoff_frequency_2 ** 2 - cutoff_frequency_1 ** 2))
                    else:
                        mask[i, j] = 1
            f_transform_shifted = f_transform_shifted * mask
            inverse_f_transform = ImageOperator.inverse_fourier_transform(f_transform_shifted)
            return np.abs(inverse_f_transform)

    @staticmethod
    def rgb2array(qimage):
        qimage = qimage.convertToFormat(QImage.Format.Format_RGB32)
        width = qimage.width()
        height = qimage.height()
        image_array = np.zeros((height, width, 4), dtype=np.uint8)
        for i in range(height):
            for j in range(width):
                pixel = qimage.pixel(j, i)
                image_array[i, j, 0] = qRed(pixel)
                image_array[i, j, 1] = qGreen(pixel)
                image_array[i, j, 2] = qBlue(pixel)
                image_array[i, j, 3] = qAlpha(pixel)
        return image_array[..., :3]  # 剥离Alpha通道并返回RGB数据

    @staticmethod
    def draw_rgb_histogram(image_array, height, width) -> QImage:
        image_array = ImageOperator.ImageProcessorWidget.normalize(image_array)
        histogram = ImageOperator.GrayHistogramWidget.calculate_histogram(image_array)
        histogram_pixmap = QPixmap(256, height)
        histogram_pixmap.fill(Qt.GlobalColor.white)
        painter = QPainter(histogram_pixmap)
        painter.setPen(QColor(Qt.GlobalColor.black))
        max_hist_value = max(histogram)
        scale_factor = height / max_hist_value if max_hist_value > 0 else 1.0
        for i in range(256):
            scaled_height = int(histogram[i] * scale_factor)
            painter.drawLine(i, height, i, height - scaled_height)
        painter.end()
        histogram_pixmap = histogram_pixmap.scaled(width, height, Qt.AspectRatioMode.IgnoreAspectRatio)
        qimage = QImage(histogram_pixmap.toImage())
        return qimage

    @staticmethod
    def array_to_qimage_rgb(image_array):
        height, width, channel = image_array.shape
        qimage = QImage(width, height, QImage.Format.Format_RGB32)
        for i in range(height):
            for j in range(width):
                qimage.setPixel(j, i, qRgb(image_array[i, j, 0], image_array[i, j, 1], image_array[i, j, 2]))
        return qimage

    class RGBProcess(QWidget):
        def __init__(self, image_info):
            super().__init__()

            self.setWindowTitle("RGB图像")
            self.resize(400, 400)

            self.image = image_info[0]

            self.rgb2yuv_button = QPushButton("RGB转YUV", self)
            self.label_1 = QLabel(self)
            self.label_1.setText("Y")
            self.label_2 = QLabel(self)
            self.label_2.setText("U")
            self.label_3 = QLabel(self)
            self.label_3.setText("V")

            self.image_1 = QLabel(self)
            self.image_2 = QLabel(self)
            self.image_3 = QLabel(self)

            self.rgb2_256_button = QPushButton("RGB转256(统计出现频率最高的256色)", self)
            self.image_label = QLabel(self)

            self.rgb2yuv_button.clicked.connect(self.rgb2yuv)
            self.rgb2_256_button.clicked.connect(self.rgb2_256_custom)

            self.layout = QVBoxLayout(self)
            self.layout.addWidget(self.image_label)

            layout = QHBoxLayout(self)
            layout.addWidget(self.image_1)
            layout.addWidget(self.image_2)
            layout.addWidget(self.image_3)
            self.layout.addLayout(layout)

            layout = QHBoxLayout(self)
            layout.addWidget(self.label_1)
            layout.addWidget(self.label_2)
            layout.addWidget(self.label_3)
            self.layout.addLayout(layout)

            self.layout.addWidget(self.rgb2yuv_button)
            self.layout.addWidget(self.rgb2_256_button)

            self.setLayout(self.layout)

            ImageOperator.set_image(self.image_label, self.image, 200 * 3, 200 * 3)

        @staticmethod
        def rgb2yuv_static(image_array):
            yuv_image = np.zeros(image_array.shape, dtype=np.int8)
            yuv_image[..., 0] = 0.299 * image_array[..., 0] + 0.587 * image_array[..., 1] + 0.114 * image_array[..., 2]
            yuv_image[..., 1] = -0.169 * image_array[..., 0] - 0.332 * image_array[..., 1] + 0.500 * image_array[..., 2]
            yuv_image[..., 2] = 0.500 * image_array[..., 0] - 0.419 * image_array[..., 1] - 0.081 * image_array[..., 2]
            return yuv_image

        def rgb2yuv(self):
            image_array = ImageOperator.rgb2array(self.image)
            yuv_array = self.rgb2yuv_static(image_array)
            image = ImageOperator.draw_rgb_histogram(yuv_array[..., 0], 128, 256)
            ImageOperator.set_image(self.label_1, image, 256, 128)
            ImageOperator.set_image(self.image_1, ImageOperator.array_to_qimage(yuv_array[..., 0]), 256, 200)
            image = ImageOperator.draw_rgb_histogram(yuv_array[..., 1], 128, 256)
            ImageOperator.set_image(self.label_2, image, 256, 128)
            ImageOperator.set_image(self.image_2, ImageOperator.array_to_qimage(yuv_array[..., 1]), 256, 200)
            image = ImageOperator.draw_rgb_histogram(yuv_array[..., 2], 128, 256)
            ImageOperator.set_image(self.label_3, image, 256, 128)
            ImageOperator.set_image(self.image_3, ImageOperator.array_to_qimage(yuv_array[..., 2]), 256, 200)
            ImageOperator.set_image(self.image_label, self.image, 200 * 3, 200 * 3)

        def rgb2_256(self):
            # image=self.image.convertToFormat(QImage.Format.Format_Indexed8)
            pil_image = Image.fromqimage(self.image)
            # 将PIL Image转换为256色图像
            quantized_image = pil_image.quantize(colors=256)
            # 转换为QImage
            image = ImageQt.toqimage(quantized_image)
            ImageOperator.set_image(self.image_label, image, 200 * 3, 200 * 3)

        def rgb2_256_custom(self):
            # 将图像转换为NumPy数组
            image_array = ImageOperator.rgb2array(self.image)
            image_array = image_array.reshape(-1, 3)

            # 首先截断颜色为12位
            truncated_colors = []
            for color in image_array:
                truncated_color = (color[0] >> 4, color[1] >> 4, color[2] >> 4)
                truncated_colors.append(truncated_color)

            # 统计每种颜色的出现次数
            color_count = {}
            for color in truncated_colors:
                color = tuple(color)
                if color in color_count:
                    color_count[color] += 1
                else:
                    color_count[color] = 1

            # 按出现次数降序排序颜色
            sorted_colors = sorted(color_count.items(), key=lambda x: x[1], reverse=True)

            # 选择前256种颜色
            selected_colors = sorted_colors[:256]

            # 生成调色板
            palette = np.array(
                [color[0] for color in selected_colors])

            # 将图像的每个像素映射到最接近的颜色
            image_array = np.array(truncated_colors)
            image_array = image_array.reshape(-1, 1, 3)

            # 对于每个像素，找到最接近的颜色
            distances = np.sum(np.square(image_array - palette), axis=2)
            indices = np.argmin(distances, axis=1)
            image_array = palette[indices]
            image_array = image_array.reshape(self.image.height(), self.image.width(), 3)

            # 将NumPy数组转换为QImage
            image = ImageOperator.array_to_qimage_rgb(image_array * 16)
            ImageOperator.set_image(self.image_label, image, 200 * 3, 200 * 3)

    class EdgeDetection(QWidget):
        def __init__(self, image_info):
            super().__init__()

            self.setWindowTitle("边缘检测")
            self.resize(400, 400)

            self.image = image_info[0]

            self.combo_box = QComboBox(self)
            self.combo_box.addItem("Sobel算子")
            self.combo_box.addItem("Prewitt算子")
            self.combo_box.addItem("Laplacian算子")
            self.combo_box.addItem("5X5拉普拉斯-高斯卷积核")

            self.button = QPushButton("确定", self)
            self.button.clicked.connect(self.edge_detection_0)

            self.button_1 = QPushButton("霍夫变换(上面的输入框为直线变换的阈值)")
            self.button_1.clicked.connect(self.hough)

            self.image_label = QLabel(self)

            self.text_edit_low = QLineEdit(self)
            self.text_edit_low.setText("0.2")
            self.text_edit_high = QLineEdit(self)
            self.text_edit_high.setText("0.4")

            self.button_2 = QPushButton("Canny", self)
            self.button_2.clicked.connect(self.my_canny)

            self.button_3 = QPushButton("边缘跟踪（使用上面的低阈值，效果一般，对于霍夫算法的图，阈值可设定为0.1）", self)
            self.button_3.clicked.connect(self.edge_track)


            self.text_edit = QLineEdit(self)
            self.text_edit.setText("0.3")

            self.layout = QVBoxLayout(self)
            self.layout.addWidget(self.image_label)


            layout = QHBoxLayout(self)
            layout.addWidget(self.combo_box)
            layout.addWidget(self.button)
            self.layout.addLayout(layout)

            layout = QHBoxLayout(self)
            layout.addWidget(self.text_edit_low)
            layout.addWidget(self.text_edit_high)
            self.layout.addLayout(layout)

            self.layout.addWidget(self.button_2)
            self.layout.addWidget(self.button_3)

            self.layout.addWidget(self.text_edit)
            self.layout.addWidget(self.button_1)

            self.setLayout(self.layout)
            self.low = 0.2
            self.high = 0.4
            self.threshold=0.3
            ImageOperator.set_image(self.image_label, self.image, 512, 512)

        def load_threshold(self, flag=True):
            try:
                if flag:
                    low = float(self.text_edit_low.text())
                    high = float(self.text_edit_high.text())
                    if low > high:
                        raise Exception("低阈值不能大于高阈值")
                    if low > 0.999 or low < 0.001 or high > 0.999 or high < 0.001:
                        raise Exception("请输入正确阈值(0.001-0.999)")
                else:
                    low = float(self.text_edit_low.text())
                    if low > 0.999 or low < 0.001:
                        raise Exception("请输入正确阈值(0.001-0.999)")
                    self.low = low
                    return True
            except Exception as e:
                QMessageBox.warning(self, "错误", str(e))
                return False
            self.low = low
            self.high = high
            return True

        def load_threshold_1(self):
            try:
                threshold = float(self.text_edit.text())
                if threshold > 0.999 or threshold < 0.001:
                    raise Exception("请输入正确阈值(0.001-0.999)")
            except Exception as e:
                QMessageBox.warning(self, "错误", str(e))
                return False
            self.threshold = threshold
            return True

        def edge_detection_cv(self):
            image_array = ImageOperator.qimage_to_array(self.image)
            image_array = cv2.GaussianBlur(image_array, (5, 5), 0)
            edge_array = cv2.Canny(image_array, 100, 200)
            image = ImageOperator.array_to_qimage(edge_array)
            ImageOperator.set_image(self.image_label, image, 512, 512)

        def edge_detection_0(self):
            self.edge_detection()

        def edge_detection(self, flag=True):
            image_array = ImageOperator.qimage_to_array(self.image)
            image_array = cv2.GaussianBlur(image_array, (5, 5), 0)
            result = image_array
            if self.combo_box.currentIndex() == 0:
                image_array_1 = ImageOperator.Filter.universal_convolve(image_array,
                                                                        np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]),
                                                                        flag=False)
                image_array_2 = ImageOperator.Filter.universal_convolve(image_array,
                                                                        np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]),
                                                                        flag=False)
                result = np.sqrt(np.square(image_array_1) + np.square(image_array_2))
            elif self.combo_box.currentIndex() == 1:
                image_array_1 = ImageOperator.Filter.universal_convolve(image_array,
                                                                        np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]),
                                                                        flag=False)
                image_array_2 = ImageOperator.Filter.universal_convolve(image_array,
                                                                        np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]),
                                                                        flag=False)
                result = np.sqrt(np.square(image_array_1) + np.square(image_array_2))
            elif self.combo_box.currentIndex() == 2:
                result = ImageOperator.Filter.universal_convolve(image_array,
                                                                 np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]]))
            elif self.combo_box.currentIndex() == 3:
                result = ImageOperator.Filter.universal_convolve(image_array, np.array(
                    [[0, 0, -1, 0, 0], [0, -1, -2, -1, 0], [-1, -2, 16, -2, -1], [0, -1, -2, -1, 0], [0, 0, -1, 0, 0]]))
            # image_array = ImageOperator.ImageProcessorWidget.normalize(image_array)

            if not flag:
                return result

            image = ImageOperator.array_to_qimage(result)
            ImageOperator.set_image(self.image_label, image, 512, 512)

        @staticmethod
        def gradients(image_array):
            W, H = image_array.shape
            image_array = image_array.astype(np.float64)
            dx = np.zeros([W - 1, H - 1])
            dy = np.zeros([W - 1, H - 1])
            M = np.zeros([W - 1, H - 1])
            theta = np.zeros([W - 1, H - 1])

            for i in range(W - 1):
                for j in range(H - 1):
                    dx[i, j] = image_array[i + 1, j] - image_array[i, j]
                    dy[i, j] = image_array[i, j + 1] - image_array[i, j]
                    # 图像梯度幅值作为图像强度值
                    M[i, j] = np.sqrt(np.square(dx[i, j]) + np.square(dy[i, j]))
                    # 计算  θ - artan(dx/dy)
                    theta[i, j] = np.arctan(dx[i, j] / (dy[i, j] + 0.000000001)) * 180 / np.pi

            return dx, dy, M, theta

        @staticmethod
        def NMS(M, dx, dy):
            d = np.copy(M)
            W, H = M.shape
            NMS = np.copy(d)
            NMS[0, :] = NMS[W - 1, :] = NMS[:, 0] = NMS[:, H - 1] = 0
            for i in range(1, W - 1):
                for j in range(1, H - 1):
                    # 如果当前梯度为0，该点就不是边缘点
                    if M[i, j] == 0:
                        NMS[i, j] = 0
                    else:
                        gradX = dx[i, j]  # 当前点 x 方向导数
                        gradY = dy[i, j]  # 当前点 y 方向导数
                        gradTemp = d[i, j]  # 当前梯度点
                        # 如果 y 方向梯度值比较大，说明导数方向趋向于 y 分量
                        if np.abs(gradY) > np.abs(gradX):
                            weight = np.abs(gradX) / np.abs(gradY)  # 权重
                            grad2 = d[i, j - 1]
                            grad4 = d[i, j + 1]
                            if gradX * gradY > 0:
                                grad1 = d[i - 1, j - 1]
                                grad3 = d[i + 1, j + 1]
                            else:
                                grad1 = d[i + 1, j - 1]
                                grad3 = d[i - 1, j + 1]
                        # 如果 x 方向梯度值比较大
                        else:
                            weight = np.abs(gradY) / np.abs(gradX)
                            grad2 = d[i - 1, j]
                            grad4 = d[i + 1, j]
                            if gradX * gradY > 0:
                                grad1 = d[i - 1, j - 1]
                                grad3 = d[i + 1, j + 1]
                            else:
                                grad1 = d[i - 1, j + 1]
                                grad3 = d[i + 1, j - 1]
                        # 利用 grad1-grad4 对梯度进行插值
                        gradTemp1 = weight * grad1 + (1 - weight) * grad2
                        gradTemp2 = weight * grad3 + (1 - weight) * grad4
                        # 当前像素的梯度是局部的最大值，可能是边缘点
                        if gradTemp >= gradTemp1 and gradTemp >= gradTemp2:
                            NMS[i, j] = gradTemp
                        else:
                            # 不可能是边缘点
                            NMS[i, j] = 0
            return NMS

        @staticmethod
        def double_threshold(NMS, low, high):
            W, H = NMS.shape
            DT = np.zeros([W, H])
            # 定义高低阈值
            TL = low * np.max(NMS)
            TH = high * np.max(NMS)
            for i in range(1, W - 1):
                for j in range(1, H - 1):
                    # 双阈值选取
                    if NMS[i, j] < TL:
                        DT[i, j] = 0
                    elif NMS[i, j] > TH:
                        DT[i, j] = 1
                    # 连接
                    elif (NMS[i - 1, j - 1:j + 1] < TH).any() or (NMS[i + 1, j - 1:j + 1] < TH).any() or (
                            NMS[i, [j - 1, j + 1]] < TH).any():
                        DT[i, j] = 1
            return DT

        @staticmethod
        def canny(image_array, low, high):
            image_array = cv2.GaussianBlur(image_array, (5, 5), 0)
            dx, dy, M, theta = ImageOperator.EdgeDetection.gradients(image_array)
            NMS = ImageOperator.EdgeDetection.NMS(M, dx, dy)
            DT = ImageOperator.EdgeDetection.double_threshold(NMS, low, high)
            return DT * 255

        def my_canny(self):
            if not self.load_threshold():
                return
            image_array = ImageOperator.qimage_to_array(self.image)
            DT = self.canny(image_array, self.low, self.high)
            image = ImageOperator.array_to_qimage(DT)
            ImageOperator.set_image(self.image_label, image, image.width(), image.height())

        def hough(self):
            image_array = ImageOperator.qimage_to_array(self.image)
            height, width = image_array.shape

            # 创建一张黑色背景的Pixmap
            pixmap = QPixmap(width, height)
            pixmap.fill(Qt.GlobalColor.black)

            painter = QPainter(pixmap)
            painter.setPen(QColor(Qt.GlobalColor.white))  # 设置绘制直线的颜色，这里使用白色

            # 使用霍夫变换检测直线
            self.load_threshold()
            edges = self.canny(image_array, self.low, self.high).astype(np.uint8)
            # edges = cv2.Canny(image_array, 100, 200)
            # lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)  # 调整阈值根据需要

            if not self.load_threshold_1():
                return
            lines = self.hough_custom(edges,self.threshold)

            if lines is not None:
                for line in lines:
                    rho, theta = line
                    a = np.cos(theta)
                    b = np.sin(theta)
                    x0 = a * rho
                    y0 = b * rho
                    x1 = int(x0 + 1000 * (-b))
                    y1 = int(y0 + 1000 * a)
                    x2 = int(x0 - 1000 * (-b))
                    y2 = int(y0 - 1000 * a)
                    # 绘制检测到的直线
                    painter.drawLine(QPointF(x1, y1), QPointF(x2, y2))

            # circles = self.hough_circle(edges,0.99)
            # for circle in circles:
            #     x = int(circle[0])
            #     y = int(circle[1])
            #     r = int(circle[2])
            #     painter.drawEllipse(QPointF(x, y), r, r)

            circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, 100, param1=100, param2=30)
            if circles is not None:
                for circle in circles[0]:
                    x = int(circle[0])
                    y = int(circle[1])
                    r = int(circle[2])
                    painter.drawEllipse(QPointF(x, y), r, r)

            painter.end()

            # 更新图像控件
            image = pixmap.toImage()
            ImageOperator.set_image(self.image_label, image, image.width(), image.height())

        @staticmethod
        def hough_custom(edges,threshold=0.35):
            # 生成直线参数空间
            thetas = np.deg2rad(np.arange(0, 180, step=1))
            max_distance = int(np.sqrt(edges.shape[0] ** 2 + edges.shape[1] ** 2))+1
            distances = range(-max_distance, max_distance+1)
            votes = np.zeros((len(distances), len(thetas)))
            # 遍历所有边缘点
            for i in range(edges.shape[0]):
                for j in range(edges.shape[1]):
                    if edges[i, j] == 255:
                        # 遍历所有直线参数
                        for l in range(len(thetas)):
                            # 计算距离
                            distance=int(np.round(j * np.cos(thetas[l]) + i * np.sin(thetas[l])))
                            if -max_distance <= distance <= max_distance:
                                # 进行投票
                                votes[distance + max_distance, l] += 1
            max_value=np.max(votes)
            # 找到所有大于阈值的直线
            lines = []
            for i in range(votes.shape[0]):
                for j in range(votes.shape[1]):
                    if votes[i, j] >= max_value*threshold:
                        lines.append([distances[i], thetas[j]])
            return lines

        @staticmethod
        def hough_circle(edges, threshold=0.8):
            r_max=int(np.sqrt(edges.shape[0] ** 2 + edges.shape[1] ** 2))+1
            if r_max<10:
                return None
            r_min=10
            x_max=edges.shape[0]
            y_max=edges.shape[1]
            votes=np.zeros((x_max,y_max,r_max-r_min+1))
            for i in range(x_max):
                for j in range(y_max):
                    if edges[i,j]==255:
                        for x in range(x_max):
                            for y in range(y_max):
                                r=int(np.round(np.sqrt((x-i)**2+(y-j)**2)))
                                if r_min<=r<=r_max:
                                    votes[x,y,r-r_min]+=1

            max_value=np.max(votes)
            circles=[]
            for i in range(votes.shape[0]):
                for j in range(votes.shape[1]):
                    for k in range(votes.shape[2]):
                        if votes[i,j,k]>=max_value*threshold:
                            circles.append([i,j,k+r_min])
            return circles

        def edge_track(self):
            if not self.load_threshold(False):
                return
            image_array = ImageOperator.qimage_to_array(self.image)
            edge_image = ImageOperator.EdgeDetection.edge_tracking(image_array, self.low)
            image = ImageOperator.array_to_qimage(edge_image)
            ImageOperator.set_image(self.image_label, image, 512, 512)

        @staticmethod
        def edge_tracking(image_array, threshold=0.2):
            image_array = cv2.GaussianBlur(image_array, (5, 5), 0)
            dx, dy, M, _ = ImageOperator.EdgeDetection.gradients(image_array)
            W, H = M.shape
            # M=ImageOperator.EdgeDetection.NMS(M,dx,dy)
            visited = np.zeros((W, H), dtype=bool)
            edge_image = np.zeros((W, H), dtype=np.uint8)

            threshold = threshold * np.max(M)

            # 找到最大梯度点
            cur_index = np.unravel_index(np.argmax(M), M.shape)

            direct = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]
            cur_direct = 2

            def valid_coord(index):
                return 0 <= index[0] < W - 1 and 0 <= index[1] < H - 1

            while True:
                cur_direct = (cur_direct - 2) % 8
                visited[cur_index] = True
                edge_image[cur_index] = 255  # 标记为强边缘
                num = 0
                max_d = 0
                max_index = None
                max_direct = None
                while True:
                    cur_direct = (cur_direct + 1) % 8
                    new_index = (cur_index[0] + direct[cur_direct][0], cur_index[1] + direct[cur_direct][1])
                    if num > 4:
                        if max_index is not None:
                            cur_index = max_index
                            cur_direct = max_direct
                            break
                        else:
                            return edge_image
                    if valid_coord(new_index) and M[new_index] > max_d and not visited[new_index] and M[
                        new_index] > threshold:
                        max_d = M[new_index]
                        max_index = new_index
                        max_direct = cur_direct
                    num += 1

    class ImageCompress:
        @staticmethod
        def huffman(image_info):
            image_array=ImageOperator.qimage_to_array(image_info[0])
            hist,_=np.histogram(image_array,bins=256,range=(0,255))
            hist=hist/np.sum(hist)
            huffman_codes=ImageOperator.ImageCompress.build_huffman_tree(hist)
            compressed_data = ''
            for i in range(image_array.shape[0]):
                for j in range(image_array.shape[1]):
                    compressed_data+=huffman_codes[image_array[i,j]]
            binary_data=ImageOperator.ImageCompress.binary_string_to_bytes(compressed_data)
            info={'height':image_array.shape[0],'width':image_array.shape[1],'huffman_codes':huffman_codes,'binary_data':binary_data,'length':len(compressed_data)}
            FileManager.save_information(image_info,info,'huffman')

        @staticmethod
        def build_huffman_tree(histogram):
            heap = [[weight, [pixel, ""]] for pixel, weight in enumerate(histogram) if weight > 0]
            heapq.heapify(heap)

            while len(heap) > 1:
                lo = heapq.heappop(heap)
                hi = heapq.heappop(heap)
                for pair in lo[1:]:
                    pair[1] = '0' + pair[1]
                for pair in hi[1:]:
                    pair[1] = '1' + pair[1]
                heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])

            huffman_tree = heap[0]
            huffman_codes = dict(huffman_tree[1:])
            return huffman_codes

        @staticmethod
        def huffman_decompress(dict):
            width=dict['width']
            height=dict['height']
            huffman_codes=dict['huffman_codes']
            binary_data=dict['binary_data']
            length=dict['length']
            compressed_data = ImageOperator.ImageCompress.bytes_to_binary_string(binary_data,length)
            huffman_codes_reverse = {v: k for k, v in huffman_codes.items()}
            data = ''
            pixel_list = []
            for bit in compressed_data:
                data += bit
                if data in huffman_codes_reverse:
                    pixel_list.append(huffman_codes_reverse[data])
                    data = ''
            image_array=np.array(pixel_list).reshape(height,width)
            return image_array

        @staticmethod
        def binary_string_to_bytes(binary_string):
            # 确保二进制字符串长度是 8 的倍数
            binary_string = binary_string.ljust((len(binary_string) + 7) // 8 * 8, '0')
            # 将二进制字符串转换为整数
            num = int(binary_string, 2)
            # 计算需要多少字节来存储这个整数
            num_bytes = (num.bit_length() + 7) // 8
            # 使用 to_bytes 方法将整数转换为 bytes 对象
            byte_data = num.to_bytes(num_bytes, byteorder='big')
            return byte_data

        @staticmethod
        def bytes_to_binary_string(byte_data, length):
            # 将字节数据转换为二进制字符串
            binary_string = ''.join(format(byte, '08b') for byte in byte_data)
            # 截取所需长度的二进制字符串
            binary_string = binary_string[:length]
            return binary_string

        @staticmethod
        def rlc_compress(image_info):
            pair_list=[]
            image_array=ImageOperator.qimage_to_array(image_info[0])
            cur=image_array[0,0]
            num=0
            for i in range(image_array.shape[0]):
                for j in range(image_array.shape[1]):
                    if image_array[i,j]==cur:
                        num+=1
                        if num==255:
                            pair_list.append([cur,np.uint8(num)])
                            num=0
                    else:
                        pair_list.append([cur,np.uint8(num)])
                        cur=image_array[i,j]
                        num=1
            pair_list.append([cur, num])
            pair_list = np.array(pair_list, dtype=np.uint8)
            info={'height':image_array.shape[0],'width':image_array.shape[1],'pair_list':pair_list}
            FileManager.save_information(image_info,info,'rlc')

        @staticmethod
        def rlc_decompress(dict):
            width=dict['width']
            height=dict['height']
            pair_list=dict['pair_list']
            image_array=np.zeros((height,width),dtype=np.uint8)
            index=0
            for pair in pair_list:
                for i in range(pair[1]):
                    image_array[index//width,index%width]=pair[0]
                    index+=1
            return image_array

        @staticmethod
        def huffman_rlc_compress(image_info):
            image_array = ImageOperator.qimage_to_array(image_info[0])
            hist, _ = np.histogram(image_array, bins=256, range=(0, 255))
            hist = hist / np.sum(hist)
            huffman_codes = ImageOperator.ImageCompress.build_huffman_tree(hist)
            compressed_data = ''
            for i in range(image_array.shape[0]):
                for j in range(image_array.shape[1]):
                    compressed_data += huffman_codes[image_array[i, j]]
            pair_list = []
            cur = compressed_data[0]
            num = 0
            for i in range(len(compressed_data)):
                if compressed_data[i] == cur:
                    num += 1
                    if num == 255:
                        pair_list.append([1 if cur == '1' else 0, np.uint8(num)])
                        num = 0
                else:
                    pair_list.append([1 if cur == '1' else 0, np.uint8(num)])
                    cur = compressed_data[i]
                    num = 1
            pair_list.append([1 if cur == '1' else 0, num])
            bool_list = []
            num_list = []
            for pair in pair_list:
                bool_list.append(pair[0])
                num_list.append(pair[1])
            # 将bool_list转换为bytes
            # 进行补全
            if len(bool_list) % 8 != 0:
                bool_list.extend([0] * (8 - len(bool_list) % 8))
            bool_list = np.array(bool_list, dtype=np.uint8)
            bool_list = bool_list.reshape(-1, 8)
            bool_list = np.packbits(bool_list, axis=1)
            # 将num_list转换为bytes
            num_list = np.array(num_list, dtype=np.uint8)
            info = {'height': image_array.shape[0], 'width': image_array.shape[1], 'huffman_codes': huffman_codes,
                    'bool_list': bool_list, 'num_list': num_list}
            FileManager.save_information(image_info, info, 'huffman_rlc')

        @staticmethod
        def huffman_rlc_decompress(dict):
            height = dict['height']
            width = dict['width']
            huffman_codes = dict['huffman_codes']
            bool_list = dict['bool_list']
            num_list = dict['num_list']

            # Reverse the conversion from bytes to original formats
            bool_list = np.unpackbits(bool_list).flatten()
            num_list = num_list.tolist()

            # Perform Run-Length Decoding (RLD) to recover compressed data
            compressed_data = ''
            for i in range(len(num_list)):
                for j in range(num_list[i]):
                    compressed_data += '1' if bool_list[i] == 1 else '0'
            huffman_codes_reverse = {v: k for k, v in huffman_codes.items()}
            data = ''
            pixel_list = []
            for bit in compressed_data:
                data += bit
                if data in huffman_codes_reverse:
                    pixel_list.append(huffman_codes_reverse[data])
                    data = ''
            image_array = np.array(pixel_list).reshape(height, width)
            return image_array

        @staticmethod
        def jpeg_compress(image_info,quality=95):
            # 使用opencv库
            image_array = ImageOperator.qimage_to_array(image_info[0])
            params = [cv2.IMWRITE_JPEG_QUALITY, quality]
            compressed_img = cv2.imencode('.jpg', image_array, params)[1]
            filepath=FileManager.save_information(image_info,compressed_img,'jpeg',flag=False)
            with open(filepath,'wb') as f:
                f.write(compressed_img)


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
            image = QImage(file_path, "RGB32")
            image = image.convertToFormat(QImage.Format.Format_RGB32)
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
        file_dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptSave)
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
        return image

    @staticmethod
    def save_information(image_info,info,type, parent=None,flag=True):
        file_dialog = QFileDialog(parent)
        file_dialog.setWindowTitle("保存图像")
        file_dialog.setFileMode(QFileDialog.FileMode.AnyFile)
        file_dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptSave)
        file_dialog.setNameFilter("信息文件 (*."+type+")")

        # 删除后缀
        file_name=image_info[1].split('/')[-1]
        file_name=file_name.split('.')[0]

        file_dialog.selectFile(file_name + '.'+type)

        if file_dialog.exec() == QFileDialog.DialogCode.Accepted:
            file_path = file_dialog.selectedFiles()[0]
            if flag:
                with open(file_path,'wb') as f:
                    pickle.dump(info,f)
            else:
                return file_path

    @staticmethod
    def load_information(type,parent=None):
        file_dialog = QFileDialog(parent)
        file_dialog.setWindowTitle("打开图像")
        file_dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
        file_dialog.setNameFilter("信息文件 (*."+type+")")
        if file_dialog.exec() == QFileDialog.DialogCode.Accepted:
            file_path = file_dialog.selectedFiles()[0]
            with open(file_path,'rb') as f:
                try:
                    info=pickle.load(f)
                    return info
                except:
                    raise Exception("文件格式错误")
        else:
            return None
