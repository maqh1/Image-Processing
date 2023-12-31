from PyQt6 import QtGui, QtWidgets

from FileManager import ImageViewer


class MainWindow(QtWidgets.QMainWindow):
    _instance = None

    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setWindowIcon(QtGui.QIcon("icon/preview.ico"))
        self.setWindowTitle("图像处理")
        self.resize(800, 600)

        self.MenuBar = QtWidgets.QMenuBar(self)
        self.MenuBar.setObjectName("MenuBar")
        self.setMenuBar(self.MenuBar)  # 将菜单栏设置到主窗口

        self.MenuFile = self.MenuBar.addMenu("文件")

        self.ActionNew = self.MenuFile.addAction("新建")
        self.ActionOpen = self.MenuFile.addAction("打开")
        self.ActionSave = self.MenuFile.addAction("保存")
        self.ActionSaveAs = self.MenuFile.addAction("另存为")
        self.ActionClose = self.MenuFile.addAction("关闭")

        imageViewer = ImageViewer.get_instance(self)

        self.ActionNew.triggered.connect(imageViewer.new_image)
        self.ActionOpen.triggered.connect(imageViewer.open_image)
        self.ActionSave.triggered.connect(imageViewer.save_image)
        self.ActionSaveAs.triggered.connect(imageViewer.save_image_as)
        self.ActionClose.triggered.connect(imageViewer.close_tab)

        self.MenuOperation = self.MenuBar.addMenu("操作")

        self.Action_sampling_and_quantization = self.MenuOperation.addAction("采样和量化")
        self.Action_bit_plane_decomposition = self.MenuOperation.addAction("位平面分解")
        self.Action_bmp2txt = self.MenuOperation.addAction("BMP转TXT")
        self.Action_gray_histogram = self.MenuOperation.addAction("灰度直方图")
        self.Action_processor = self.MenuOperation.addAction("图像变换")
        self.Action_histogram_equalization = self.MenuOperation.addAction("直方图均衡化")
        self.Action_matrix_transform = self.MenuOperation.addAction("矩阵变换")
        self.Action_trans_scale_rotate = self.MenuOperation.addAction("平移、缩放、旋转")
        self.Action_fourier_transform = self.MenuOperation.addAction("傅里叶变换")
        self.Action_wavelet_transform = self.MenuOperation.addAction("小波变换")
        self.Action_image_enhancement = self.MenuOperation.addAction("图像增强")
        self.Action_rgb_process = self.MenuOperation.addAction("彩色图像处理")
        self.Action_edge_detection = self.MenuOperation.addAction("边缘检测")

        self.Action_sampling_and_quantization.triggered.connect(imageViewer.sampling_and_quantization)
        self.Action_bit_plane_decomposition.triggered.connect(imageViewer.bit_plane_decomposition)
        self.Action_bmp2txt.triggered.connect(imageViewer.bmp2txt)
        self.Action_gray_histogram.triggered.connect(imageViewer.gray_histogram)
        self.Action_processor.triggered.connect(imageViewer.image_processor)
        self.Action_histogram_equalization.triggered.connect(imageViewer.histogram_equalization)
        self.Action_matrix_transform.triggered.connect(imageViewer.matrix_transform)
        self.Action_trans_scale_rotate.triggered.connect(imageViewer.trans_scale_rotate)
        self.Action_fourier_transform.triggered.connect(imageViewer.fourier_transform)
        self.Action_wavelet_transform.triggered.connect(imageViewer.wavelet_transform)
        self.Action_image_enhancement.triggered.connect(imageViewer.image_enhancement)
        self.Action_rgb_process.triggered.connect(imageViewer.rgb_process)
        self.Action_edge_detection.triggered.connect(imageViewer.edge_detection)

        self.MenuImages = self.MenuBar.addMenu("多图像操作")
        self.Action_Object_Detection = self.MenuImages.addAction("对象检测")
        self.Action_Speed_Calculation = self.MenuImages.addAction("速度计算")

        self.Action_Object_Detection.triggered.connect(imageViewer.object_detection)
        self.Action_Speed_Calculation.triggered.connect(imageViewer.speed_calculation)

        self.MenuImages = self.MenuBar.addMenu("压缩和解压缩")
        self.Action_Huffman_Compress = self.MenuImages.addAction("哈夫曼压缩")
        self.Action_Huffman_Decompress = self.MenuImages.addAction("哈夫曼解压缩")
        self.Action_RLC_Compress = self.MenuImages.addAction("RLC压缩")
        self.Action_RLC_Decompress= self.MenuImages.addAction("RLC解压缩")
        self.Action_Huffman_RLC_Compress = self.MenuImages.addAction("哈夫曼+RLC压缩")
        self.Action_Huffman_RLC_Decompress = self.MenuImages.addAction("哈夫曼+RLC解压缩")
        self.Action_JPEG_Compress = self.MenuImages.addAction("JPEG压缩")

        self.Action_Huffman_Compress.triggered.connect(imageViewer.huffman_compress)
        self.Action_Huffman_Decompress.triggered.connect(imageViewer.huffman_decompress)
        self.Action_RLC_Compress.triggered.connect(imageViewer.rlc_compress)
        self.Action_RLC_Decompress.triggered.connect(imageViewer.rlc_decompress)
        self.Action_Huffman_RLC_Compress.triggered.connect(imageViewer.huffman_rlc_compress)
        self.Action_Huffman_RLC_Decompress.triggered.connect(imageViewer.huffman_rlc_decompress)
        self.Action_JPEG_Compress.triggered.connect(imageViewer.jpeg_compress)

        self.setCentralWidget(imageViewer)

    @staticmethod
    def get_instance(parent=None):
        if MainWindow._instance is None:
            MainWindow._instance = MainWindow(parent)
        return MainWindow._instance
