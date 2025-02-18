import sys
import numpy as np
import os,time
from dataclasses import dataclass
from typing import Optional,Union,Dict
from PyQt6.QtWidgets import (
    QApplication, QSystemTrayIcon, QMenu, QWidget, QVBoxLayout,
    QPushButton, QFileDialog, QMessageBox, QLabel, QTableWidget, QTableWidgetItem, QDialog, QTextEdit
)
from PyQt6.QtGui import (
    QGuiApplication, QPainter, QPen, QColor, QAction, QImage, QIcon,
    QCursor, QMovie, QPixmap, QBrush, QPolygon, QPainterPath
)
from PyQt6.QtCore import Qt, QRect, QPoint, QTimer, QThread, pyqtSignal,QSize
from rapidocr_onnxruntime import RapidOCR
from rapid_table import RapidTable
from pathlib import Path

# 输入
@dataclass
class RapidTableInput:
    model_type: Optional[str] = "slanet_plus"
    model_path: Union[str, Path, None, Dict[str, str]] = None
    use_cuda: bool = False
    device: str = "cpu"

# 输出
@dataclass
class RapidTableOutput:
    pred_html: Optional[str] = None
    cell_bboxes: Optional[np.ndarray] = None
    logic_points: Optional[np.ndarray] = None
    elapse: Optional[float] = None

class ScreenshotWidget(QWidget):
    closed = pyqtSignal()  # 添加关闭信号
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setWindowState(Qt.WindowState.WindowFullScreen)
        
        self.setCursor(Qt.CursorShape.CrossCursor)  # 设置鼠标为十字光标
        
        self.begin = QPoint()
        self.end = QPoint()
        self.screen = QGuiApplication.primaryScreen().grabWindow(QApplication.primaryScreen().geometry().x())
        self.initUI()

    def initUI(self):
        self.show()
    
    def paintEvent(self, event):

        painter = QPainter(self)
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Clear)
        painter.fillRect(self.rect(), Qt.GlobalColor.transparent)
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)

        painter.drawPixmap(0, 0, self.screen)

        # 确保选区方向正确
        rect = QRect(self.begin, self.end).normalized()

        # 设置画笔颜色和线宽（红色边框）
        painter.setPen(QPen(QColor(255, 0, 0), 2))
        painter.drawRect(rect)

        # 绘制半透明背景
        painter.setBrush(QColor(255, 0, 0, 0))  # 50 代表半透明
        painter.drawRect(rect)

        # 获取缩放因子
        scale_factor = QGuiApplication.primaryScreen().devicePixelRatio()

        # 设置字体样式
        font = painter.font()
        font.setPointSize(int(font.pointSize() * scale_factor))  # 调整字体大小
        font.setBold(True)
        painter.setFont(font)

        # 计算文本内容（矩形大小）
        text = f"{rect.width()}x{rect.height()}"

        # 计算文本背景框
        text_rect = QRect(rect.topRight() + QPoint(5, -20), QSize(60, 20))

        # 画半透明文本背景
        painter.setBrush(QColor(0, 0, 0, 150))  # 黑色半透明背景
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRect(text_rect)

        # 画文本
        painter.setPen(Qt.GlobalColor.white)
        painter.drawText(text_rect, Qt.AlignmentFlag.AlignCenter, text)

        # 添加屏幕边界保护
        screen = QGuiApplication.screenAt(self.begin)
        if screen:
            screen_rect = screen.geometry()
            draw_rect = rect.intersected(screen_rect.translated(-self.pos()))
            painter.drawRect(draw_rect)

    def mousePressEvent(self, event):
        # 处理鼠标右键
        if event.button() == Qt.MouseButton.RightButton:
            self._cleanup_exit()
            event.accept()
            return

        # 新增：关闭现有选项窗口
        if hasattr(self, 'option_window') and self.option_window:
            self.option_window.close()
            self.option_window = None  # 重要！清除窗口引用

        self.begin = event.globalPosition().toPoint()  # 改为全局坐标
        self.end = self.begin
        self.update()

    def mouseMoveEvent(self, event):
        self.end = event.globalPosition().toPoint()  # 改为全局坐标
        self.update()

    def mouseReleaseEvent(self, event):
        # 仅更新显示，不隐藏窗口
        self.end = event.globalPosition().toPoint()  # 改为全局坐标
        self.update()  # 触发重绘
        self.processScreenshot()

    def _cleanup_exit(self):
        """统一的退出清理方法"""
        # 关闭所有子窗口
        if hasattr(self, 'option_window') and self.option_window:
            self.option_window.close()
        
        # 隐藏主窗口
        self.hide()
        
        # 发送关闭信号（需要先在类定义中添加 closed 信号）
        self.closed.emit()
        
        # 延迟销毁对象
        QTimer.singleShot(100, self.deleteLater)

    def closeEvent(self, event):
        self._cleanup_exit()
        super().closeEvent(event)

    def processScreenshot(self):
        # 临时隐藏窗口（关键修改）
        self.setVisible(False)
        QApplication.processEvents()  # 强制刷新UI
        
        try:
            # 获取屏幕对象
            screen = QGuiApplication.screenAt(self.begin)
            if not screen:
                screen = QGuiApplication.primaryScreen()

            # 截取完整屏幕（此时窗口已隐藏）
            full_screenshot = screen.grabWindow(0)
            
            # 计算实际物理坐标
            screen_geo = screen.geometry()
            local_begin = self.begin - screen_geo.topLeft()
            local_end = self.end - screen_geo.topLeft()
            
            scale_factor = screen.devicePixelRatio()
            x1 = int(min(local_begin.x(), local_end.x()) * scale_factor)
            y1 = int(min(local_begin.y(), local_end.y()) * scale_factor)
            width = int(abs(local_begin.x() - local_end.x()) * scale_factor)
            height = int(abs(local_begin.y() - local_end.y()) * scale_factor)

            # 裁剪图像
            self.cropped_image = full_screenshot.toImage().copy(x1, y1, width, height)
        finally:
            # 立即恢复窗口显示
            self.setVisible(True)
            QApplication.processEvents()

        self.showOptions()

    def showOptions(self):
        # 创建窗口前检查旧实例
        if hasattr(self, 'option_window') and self.option_window:
            self.option_window.close()
        # 保持主窗口可见且可绘制
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint 
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.show()

        # 创建选项窗口时设置父对象为截图窗口
        self.option_window = QWidget(self)  # 注意这里的父对象设置
        self.option_window.setWindowFlags(
            Qt.WindowType.FramelessWindowHint 
            | Qt.WindowType.Tool 
            | Qt.WindowType.WindowStaysOnTopHint
        )

        layout = QVBoxLayout()
        
        btn_save = QPushButton("保存截图")
        btn_save.clicked.connect(self.saveScreenshot)
        
        btn_retry = QPushButton("重新截取")
        btn_retry.clicked.connect(self.retryScreenshot)
        
        btn_ocr = QPushButton("文字识别")
        btn_ocr.clicked.connect(self.tableRecognition)

        btn_save_and_recognize = QPushButton("表格识别") 
        btn_save_and_recognize.clicked.connect(self.saveAndRecognize)

        layout.addWidget(btn_save)
        layout.addWidget(btn_retry)
        layout.addWidget(btn_ocr)
        layout.addWidget(btn_save_and_recognize) 

        self.option_window.setLayout(layout)

        # 让选项窗口出现在选区右下角
        self.option_window.move(self.end + QPoint(20, 20))

        self.option_window.show()
        
        # 设置窗口样式
        self.option_window.setStyleSheet("""
            QWidget {
                background-color: rgba(50, 50, 50, 200);
                border-radius: 10px;
                padding: 5px;
            }
            QPushButton {
                color: white;
                background-color: rgba(0, 0, 0, 100);
                border: 1px solid gray;
                border-radius: 5px;
                padding: 5px;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: rgba(100, 100, 100, 150);
            }
        """)
        
        # 确保窗口不会超出屏幕范围
        screen = QGuiApplication.screenAt(self.end)
        if screen:
            screen_geo = screen.availableGeometry()
            window_width = self.option_window.sizeHint().width()
            window_height = self.option_window.sizeHint().height()

            x = self.end.x() + 20
            y = self.end.y() + 20

            if x + window_width > screen_geo.right():
                x = screen_geo.right() - window_width - 10
            if y + window_height > screen_geo.bottom():
                y = screen_geo.bottom() - window_height - 10

            self.option_window.move(x, y)

    def saveScreenshot(self):
        path, _ = QFileDialog.getSaveFileName(
            None, "保存截图", "", "PNG Image (*.png);;JPEG Image (*.jpg)"
        )
        if path:
            self.cropped_image.save(path)
            self.option_window.close()
            self.destroy()

    def showRecognitionError(self, msg):
        self.loading_dialog.close()
        QMessageBox.critical(None, "识别错误", f"表格识别失败：{msg}")

    def retryScreenshot(self):
        self.option_window.close()
        self.begin = QPoint()  # 重置选区坐标
        self.end = QPoint()
        self.show()  # 重新显示主窗口

    def tableRecognition(self):
        try:
            self.option_window.close()
            self.destroy()
            
            # 转换为OpenCV格式
            img = self.cropped_image
            img = img.convertToFormat(QImage.Format.Format_RGBA8888)
            width = img.width()
            height = img.height()
            ptr = img.bits()
            ptr.setsize(img.sizeInBytes())
            arr = np.array(ptr).reshape(height, width, 4)
            
            # 显示加载动画
            self.loading_dialog = LoadingDialog()
            self.loading_dialog.show()
            
            # 创建并启动识别线程
            self.recognition_thread = OCRRecognitionThread(arr)
            self.recognition_thread.finished.connect(self.showRecognitionResult)
            self.recognition_thread.error.connect(self.showRecognitionError)
            self.recognition_thread.start()
            
        except Exception as e:
            QMessageBox.critical(None, "识别错误", f"表格处理发生错误：{str(e)}")
            

    def showRecognitionResult(self, result):
        self.loading_dialog.close()
        
        result_dialog = QDialog()
        result_dialog.setWindowTitle("识别结果")
        result_dialog.resize(600, 400)
        
        layout = QVBoxLayout()
        
        table = QTableWidget(len(result), 2)
        table.setHorizontalHeaderLabels(["位置", "内容"])
        table.horizontalHeader().setStretchLastSection(True)
        
        for i, line in enumerate(result):
            box, text, confidence = line
            table.setItem(i, 0, QTableWidgetItem(f"({box[0][0]:.0f}, {box[0][1]:.0f})"))
            table.setItem(i, 1, QTableWidgetItem(text))
        
        btn_copy = QPushButton("复制全部内容")
        btn_copy.clicked.connect(lambda: QApplication.clipboard().setText("\n".join([line[1] for line in result])))
        
        layout.addWidget(table)
        layout.addWidget(btn_copy)
        result_dialog.setLayout(layout)
        result_dialog.exec()

    def saveAndRecognize(self):
        self.option_window.close()
        self.destroy()

        # 确保 `tmp/` 目录存在
        save_dir = os.path.join(os.getcwd(), "tmp")
        os.makedirs(save_dir, exist_ok=True)

        # 生成带时间戳的文件名
        timestamp = time.strftime("%Y%m%d_%H%M%S")  # 生成 "YYYYMMDD_HHMMSS" 格式时间戳
        save_path = os.path.join(save_dir, f"screenshot_{timestamp}.png")

        # 保存截图
        self.cropped_image.save(save_path)

        # 显示加载对话框
        self.loading_dialog = LoadingDialog()
        self.loading_dialog.show()

        # 创建并启动识别线程
        self.recognition_thread = RecognitionThread(save_path)
        self.recognition_thread.finished.connect(self.showResultWindow)
        self.recognition_thread.error.connect(self.showRecognitionError)
        self.recognition_thread.start()

    def showResultWindow(self, result_text):
        # 关闭加载对话框
        self.loading_dialog.close()
        # 识别结果窗口
        result_window = QDialog()
        result_window.setWindowTitle("表格识别结果")
        result_window.setWindowFlags(Qt.WindowType.WindowStaysOnTopHint)

        layout = QVBoxLayout()

        # 可复制的文本框
        text_edit = QTextEdit()
        text_edit.setText(result_text)
        text_edit.setReadOnly(True)  # 只读模式
        layout.addWidget(text_edit)

        text_edit.setFixedHeight(400)  # 根据需要设置文本框高度
        text_edit.setFixedWidth(600)   # 根据需要设置文本框宽度

        # 关闭按钮
        btn_close = QPushButton("关闭")
        btn_close.clicked.connect(result_window.close)
        layout.addWidget(btn_close)

        result_window.setLayout(layout)
        result_window.exec()
        
    def keyPressEvent(self, event):
        # 处理 ESC 键
        if event.key() == Qt.Key.Key_Escape:
            self._cleanup_exit()
            event.accept()
        else:
            super().keyPressEvent(event)

class OCRRecognitionThread(QThread):
    finished = pyqtSignal(list)
    error = pyqtSignal(str)
    
    def __init__(self, image_array):
        super().__init__()
        self.image_array = image_array
    
    def run(self):
        try:
            ocr_engine = RapidOCR()
            result, _ = ocr_engine(self.image_array)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))
            
class RecognitionThread(QThread):
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, image_path):
        super().__init__()
        self.image_path = image_path

    def run(self):
        try:
            input_args = RapidTableInput(model_type="unitable")
            table_engine = RapidTable(input_args)
            table_results = table_engine(self.image_path)
            self.finished.emit(table_results.pred_html)
        except Exception as e:
            self.error.emit(str(e))

class LoadingDialog(QDialog):
    def __init__(self):
        super().__init__()

        # 设置无边框 & 透明
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.Tool)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        # 布局
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)  # 去掉内边距

        # 提示文本
        self.label = QLabel("正在识别中，请稍等...", self)
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setStyleSheet("color: red; font-size: 14px;")  # 文字颜色

        # 旋转加载动画
        self.loading_gif = QMovie("loading.gif")  # 确保 GIF 具有透明背景
        self.loading_label = QLabel(self)
        self.loading_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.loading_label.setMovie(self.loading_gif)
        self.loading_gif.start()

        # 添加到布局
        layout.addWidget(self.label)
        layout.addWidget(self.loading_label)

        # 自动匹配 GIF 大小
        self.loading_label.setFixedSize(self.loading_gif.frameRect().size())
        self.setFixedSize(self.loading_label.size())

        # 居中显示
        screen = QApplication.primaryScreen().geometry()
        self.move((screen.width() - self.width()) // 2, (screen.height() - self.height()) // 2)

class TrayIcon(QSystemTrayIcon):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setIcon(QGuiApplication.windowIcon())
        
        menu = QMenu()
        action_screenshot = QAction(QIcon(self.create_square_icon()), "截图", self)
        action_screenshot.triggered.connect(self.takeScreenshot)
        menu.addAction(action_screenshot)
        
        action_exit = QAction(QIcon(self.create_triangle_icon()), "退出", self)
        action_exit.triggered.connect(QApplication.instance().quit)
        menu.addAction(action_exit)
        
        self.setContextMenu(menu)

    def takeScreenshot(self):
        # 关闭所有已经存在的识别结果窗口
        for widget in QApplication.topLevelWidgets():
            if isinstance(widget, QDialog):  # 确保是QDialog类型的窗口
                title = widget.windowTitle()
                if title in ["表格识别结果", "识别结果"]:  # 根据窗口标题来判断
                    # 获取布局并删除所有控件
                    layout = widget.layout()
                    if layout:
                        for i in range(layout.count()):
                            child_widget = layout.itemAt(i).widget()
                            if child_widget:
                                child_widget.deleteLater()  # 删除子控件
                    widget.close()
                    widget.deleteLater()  # 销毁窗口本身

                    # 强制隐藏该窗口，确保窗口彻底消失
                    widget.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, False)  # 禁用透明绘制
                    widget.hide()  # 隐藏窗口

        # 创建新的截图窗口
        self.screenshot_widget = ScreenshotWidget()

    def create_square_icon(self):
        """创建带边框的方形图标"""
        pixmap = QPixmap(32, 32)
        pixmap.fill(Qt.GlobalColor.transparent)
        
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # 绘制蓝色边框方形
        painter.setPen(QPen(Qt.GlobalColor.blue, 2))
        painter.setBrush(QBrush(Qt.GlobalColor.transparent))
        painter.drawRect(4, 4, 24, 24)
        
        painter.end()
        return QIcon(pixmap)

    def create_triangle_icon(self):
        """创建红色填充三角形图标"""
        pixmap = QPixmap(32, 32)
        pixmap.fill(Qt.GlobalColor.transparent)
        
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # 绘制红色三角形
        path = QPainterPath()
        path.moveTo(16, 6)
        path.lineTo(26, 26)
        path.lineTo(6, 26)
        path.closeSubpath()
        
        painter.fillPath(path, Qt.GlobalColor.red)
        painter.end()
        
        return QIcon(pixmap)
    
if __name__ == "__main__":
    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
  
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)
    
    # 设置托盘图标
    tray = TrayIcon()
    tray.setIcon(QIcon("icon.png"))  # 需要准备一个图标文件
    tray.show()
    
    sys.exit(app.exec())