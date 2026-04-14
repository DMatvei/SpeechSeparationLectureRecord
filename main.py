import sys
import os
import time

from PyQt6 import uic
from PyQt6.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox

from pipeline import process


class MainWindom(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi("main_window.ui", self)

        # привязка кнопок
        self.pushButton_view.clicked.connect(self.browse_file)
        self.pushButton_start.clicked.connect(self.run_processing)
        self.pushButton_openFolder.clicked.connect(self.open_output_folder)

        # прогресс бар
        self.progressBar.setValue(0)

        self.output_dir = os.path.join(os.getcwd(), "output")


    def browse_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Выберите аудофайл", "",
            "Audio Files (*.wav *.mp3 *.flac *.mp4);; All Files (*)"
        )
        if path:
            self.lineEdit.setText(path)


    def run_processing(self):
        input_path = self.lineEdit.text()
        if not input_path:
            QMessageBox.warning(self, "Ошибка", "Сначала выберите файл")
            return

        self.progressBar.setValue(0)
        self.statusbar.showMessage("Обработка...")

        try:
            process(
                input_path,
                self.output_dir,
                on_progress=lambda v: self.progressBar.setValue(v)
            )
            self.statusbar.showMessage("Готово!")
            QMessageBox.information(self, "Готово", "Обработка завершена")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", str(e))
            self.statusbar.showMessage("Ошибка")


    def open_output_folder(self):
        os.makedirs(self.output_dir, exist_ok=True)
        os.startfile(self.output_dir) # Windows


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindom()
    window.show()
    sys.exit(app.exec())