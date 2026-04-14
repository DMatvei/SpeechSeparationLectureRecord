import sys
import os
from PyQt6 import uic
from PyQt6.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox


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
            "Audio Files (*.wav *.mp3 *.flac);; All Files (*)"
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

        # заглушка
        self.process_audio(input_path)


    def process_audio(self, input_path:str):
        """тут основная работа всего диплома должна быть"""
        #to-do привязать работу обратботки сюда

        # имитация
        for i in range(101):
            self.progressBar.setValue(i)
            QApplication.processEvents()

        os.makedirs(self.output_dir, exist_ok=True)
        self.statusbar.showMessage("Готово")
        QMessageBox.information(self, "Готово", "Обработка завершена")

    def open_output_folder(self):
        os.makedirs(self.output_dir, exist_ok=True)
        os.startfile(self.output_dir) # Windows


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindom()
    window.show()
    sys.exit(app.exec())