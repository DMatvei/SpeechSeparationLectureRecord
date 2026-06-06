import sys
import os


from PyQt6 import uic
from PyQt6.QtCore import Qt, QUrl, QTime
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
from PyQt6.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox, QDialog

from pipeline import process

QUALITY_PRESETS = {
    "low" : {},
    "medium" : {},
    "high" : {}
}
AUDIO_FILTER = "Audio Files (*.wav *.mp3 *.flac *.mp4);;All Files (*)"
AUDIO_EXTS = (".wav", ".mp3", ".flac", ".mp4", ".m4a", ".ogg")


class RefDialog(QDialog):
    """
    для вызова
    dlg = RefDialog(audio_path, start_ms, end_ms, parent=self)
if dlg.exec() == QDialog.DialogCode.Accepted:
    ...  # фрагмент подтверждён
else:
    ...  # перевыбрать другой


    """

    def __init__(self, audio_path, start_ms, end_ms, parent = None):
        super().__init__(parent)
        uic.loadUi("ref_dialog.ui", self)

        self.start_ms = start_ms
        self.end_ms = end_ms

        self.timecodeLabel.setText(
            f'Фрагмент: {self._fmt(start_ms)} - {self._fmt(end_ms)}'
        )

        # плеер -----------------
        self.player = QMediaPlayer(self)
        self.audio_out = QAudioOutput(self)
        self.player.setAudioOutput(self.audio_out)
        self.player.setSource(QUrl.fromLocalFile(os.path.abspath(audio_path)))

        self.playSlider.setRange(start_ms, end_ms)
        self.playSlider.setValue(start_ms)

        # Сигналы ---------------------------------------
        self.pushButton_play.clicked.connect(self.toggle_play)
        self.pushButton_confirm.clicked.connect(self.accept)
        self.pushButton_reroll.clicked.connect(self.reject)
        self.player.positionChanged.connect(self.on_position)
        self.playSlider.sliderMoved.connect(self.player.setPosition)


    @staticmethod
    def _fmt(ms):
        t = QTime(0, 0).addMSecs(int(ms))
        return t.toString("HH:mm:ss")

    def toggle_play(self):
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.player.pause()
            self.pushButton_play.setText("▶")
        else:
            if self.player.position() < self.start_ms or self.player.position() >= self.end_ms:
                self.player.setPosition(self.start_ms)
            self.player.play()
            self.pushButton_play.setText("⏸")

    def on_position(self, pos):
        # останавливаемся на конце выбранного участка
        if pos >= self.end_ms:
            self.player.pause()
            self.player.setPosition(self.start_ms)
            self.pushButton_play.setText("▶ Слушать")
            self.playSlider.setValue(self.start_ms)
            return
        self.playSlider.setValue(pos)
        self.playTimeLabel.setText(self._fmt(pos - self.start_ms).lstrip("0:") or "0:00")

    def closeEvent(self, event):
        self.player.stop()
        super().closeEvent(event)






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

        # dnd
        self.setAcceptDrops(True)

    # dnd -----------------------------------------
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if urls and urls[0].toLocalFile().lower().endswith(AUDIO_EXTS):
                event.acceptProposedAction()
                self.dropZone.setStyleSheet(
                    "QLabel { border: 2px dashed #3b82f6; border-radius: 8px;"
                    " color: #3b82f6; background: #eff6ff; }"
                )
                return
        event.ignore()

    def dragLeaveEvent(self, event):
        self._reset_dropzone_style()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            if path.lower().endswith(AUDIO_EXTS):
                self.lineEdit.setText(path)
        self._reset_dropzone_style()

    def _reset_dropzone_style(self):
        self.dropZone.setStyleSheet(
            "QLabel { border: 2px dashed #b8b8b8; border-radius: 8px;"
            " color: #777; background: #fafafa; }"
        )

    #  Качество -------------------------------
    def selected_quality(self):
        if self.radioLow.isChecked():
            return QUALITY_PRESETS['low']
        if self.radioHigh.isChecked():
            return QUALITY_PRESETS['high']
        return QUALITY_PRESETS['medium']


    # Основные действия ------------------------
    def browse_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Выберите аудофайл", "",
            AUDIO_FILTER
        )
        if path:
            self.lineEdit.setText(path)


    def run_processing(self):
        input_path = self.lineEdit.text()
        if not input_path:
            QMessageBox.warning(self, "Ошибка", "Сначала выберите файл")
            return

        params = self.selected_quality()

        self.progressBar.setValue(0)
        self.statusLabel.setText("Обработка…")
        self.statusbar.showMessage("Обработка…")
        self.pushButton_start.setEnabled(False)

        try:
            process(
                input_path,
                self.output_dir,
                on_progress=lambda v: self.progressBar.setValue(v),
                **params,
            )
            self.statusLabel.setText("Готово")
            self.statusbar.showMessage("Готово!")
            self.pushButton_openFolder.setEnabled(True)
            QMessageBox.information(self, "Готово", "Обработка завершена")
        except Exception as e:
            self.statusLabel.setText("Ошибка")
            self.statusbar.showMessage("Ошибка")
            QMessageBox.critical(self, "Ошибка", str(e))
        finally:
            self.pushButton_start.setEnabled(True)

    def open_output_folder(self):
        os.makedirs(self.output_dir, exist_ok=True)
        os.startfile(self.output_dir) # Windows


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindom()
    window.show()
    sys.exit(app.exec())