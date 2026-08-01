import sys

from PySide6.QtWidgets import QApplication

from ebdm_app.main_window import MainWindow


def main() -> int:
    """Launch the EBDM desktop application."""
    app = QApplication(sys.argv)
    app.setApplicationName("EBDM Desktop")

    window = MainWindow()
    window.show()

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
