from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QButtonGroup,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ebdm_app.binary_binary_page import BinaryBinaryPage
from ebdm_app.continuous_continuous_page import ContinuousContinuousPage
from ebdm_app.binary_continuous_page import BinaryContinuousPage


class MainWindow(QMainWindow):
    """Main application window for the EBDM desktop prototype."""

    def __init__(self) -> None:
        super().__init__()

        self.setWindowTitle("EBDM Desktop")
        self.resize(1180, 760)
        self.setMinimumSize(960, 640)

        self._build_interface()
        self._apply_styles()

    def _build_interface(self) -> None:
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        root_layout = QHBoxLayout(central_widget)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        sidebar = self._create_sidebar()
        self.page_stack = self._create_page_stack()

        root_layout.addWidget(sidebar)
        root_layout.addWidget(self.page_stack, 1)

    def _create_sidebar(self) -> QFrame:
        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(250)

        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(18, 24, 18, 20)
        layout.setSpacing(8)

        app_title = QLabel("EBDM")
        app_title.setObjectName("appTitle")

        app_subtitle = QLabel("Joint distribution estimation")
        app_subtitle.setObjectName("appSubtitle")

        layout.addWidget(app_title)
        layout.addWidget(app_subtitle)
        layout.addSpacing(24)

        section_label = QLabel("METHODS")
        section_label.setObjectName("sectionLabel")
        layout.addWidget(section_label)
        layout.addSpacing(4)

        self.navigation_group = QButtonGroup(self)
        self.navigation_group.setExclusive(True)

        navigation_items = [
            ("Binary–Binary", 0),
            ("Continuous–Continuous", 1),
            ("Binary–Continuous", 2),
        ]

        for text, page_index in navigation_items:
            button = QPushButton(text)
            button.setObjectName("navButton")
            button.setCheckable(True)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.setSizePolicy(
                QSizePolicy.Policy.Expanding,
                QSizePolicy.Policy.Fixed,
            )
            button.clicked.connect(
                lambda checked=False, index=page_index:
                self.page_stack.setCurrentIndex(index)
            )

            self.navigation_group.addButton(button)
            layout.addWidget(button)

            if page_index == 0:
                button.setChecked(True)

        layout.addStretch()

        status_label = QLabel("Prototype · Version 0.1")
        status_label.setObjectName("statusLabel")
        layout.addWidget(status_label)

        return sidebar

    def _create_page_stack(self) -> QStackedWidget:
        stack = QStackedWidget()
        stack.setObjectName("pageStack")

        stack.addWidget(BinaryBinaryPage())

        stack.addWidget(ContinuousContinuousPage())

        stack.addWidget(BinaryContinuousPage())

        return stack

    def _create_placeholder_page(
        self,
        title: str,
        description: str,
    ) -> QWidget:
        page = QWidget()

        layout = QVBoxLayout(page)
        layout.setContentsMargins(46, 38, 46, 38)
        layout.setSpacing(20)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        title_label = QLabel(title)
        title_label.setObjectName("pageTitle")

        description_label = QLabel(description)
        description_label.setObjectName("pageDescription")
        description_label.setWordWrap(True)

        card = QFrame()
        card.setObjectName("contentCard")

        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(28, 26, 28, 26)

        placeholder = QLabel(
            "This method interface will be added after the "
            "Binary–Binary workflow is connected and tested."
        )
        placeholder.setObjectName("placeholder")
        placeholder.setWordWrap(True)
        placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        placeholder.setMinimumHeight(300)

        card_layout.addWidget(placeholder)

        layout.addWidget(title_label)
        layout.addWidget(description_label)
        layout.addWidget(card)

        return page

    def _apply_styles(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow {
                background: #ffffff;
            }

            QWidget {
                font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
                color: #202123;
            }

            QFrame#sidebar {
                background: #f7f7f8;
                border-right: 1px solid #e5e5e5;
            }

            QLabel#appTitle {
                font-size: 23px;
                font-weight: 650;
                color: #202123;
            }

            QLabel#appSubtitle {
                font-size: 12px;
                color: #7a7a7a;
            }

            QLabel#sectionLabel {
                font-size: 11px;
                font-weight: 600;
                color: #8a8a8a;
                letter-spacing: 1px;
            }

            QPushButton#navButton {
                min-height: 42px;
                padding: 0 14px;
                border: none;
                border-radius: 10px;
                background: transparent;
                text-align: left;
                font-size: 14px;
                color: #303030;
            }

            QPushButton#navButton:hover {
                background: #ececee;
            }

            QPushButton#navButton:checked {
                background: #e7e7e9;
                font-weight: 600;
            }

            QLabel#statusLabel {
                font-size: 11px;
                color: #999999;
            }

            QStackedWidget#pageStack {
                background: #ffffff;
            }

            QLabel#pageTitle {
                font-size: 29px;
                font-weight: 650;
                color: #202123;
            }

            QLabel#pageDescription {
                font-size: 15px;
                color: #666666;
            }

            QFrame#contentCard {
                background: #ffffff;
                border: 1px solid #e5e5e5;
                border-radius: 16px;
            }

            QLabel#placeholder {
                padding: 28px;
                border: 1px dashed #d6d6d6;
                border-radius: 12px;
                background: #fafafa;
                color: #8a8a8a;
                font-size: 13px;
            }
            """
        )
