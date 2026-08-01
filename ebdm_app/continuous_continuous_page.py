from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QStyledItemDelegate,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ebdm_app.estimators.continuous_continuous import (
    estimate_continuous_continuous,
)


class CenteredItemDelegate(QStyledItemDelegate):
    """Center table values in both display and editing modes."""

    def initStyleOption(self, option, index) -> None:
        super().initStyleOption(option, index)
        option.displayAlignment = Qt.AlignmentFlag.AlignCenter

    def createEditor(self, parent, option, index):
        editor = super().createEditor(parent, option, index)

        if hasattr(editor, "setAlignment"):
            editor.setAlignment(Qt.AlignmentFlag.AlignCenter)

        return editor


class ContinuousContinuousPage(QWidget):
    """Interface for continuous–continuous estimation."""

    def __init__(self) -> None:
        super().__init__()
        self._build_interface()
        self._apply_styles()
        self._load_example()
        self._update_method_settings()

    def _build_interface(self) -> None:
        page_layout = QVBoxLayout(self)
        page_layout.setContentsMargins(0, 0, 0, 0)
        page_layout.setSpacing(0)

        scroll_area = QScrollArea()
        scroll_area.setObjectName("ccScrollArea")
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        scroll_area.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )

        content_widget = QWidget()
        content_widget.setObjectName("ccScrollContent")

        outer_layout = QVBoxLayout(content_widget)
        outer_layout.setContentsMargins(46, 38, 46, 38)
        outer_layout.setSpacing(20)
        outer_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        title = QLabel("Continuous–Continuous")
        title.setObjectName("ccPageTitle")

        description = QLabel(
            "Estimate marginal normal distributions and their correlation "
            "from study-level means, variances and sample sizes."
        )
        description.setObjectName("ccPageDescription")
        description.setWordWrap(True)

        outer_layout.addWidget(title)
        outer_layout.addWidget(description)

        input_card = QFrame()
        input_card.setObjectName("ccCard")

        input_layout = QVBoxLayout(input_card)
        input_layout.setContentsMargins(26, 24, 26, 24)
        input_layout.setSpacing(16)

        card_header = QHBoxLayout()

        header_text = QVBoxLayout()
        header_text.setSpacing(3)

        card_title = QLabel("Study-level summaries")
        card_title.setObjectName("ccCardTitle")

        card_subtitle = QLabel(
            "Enter one study per row. Variances are required for the proposed method."
        )
        card_subtitle.setObjectName("ccMutedText")

        header_text.addWidget(card_title)
        header_text.addWidget(card_subtitle)

        card_header.addLayout(header_text)
        card_header.addStretch()

        example_button = QPushButton("Load example")
        example_button.setObjectName("ccSecondaryButton")
        example_button.clicked.connect(self._load_example)

        add_button = QPushButton("Add row")
        add_button.setObjectName("ccSecondaryButton")
        add_button.clicked.connect(self._add_row)

        remove_button = QPushButton("Remove selected")
        remove_button.setObjectName("ccSecondaryButton")
        remove_button.clicked.connect(self._remove_selected_rows)

        card_header.addWidget(example_button)
        card_header.addWidget(add_button)
        card_header.addWidget(remove_button)

        input_layout.addLayout(card_header)

        self.input_table = QTableWidget(5, 5)
        self.input_table.setObjectName("ccInputTable")
        self.input_table.setHorizontalHeaderLabels(
            [
                "Sample size (nᵢ)",
                "Mean X (x̄ᵢ)",
                "Mean Y (ȳᵢ)",
                "Variance X (s²xᵢ)",
                "Variance Y (s²yᵢ)",
            ]
        )
        self.input_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.input_table.verticalHeader().setVisible(False)
        self.input_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.input_table.setAlternatingRowColors(True)
        self.input_table.setItemDelegate(
            CenteredItemDelegate(self.input_table)
        )
        self.input_table.setMinimumHeight(245)

        input_layout.addWidget(self.input_table)

        settings_layout = QHBoxLayout()
        settings_layout.setSpacing(10)

        method_label = QLabel("Method")
        method_label.setObjectName("ccFieldLabel")

        self.method_combo = QComboBox()
        self.method_combo.setObjectName("ccComboBox")
        self.method_combo.addItem("Proposed MLE", "proposed")
        self.method_combo.addItem("Weighted means", "weighted")
        self.method_combo.setMinimumWidth(170)
        self.method_combo.currentIndexChanged.connect(
            self._update_method_settings
        )

        ci_label = QLabel("Confidence interval")
        ci_label.setObjectName("ccFieldLabel")

        self.ci_combo = QComboBox()
        self.ci_combo.setObjectName("ccComboBox")
        self.ci_combo.addItem("Likelihood-ratio", "lr")
        self.ci_combo.addItem("Normal approximation", "normal")
        self.ci_combo.addItem("None", "none")
        self.ci_combo.setMinimumWidth(205)

        settings_layout.addWidget(method_label)
        settings_layout.addWidget(self.method_combo)
        settings_layout.addSpacing(12)
        settings_layout.addWidget(ci_label)
        settings_layout.addWidget(self.ci_combo)
        settings_layout.addStretch()

        self.estimate_button = QPushButton("Estimate distribution")
        self.estimate_button.setObjectName("ccPrimaryButton")
        self.estimate_button.setCursor(
            Qt.CursorShape.PointingHandCursor
        )
        self.estimate_button.clicked.connect(self._run_estimation)

        settings_layout.addWidget(self.estimate_button)

        input_layout.addLayout(settings_layout)
        outer_layout.addWidget(input_card)

        results_card = QFrame()
        results_card.setObjectName("ccCard")

        results_layout = QVBoxLayout(results_card)
        results_layout.setContentsMargins(26, 22, 26, 22)
        results_layout.setSpacing(8)

        results_title = QLabel("Results")
        results_title.setObjectName("ccCardTitle")

        self.results_status = QLabel(
            "Enter study-level summaries and run the estimator."
        )
        self.results_status.setObjectName("ccResultPlaceholder")
        self.results_status.setWordWrap(True)
        self.results_status.setMinimumHeight(135)
        self.results_status.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )

        results_layout.addWidget(results_title)
        results_layout.addWidget(self.results_status)

        outer_layout.addWidget(results_card)
        outer_layout.addStretch()

        scroll_area.setWidget(content_widget)
        page_layout.addWidget(scroll_area)

    def _load_example(self) -> None:
        """Load the data used in the estimator smoke test."""
        example_rows = [
            (100, 10.2, 5.3, 4.0, 2.0),
            (120, 11.1, 5.4, 4.5, 2.2),
            (90, 9.8, 4.9, 3.8, 1.9),
            (110, 10.7, 5.8, 4.2, 2.1),
            (105, 10.4, 5.0, 4.1, 2.0),
        ]

        self.input_table.setRowCount(len(example_rows))

        for row_index, values in enumerate(example_rows):
            for column_index, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                item.setTextAlignment(
                    Qt.AlignmentFlag.AlignCenter
                )
                self.input_table.setItem(
                    row_index,
                    column_index,
                    item,
                )

        self.results_status.setText(
            "Example data loaded. Click “Estimate distribution” "
            "to run the selected method."
        )

    def _add_row(self) -> None:
        row_index = self.input_table.rowCount()
        self.input_table.insertRow(row_index)

    def _remove_selected_rows(self) -> None:
        selected_rows = {
            index.row()
            for index in self.input_table.selectedIndexes()
        }

        for row_index in sorted(selected_rows, reverse=True):
            self.input_table.removeRow(row_index)

        if self.input_table.rowCount() == 0:
            self.input_table.insertRow(0)

    def _update_method_settings(self) -> None:
        """Enable confidence intervals only for the proposed method."""
        is_proposed = (
            self.method_combo.currentData() == "proposed"
        )

        self.ci_combo.setEnabled(is_proposed)

        if not is_proposed:
            none_index = self.ci_combo.findData("none")
            self.ci_combo.setCurrentIndex(none_index)

    def _collect_input_rows(
        self,
    ) -> list[tuple[int, float, float, float | None, float | None]]:
        method = self.method_combo.currentData()
        rows = []

        for row_index in range(self.input_table.rowCount()):
            raw_values = []

            for column_index in range(5):
                item = self.input_table.item(
                    row_index,
                    column_index,
                )
                value = item.text().strip() if item else ""
                raw_values.append(value)

            required_values = raw_values[:3]

            if all(value == "" for value in raw_values):
                continue

            if any(value == "" for value in required_values):
                raise ValueError(
                    f"Row {row_index + 1} must include nᵢ, x̄ᵢ and ȳᵢ."
                )

            if method == "proposed" and any(
                value == "" for value in raw_values[3:]
            ):
                raise ValueError(
                    f"Row {row_index + 1} must include both variances "
                    "for the proposed method."
                )

            try:
                n_value = float(raw_values[0])
                xbar_value = float(raw_values[1])
                ybar_value = float(raw_values[2])
            except ValueError as error:
                raise ValueError(
                    f"Row {row_index + 1} contains a non-numeric value."
                ) from error

            if not n_value.is_integer() or n_value <= 0:
                raise ValueError(
                    f"Row {row_index + 1}: sample size must be "
                    "a positive integer."
                )

            s2x_value = None
            s2y_value = None

            if raw_values[3] != "":
                try:
                    s2x_value = float(raw_values[3])
                except ValueError as error:
                    raise ValueError(
                        f"Row {row_index + 1}: variance X must be numeric."
                    ) from error

            if raw_values[4] != "":
                try:
                    s2y_value = float(raw_values[4])
                except ValueError as error:
                    raise ValueError(
                        f"Row {row_index + 1}: variance Y must be numeric."
                    ) from error

            if method == "proposed":
                if n_value < 4:
                    raise ValueError(
                        f"Row {row_index + 1}: the proposed method "
                        "requires nᵢ ≥ 4."
                    )

                if (
                    s2x_value is None
                    or s2y_value is None
                    or s2x_value <= 0
                    or s2y_value <= 0
                ):
                    raise ValueError(
                        f"Row {row_index + 1}: variances must be positive."
                    )

            rows.append(
                (
                    int(n_value),
                    xbar_value,
                    ybar_value,
                    s2x_value,
                    s2y_value,
                )
            )

        if len(rows) < 2:
            raise ValueError(
                "Enter at least two complete study rows."
            )

        return rows

    def _run_estimation(self) -> None:
        try:
            rows = self._collect_input_rows()

            n = [row[0] for row in rows]
            xbar = [row[1] for row in rows]
            ybar = [row[2] for row in rows]

            method = self.method_combo.currentData()
            ci_method = self.ci_combo.currentData()

            if method == "proposed":
                s2x = [row[3] for row in rows]
                s2y = [row[4] for row in rows]
            else:
                s2x = None
                s2y = None
                ci_method = "none"

            self.results_status.setText("Running estimation...")
            self.estimate_button.setEnabled(False)

            result = estimate_continuous_continuous(
                n=n,
                xbar=xbar,
                ybar=ybar,
                s2x=s2x,
                s2y=s2y,
                method=method,
                ci_method=ci_method,
            )

        except (ValueError, RuntimeError) as error:
            self.results_status.setText(
                f"Estimation could not be completed.\n\n{error}"
            )
            QMessageBox.warning(
                self,
                "Estimation error",
                str(error),
            )
            return

        finally:
            self.estimate_button.setEnabled(True)

        se_text = (
            "Not available"
            if result.se is None
            else f"{result.se:.6f}"
        )

        if result.ci_lower is None or result.ci_upper is None:
            ci_text = "Not available"
        else:
            ci_text = (
                f"[{result.ci_lower:.6f}, "
                f"{result.ci_upper:.6f}]"
            )

        self.results_status.setText(
            f"Estimation completed successfully.\n\n"
            f"Method: {result.method}\n"
            f"Number of studies: {len(rows)}\n"
            f"Total sample size: {sum(n):,}\n\n"
            f"Estimated mean μx:                 {result.mu_x:.6f}\n"
            f"Estimated mean μy:                 {result.mu_y:.6f}\n"
            f"Estimated standard deviation σx:   {result.sigma_x:.6f}\n"
            f"Estimated standard deviation σy:   {result.sigma_y:.6f}\n"
            f"Estimated correlation ρ:           {result.rho:.6f}\n"
            f"Standard error of ρ:               {se_text}\n"
            f"95% confidence interval for ρ:      {ci_text}"
        )

    def _apply_styles(self) -> None:
        self.setStyleSheet(
            """
            QScrollArea#ccScrollArea {
                border: none;
                background: #ffffff;
            }

            QWidget#ccScrollContent {
                background: #ffffff;
            }

            QLabel#ccPageTitle {
                font-size: 29px;
                font-weight: 650;
                color: #202123;
            }

            QLabel#ccPageDescription {
                font-size: 15px;
                color: #666666;
            }

            QFrame#ccCard {
                background: #ffffff;
                border: 1px solid #e5e5e5;
                border-radius: 16px;
            }

            QLabel#ccCardTitle {
                font-size: 17px;
                font-weight: 600;
                color: #202123;
            }

            QLabel#ccMutedText {
                font-size: 12px;
                color: #7d7d7d;
            }

            QLabel#ccFieldLabel {
                font-size: 13px;
                font-weight: 550;
                color: #383838;
            }

            QTableWidget#ccInputTable {
                background: #ffffff;
                alternate-background-color: #fafafa;
                border: 1px solid #dddddd;
                border-radius: 10px;
                gridline-color: #eeeeee;
                font-size: 13px;
                selection-background-color: #e9e9eb;
                selection-color: #202123;
            }

            QTableWidget#ccInputTable::item {
                padding: 8px;
            }

            QHeaderView::section {
                background: #f7f7f8;
                color: #555555;
                border: none;
                border-bottom: 1px solid #dddddd;
                padding: 9px;
                font-size: 12px;
                font-weight: 600;
            }

            QPushButton#ccSecondaryButton {
                min-height: 32px;
                padding: 0 12px;
                border: 1px solid #d8d8d8;
                border-radius: 8px;
                background: #ffffff;
                color: #333333;
                font-size: 12px;
            }

            QPushButton#ccSecondaryButton:hover {
                background: #f3f3f3;
            }

            QComboBox#ccComboBox {
                min-height: 34px;
                padding: 0 34px 0 11px;
                border: 1px solid #d8d8d8;
                border-radius: 8px;
                background: #ffffff;
                font-size: 13px;
            }

            QComboBox#ccComboBox:hover {
                border-color: #bdbdbd;
            }

            QComboBox#ccComboBox:focus {
                border-color: #a8a8a8;
            }

            QComboBox#ccComboBox:disabled {
                background: #f4f4f4;
                color: #999999;
            }

            QComboBox#ccComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 32px;
                border: none;
                background: transparent;
            }

            QComboBox#ccComboBox::down-arrow {
                width: 10px;
                height: 10px;
            }

            QPushButton#ccPrimaryButton {
                min-height: 36px;
                padding: 0 18px;
                border: none;
                border-radius: 9px;
                background: #202123;
                color: #ffffff;
                font-size: 13px;
                font-weight: 600;
            }

            QPushButton#ccPrimaryButton:hover {
                background: #343541;
            }

            QPushButton#ccPrimaryButton:pressed {
                background: #111111;
            }

            QLabel#ccResultPlaceholder {
                padding: 18px;
                border: 1px dashed #d8d8d8;
                border-radius: 10px;
                background: #fafafa;
                color: #666666;
                font-size: 13px;
            }
            """
        )
