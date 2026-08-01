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

from ebdm_app.estimators.binary_continuous import (
    estimate_binary_continuous,
)


class CenteredItemDelegate(QStyledItemDelegate):
    """Center table values in display and editing modes."""

    def initStyleOption(self, option, index) -> None:
        super().initStyleOption(option, index)
        option.displayAlignment = Qt.AlignmentFlag.AlignCenter

    def createEditor(self, parent, option, index):
        editor = super().createEditor(parent, option, index)

        if hasattr(editor, "setAlignment"):
            editor.setAlignment(Qt.AlignmentFlag.AlignCenter)

        return editor


class BinaryContinuousPage(QWidget):
    """Interface for binary–continuous estimation."""

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
        scroll_area.setObjectName("bcScrollArea")
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        scroll_area.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )

        content_widget = QWidget()
        content_widget.setObjectName("bcScrollContent")

        outer_layout = QVBoxLayout(content_widget)
        outer_layout.setContentsMargins(46, 38, 46, 38)
        outer_layout.setSpacing(20)
        outer_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        title = QLabel("Binary–Continuous")
        title.setObjectName("bcPageTitle")

        description = QLabel(
            "Estimate group-specific continuous distributions from "
            "binary-group counts and aggregate study-level summaries."
        )
        description.setObjectName("bcPageDescription")
        description.setWordWrap(True)

        outer_layout.addWidget(title)
        outer_layout.addWidget(description)

        input_card = QFrame()
        input_card.setObjectName("bcCard")

        input_layout = QVBoxLayout(input_card)
        input_layout.setContentsMargins(26, 24, 26, 24)
        input_layout.setSpacing(16)

        card_header = QHBoxLayout()

        header_text = QVBoxLayout()
        header_text.setSpacing(3)

        card_title = QLabel("Study-level summaries")
        card_title.setObjectName("bcCardTitle")

        self.card_subtitle = QLabel(
            "Enter one study per row. Variances are required for Scaled GMM."
        )
        self.card_subtitle.setObjectName("bcMutedText")

        header_text.addWidget(card_title)
        header_text.addWidget(self.card_subtitle)

        card_header.addLayout(header_text)
        card_header.addStretch()

        example_button = QPushButton("Load example")
        example_button.setObjectName("bcSecondaryButton")
        example_button.clicked.connect(self._load_example)

        add_button = QPushButton("Add row")
        add_button.setObjectName("bcSecondaryButton")
        add_button.clicked.connect(self._add_row)

        remove_button = QPushButton("Remove selected")
        remove_button.setObjectName("bcSecondaryButton")
        remove_button.clicked.connect(self._remove_selected_rows)

        card_header.addWidget(example_button)
        card_header.addWidget(add_button)
        card_header.addWidget(remove_button)

        input_layout.addLayout(card_header)

        self.input_table = QTableWidget(8, 4)
        self.input_table.setObjectName("bcInputTable")
        self.input_table.setHorizontalHeaderLabels(
            [
                "Sample size (nᵢ)",
                "Group 1 count (mᵢ)",
                "Overall mean (x̄ᵢ)",
                "Overall variance (s²ᵢ)",
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
        self.input_table.setMinimumHeight(275)

        input_layout.addWidget(self.input_table)

        settings_layout = QHBoxLayout()
        settings_layout.setSpacing(10)

        method_label = QLabel("Method")
        method_label.setObjectName("bcFieldLabel")

        self.method_combo = QComboBox()
        self.method_combo.setObjectName("bcComboBox")
        self.method_combo.addItem("Scaled GMM", "gmm")
        self.method_combo.addItem("Naive estimator", "naive")
        self.method_combo.setMinimumWidth(190)
        self.method_combo.currentIndexChanged.connect(
            self._update_method_settings
        )

        settings_layout.addWidget(method_label)
        settings_layout.addWidget(self.method_combo)
        settings_layout.addStretch()

        self.estimate_button = QPushButton(
            "Estimate group distributions"
        )
        self.estimate_button.setObjectName("bcPrimaryButton")
        self.estimate_button.setCursor(
            Qt.CursorShape.PointingHandCursor
        )
        self.estimate_button.clicked.connect(
            self._run_estimation
        )

        settings_layout.addWidget(self.estimate_button)

        input_layout.addLayout(settings_layout)
        outer_layout.addWidget(input_card)

        results_card = QFrame()
        results_card.setObjectName("bcCard")

        results_layout = QVBoxLayout(results_card)
        results_layout.setContentsMargins(26, 22, 26, 22)
        results_layout.setSpacing(8)

        results_title = QLabel("Results")
        results_title.setObjectName("bcCardTitle")

        self.results_status = QLabel(
            "Enter study-level summaries and run the estimator."
        )
        self.results_status.setObjectName("bcResultPlaceholder")
        self.results_status.setWordWrap(True)
        self.results_status.setMinimumHeight(185)
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
        """Load the example used in the GMM smoke test."""
        example_rows = [
            (80, 16, 8.950000, 10.792405),
            (100, 30, 9.100000, 10.743939),
            (120, 48, 9.680000, 10.972269),
            (90, 45, 9.950000, 10.444944),
            (110, 66, 10.520000, 9.925229),
            (130, 91, 10.720000, 8.836047),
            (95, 76, 11.250000, 7.707234),
            (105, 95, 11.549048, 5.788132),
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
            "Example data loaded. Click “Estimate group distributions” "
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
        method = self.method_combo.currentData()

        if method == "gmm":
            self.card_subtitle.setText(
                "Enter one study per row. Variances are required "
                "for Scaled GMM."
            )
        else:
            self.card_subtitle.setText(
                "The Naive estimator requires sample sizes, group counts "
                "and overall means. Variances are optional."
            )

    def _collect_input_rows(
        self,
    ) -> list[tuple[int, int, float, float | None]]:
        method = self.method_combo.currentData()
        rows = []

        for row_index in range(self.input_table.rowCount()):
            raw_values = []

            for column_index in range(4):
                item = self.input_table.item(
                    row_index,
                    column_index,
                )
                value = item.text().strip() if item else ""
                raw_values.append(value)

            if all(value == "" for value in raw_values):
                continue

            if any(value == "" for value in raw_values[:3]):
                raise ValueError(
                    f"Row {row_index + 1} must include nᵢ, mᵢ and x̄ᵢ."
                )

            if method == "gmm" and raw_values[3] == "":
                raise ValueError(
                    f"Row {row_index + 1} must include s²ᵢ "
                    "for Scaled GMM."
                )

            try:
                n_value = float(raw_values[0])
                m_value = float(raw_values[1])
                mean_value = float(raw_values[2])
            except ValueError as error:
                raise ValueError(
                    f"Row {row_index + 1} contains a non-numeric value."
                ) from error

            if not n_value.is_integer() or n_value <= 0:
                raise ValueError(
                    f"Row {row_index + 1}: nᵢ must be "
                    "a positive integer."
                )

            if not m_value.is_integer():
                raise ValueError(
                    f"Row {row_index + 1}: mᵢ must be an integer."
                )

            if not 0 <= m_value <= n_value:
                raise ValueError(
                    f"Row {row_index + 1}: mᵢ must satisfy "
                    "0 ≤ mᵢ ≤ nᵢ."
                )

            variance_value = None

            if raw_values[3] != "":
                try:
                    variance_value = float(raw_values[3])
                except ValueError as error:
                    raise ValueError(
                        f"Row {row_index + 1}: s²ᵢ must be numeric."
                    ) from error

                if variance_value <= 0:
                    raise ValueError(
                        f"Row {row_index + 1}: s²ᵢ must be positive."
                    )

            rows.append(
                (
                    int(n_value),
                    int(m_value),
                    mean_value,
                    variance_value,
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

            ni = [row[0] for row in rows]
            mi = [row[1] for row in rows]
            xbar = [row[2] for row in rows]

            method = self.method_combo.currentData()

            if method == "gmm":
                s2 = [row[3] for row in rows]
            elif all(row[3] is not None for row in rows):
                s2 = [row[3] for row in rows]
            else:
                s2 = None

            self.results_status.setText("Running estimation...")
            self.estimate_button.setEnabled(False)

            result = estimate_binary_continuous(
                ni=ni,
                mi=mi,
                xbar=xbar,
                s2=s2,
                method=method,
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

        convergence_text = (
            "Yes" if result.converged else "No"
        )

        objective_text = (
            "Not available"
            if result.objective_value is None
            else f"{result.objective_value:.6f}"
        )

        warning_text = ""

        if result.method == "Naive estimator":
            larger_sigma = max(
                result.sigma1_hat,
                result.sigma0_hat,
            )
            smaller_sigma = min(
                result.sigma1_hat,
                result.sigma0_hat,
            )

            if (
                larger_sigma > 0
                and smaller_sigma / larger_sigma < 0.01
            ):
                warning_text = (
                    "\n\nWarning: one estimated standard deviation is "
                    "close to zero. The naive estimate may represent a "
                    "boundary solution and should be interpreted cautiously."
                )

        self.results_status.setText(
            f"Estimation completed successfully.\n\n"
            f"Method: {result.method}\n"
            f"Number of studies: {len(rows)}\n"
            f"Total sample size: {sum(ni):,}\n"
            f"Optimization converged: {convergence_text}\n"
            f"Objective value: {objective_text}\n\n"
            f"Group 1 mean μ₁:     "
            f"{self._format_estimate(result.mu1_hat, result.se_mu1, result.ci_mu1)}\n"
            f"Group 0 mean μ₀:     "
            f"{self._format_estimate(result.mu0_hat, result.se_mu0, result.ci_mu0)}\n"
            f"Group 1 SD σ₁:       "
            f"{self._format_estimate(result.sigma1_hat, result.se_sigma1, result.ci_sigma1)}\n"
            f"Group 0 SD σ₀:       "
            f"{self._format_estimate(result.sigma0_hat, result.se_sigma0, result.ci_sigma0)}"
            f"{warning_text}"
        )

    @staticmethod
    def _format_estimate(
        estimate: float,
        standard_error: float | None,
        confidence_interval: tuple[float, float] | None,
    ) -> str:
        if standard_error is None:
            return f"{estimate:.6f}   (SE and CI not available)"

        if confidence_interval is None:
            return (
                f"{estimate:.6f}   "
                f"SE: {standard_error:.6f}"
            )

        return (
            f"{estimate:.6f}   "
            f"SE: {standard_error:.6f}   "
            f"95% CI: [{confidence_interval[0]:.6f}, "
            f"{confidence_interval[1]:.6f}]"
        )

    def _apply_styles(self) -> None:
        self.setStyleSheet(
            """
            QScrollArea#bcScrollArea {
                border: none;
                background: #ffffff;
            }

            QWidget#bcScrollContent {
                background: #ffffff;
            }

            QLabel#bcPageTitle {
                font-size: 29px;
                font-weight: 650;
                color: #202123;
            }

            QLabel#bcPageDescription {
                font-size: 15px;
                color: #666666;
            }

            QFrame#bcCard {
                background: #ffffff;
                border: 1px solid #e5e5e5;
                border-radius: 16px;
            }

            QLabel#bcCardTitle {
                font-size: 17px;
                font-weight: 600;
                color: #202123;
            }

            QLabel#bcMutedText {
                font-size: 12px;
                color: #7d7d7d;
            }

            QLabel#bcFieldLabel {
                font-size: 13px;
                font-weight: 550;
                color: #383838;
            }

            QTableWidget#bcInputTable {
                background: #ffffff;
                alternate-background-color: #fafafa;
                border: 1px solid #dddddd;
                border-radius: 10px;
                gridline-color: #eeeeee;
                font-size: 13px;
                selection-background-color: #e9e9eb;
                selection-color: #202123;
            }

            QTableWidget#bcInputTable::item {
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

            QPushButton#bcSecondaryButton {
                min-height: 32px;
                padding: 0 12px;
                border: 1px solid #d8d8d8;
                border-radius: 8px;
                background: #ffffff;
                color: #333333;
                font-size: 12px;
            }

            QPushButton#bcSecondaryButton:hover {
                background: #f3f3f3;
            }

            QComboBox#bcComboBox {
                min-height: 34px;
                padding: 0 34px 0 11px;
                border: 1px solid #d8d8d8;
                border-radius: 8px;
                background: #ffffff;
                font-size: 13px;
            }

            QComboBox#bcComboBox:hover {
                border-color: #bdbdbd;
            }

            QComboBox#bcComboBox:focus {
                border-color: #a8a8a8;
            }

            QComboBox#bcComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 32px;
                border: none;
                background: transparent;
            }

            QComboBox#bcComboBox::down-arrow {
                width: 10px;
                height: 10px;
            }

            QPushButton#bcPrimaryButton {
                min-height: 36px;
                padding: 0 18px;
                border: none;
                border-radius: 9px;
                background: #202123;
                color: #ffffff;
                font-size: 13px;
                font-weight: 600;
            }

            QPushButton#bcPrimaryButton:hover {
                background: #343541;
            }

            QPushButton#bcPrimaryButton:pressed {
                background: #111111;
            }

            QLabel#bcResultPlaceholder {
                padding: 18px;
                border: 1px dashed #d8d8d8;
                border-radius: 10px;
                background: #fafafa;
                color: #666666;
                font-size: 13px;
            }
            """
        )
