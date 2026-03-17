"""Model selection widget."""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


@dataclass(slots=True)
class ModelItemState:
    """Represents model state in selector."""

    model_id: str
    checked: bool


class ModelSelector(QWidget):
    """Handle model activation and cascade order selection."""

    orderChanged = Signal(list)
    selectionChanged = Signal(list)
    loadClicked = Signal()
    removeClicked = Signal(str)
    modelFocused = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._ordering_enabled = True
        self._model_details: dict[str, str] = {}
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        self.title_label = QLabel("Modelos (ordem de cascata)")
        self.title_label.setObjectName("modelSelectorTitle")
        root.addWidget(self.title_label)

        self.model_list = QListWidget()
        self.model_list.setDragDropMode(QAbstractItemView.InternalMove)
        self.model_list.setDefaultDropAction(Qt.MoveAction)
        self.model_list.model().rowsMoved.connect(self._emit_order_changed)
        self.model_list.currentItemChanged.connect(self._emit_model_focused)
        root.addWidget(self.model_list, stretch=1)

        controls = QHBoxLayout()
        self.load_button = QPushButton("Carregar")
        self.remove_button = QPushButton("Remover")
        controls.addWidget(self.load_button)
        controls.addWidget(self.remove_button)
        root.addLayout(controls)

        self.load_button.clicked.connect(self.loadClicked.emit)
        self.remove_button.clicked.connect(self._emit_remove_selected)
        self.set_ordering_enabled(True)

        details_title = QLabel("Descrição do modelo")
        root.addWidget(details_title)
        self.details_view = QTextEdit()
        self.details_view.setReadOnly(True)
        self.details_view.setPlaceholderText("Selecione um modelo para ver informações e classes.")
        self.details_view.setMinimumHeight(130)
        root.addWidget(self.details_view)

    def set_models(self, model_ids: list[str], selected_ids: list[str] | None = None) -> None:
        """Populate selector list and checked state."""
        selected = set(selected_ids or [])
        self.model_list.clear()

        for model_id in model_ids:
            self._add_list_item(model_id=model_id, checked=model_id in selected)

        self._refresh_priority_labels()
        self._emit_selection_changed()
        self._emit_order_changed()
        if self.model_list.count() > 0 and self.model_list.currentItem() is None:
            self.model_list.setCurrentRow(0)

    def add_model(self, model_id: str, checked: bool = True) -> None:
        """Add one model entry to list."""
        self._add_list_item(model_id=model_id, checked=checked)
        self._refresh_priority_labels()
        self._emit_selection_changed()
        self._emit_order_changed()

    def selected_models(self) -> list[str]:
        """Return checked models in current cascade order."""
        selected: list[str] = []
        for row in range(self.model_list.count()):
            item = self.model_list.item(row)
            checkbox = self.model_list.itemWidget(item)
            if isinstance(checkbox, QCheckBox) and checkbox.isChecked():
                selected.append(item.data(Qt.UserRole))
        return selected

    def ordered_models(self) -> list[str]:
        """Return all models in current visual order."""
        ordered: list[str] = []
        for row in range(self.model_list.count()):
            item = self.model_list.item(row)
            model_id = item.data(Qt.UserRole)
            if isinstance(model_id, str):
                ordered.append(model_id)
        return ordered

    def _add_list_item(self, model_id: str, checked: bool) -> None:
        item = QListWidgetItem()
        item.setData(Qt.UserRole, model_id)
        item.setFlags(
            item.flags()
            | Qt.ItemIsDragEnabled
            | Qt.ItemIsDropEnabled
            | Qt.ItemIsSelectable
            | Qt.ItemIsEnabled
        )
        checkbox = QCheckBox(model_id)
        checkbox.setProperty("model_id", model_id)
        checkbox.setChecked(checked)
        checkbox.stateChanged.connect(self._emit_selection_changed)
        self.model_list.addItem(item)
        self.model_list.setItemWidget(item, checkbox)
        item.setSizeHint(checkbox.sizeHint())

    def _emit_selection_changed(self, *_: object) -> None:
        self.selectionChanged.emit(self.selected_models())

    def _emit_order_changed(self, *_: object) -> None:
        self._refresh_priority_labels()
        self.orderChanged.emit(self.ordered_models())

    def _emit_remove_selected(self) -> None:
        item = self.model_list.currentItem()
        if item is None:
            return
        model_id = item.data(Qt.UserRole)
        if isinstance(model_id, str):
            self.removeClicked.emit(model_id)

    def set_ordering_enabled(self, enabled: bool) -> None:
        """Enable drag-and-drop ordering when cascade mode is active."""
        self._ordering_enabled = enabled
        self.title_label.setText("Modelos (ordem de cascata)" if enabled else "Modelos (simultâneo)")
        self.model_list.setDragEnabled(enabled)
        self.model_list.setAcceptDrops(enabled)
        self.model_list.viewport().setAcceptDrops(enabled)
        self.model_list.setDropIndicatorShown(enabled)
        drag_mode = QAbstractItemView.InternalMove if enabled else QAbstractItemView.NoDragDrop
        self.model_list.setDragDropMode(drag_mode)
        self._refresh_priority_labels()

    def set_model_details(self, details: dict[str, str]) -> None:
        """Set details/description text per model id."""
        self._model_details = details
        self._emit_model_focused()

    def _refresh_priority_labels(self) -> None:
        for row in range(self.model_list.count()):
            item = self.model_list.item(row)
            checkbox = self.model_list.itemWidget(item)
            if not isinstance(checkbox, QCheckBox):
                continue
            model_id = checkbox.property("model_id")
            if not isinstance(model_id, str):
                model_id = str(item.data(Qt.UserRole))
            if self._ordering_enabled:
                checkbox.setText(f"{row + 1}. {model_id}")
            else:
                checkbox.setText(model_id)

    def _emit_model_focused(self, *_: object) -> None:
        item = self.model_list.currentItem()
        if item is None:
            self.details_view.clear()
            return
        model_id = item.data(Qt.UserRole)
        if not isinstance(model_id, str):
            self.details_view.clear()
            return
        self.modelFocused.emit(model_id)
        self.details_view.setPlainText(self._model_details.get(model_id, "Sem descrição disponível."))
