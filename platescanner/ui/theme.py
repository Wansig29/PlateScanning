"""Dark console theme for the gate screen."""

BG = "#0e1318"
PANEL = "#151c24"
PANEL_ALT = "#1b2430"
BORDER = "#26313e"
TEXT = "#e6edf3"
MUTED = "#8b98a8"
ACCENT = "#3b82f6"

RED = "#e5484d"
GREEN = "#3fb950"
AMBER = "#f5a524"
GREY = "#5b6776"

RESULT_COLORS = {"violation": RED, "clear": GREEN, "not_registered": AMBER, "no_plate": MUTED}
RESULT_LABELS = {"violation": "VIOLATION", "clear": "NO VIOLATION", "not_registered": "NOT REGISTERED",
                 "no_plate": "NO PLATE READ"}

STYLESHEET = f"""
QWidget {{
    background: {BG};
    color: {TEXT};
    font-family: "Segoe UI", sans-serif;
    font-size: 10pt;
}}
QFrame#Panel {{
    background: {PANEL};
    border: 1px solid {BORDER};
    border-radius: 8px;
}}
QFrame#Panel QWidget {{ background: transparent; }}
QLabel#PanelTitle {{
    color: {MUTED};
    font-size: 9pt;
    font-weight: 600;
    letter-spacing: 1px;
}}
QLabel#FieldName {{ color: {MUTED}; }}
QLabel#FieldValue {{ font-weight: 600; }}
QLabel#Muted {{ color: {MUTED}; }}
QLabel#ImageSlot {{
    background: {PANEL_ALT};
    border: 1px dashed {BORDER};
    border-radius: 6px;
    color: {MUTED};
}}
QFrame#TopBar {{
    background: {PANEL};
    border-bottom: 1px solid {BORDER};
}}
QFrame#TopBar QWidget {{ background: transparent; }}
QLabel#AppTitle {{ font-size: 12pt; font-weight: 700; }}
QPushButton {{
    background: {PANEL_ALT};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 6px 14px;
}}
QPushButton:hover {{ border-color: {ACCENT}; }}
QPushButton:disabled {{ color: {MUTED}; }}
QPushButton#Primary {{ background: {ACCENT}; border-color: {ACCENT}; color: white; font-weight: 600; }}
QPushButton#Primary:disabled {{ background: {PANEL_ALT}; border-color: {BORDER}; color: {MUTED}; }}
QToolButton {{
    background: {PANEL_ALT};
    border: 1px solid {BORDER};
    border-radius: 5px;
    padding: 2px 8px;
}}
QToolButton:hover {{ border-color: {ACCENT}; }}
QLineEdit {{
    background: {PANEL_ALT};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 7px 9px;
}}
QLineEdit:focus {{ border-color: {ACCENT}; }}
QTableWidget {{
    background: transparent;
    border: none;
    gridline-color: transparent;
    selection-background-color: {PANEL_ALT};
    selection-color: {TEXT};
}}
QTableWidget::item {{ padding: 4px 6px; border-bottom: 1px solid {BORDER}; }}
QHeaderView::section {{
    background: transparent;
    color: {MUTED};
    border: none;
    border-bottom: 1px solid {BORDER};
    padding: 4px 6px;
    font-weight: 600;
}}
QScrollBar:vertical {{ background: transparent; width: 10px; }}
QScrollBar::handle:vertical {{ background: {BORDER}; border-radius: 5px; min-height: 30px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; }}
QSplitter::handle {{ background: {BG}; }}
QStatusBar {{ background: {PANEL}; color: {MUTED}; border-top: 1px solid {BORDER}; }}
QStatusBar QLabel {{ background: transparent; color: {MUTED}; padding: 0 8px; }}
"""
