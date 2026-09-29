"""Dark console theme for the gate screen."""

BG = "#0b1016"
PANEL = "#121a23"
PANEL_ALT = "#1a2430"
PANEL_HOVER = "#212d3b"
BORDER = "#253140"
BORDER_STRONG = "#344357"
TEXT = "#e8eef5"
MUTED = "#8a98aa"
FAINT = "#5f6d80"
ACCENT = "#3b82f6"
ACCENT_HOVER = "#5b98f8"
VIDEO_BG = "#05080b"

RED = "#ef4444"
GREEN = "#22c55e"
AMBER = "#f59e0b"
GREY = "#5b6776"

MONO = '"Cascadia Mono", Consolas, monospace'

RESULT_COLORS = {"violation": RED, "clear": GREEN, "not_registered": AMBER, "no_plate": MUTED}
RESULT_LABELS = {"violation": "VIOLATION", "clear": "NO VIOLATION", "not_registered": "NOT REGISTERED",
                 "no_plate": "NO PLATE READ"}
RESULT_ICONS = {"violation": "⛔", "clear": "✔", "not_registered": "?", "no_plate": "–"}
# Dark, low-saturation fills behind status text (badges, banners, rows).
RESULT_TINTS = {"violation": "#3a1418", "clear": "#0f2e1c", "not_registered": "#35260a", "no_plate": "#1a2430"}


def dot(color: str, text: str) -> str:
    """Rich-text status line with a coloured dot in front."""
    return f'<span style="color:{color}; font-size:11pt;">●</span>&nbsp; {text}'


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
    border-radius: 10px;
}}
QFrame#Panel QWidget {{ background: transparent; }}
QLabel#PanelTitle {{
    color: {MUTED};
    font-size: 9pt;
    font-weight: 700;
    letter-spacing: 1px;
}}
QLabel#FieldName {{ color: {MUTED}; font-size: 9pt; }}
QLabel#FieldValue {{ font-weight: 600; font-size: 10.5pt; }}
QLabel#Muted {{ color: {MUTED}; }}
QLabel#Faint {{ color: {FAINT}; font-size: 9pt; }}
QLabel#OwnerName {{ font-size: 15pt; font-weight: 700; }}
QLabel#PlateChip, QFrame#Panel QLabel#PlateChip, QFrame#TopBar QLabel#PlateChip {{
    background: #f4f6f8;
    color: #0b1016;
    border: 2px solid #0b1016;
    border-radius: 5px;
    padding: 2px 10px;
    font-family: {MONO};
    font-size: 15pt;
    font-weight: 800;
    letter-spacing: 2px;
}}
QLabel#ImageSlot, QFrame#Panel QLabel#ImageSlot, QFrame#TopBar QLabel#ImageSlot {{
    background: {PANEL_ALT};
    border: 1px dashed {BORDER_STRONG};
    border-radius: 8px;
    color: {FAINT};
    font-size: 9pt;
}}
QFrame#DetailCard, QFrame#Panel QFrame#DetailCard, QFrame#TopBar QFrame#DetailCard {{
    background: {PANEL_ALT};
    border: 1px solid {BORDER};
    border-radius: 8px;
}}
QFrame#DetailCard QLabel {{ background: transparent; }}
QFrame#TopBar {{
    background: {PANEL};
    border-bottom: 1px solid {BORDER};
}}
QFrame#TopBar QWidget {{ background: transparent; }}
QLabel#AppTitle {{ font-size: 12.5pt; font-weight: 700; }}
QLabel#AppSubtitle {{ color: {MUTED}; font-size: 8.5pt; }}
QLabel#BrandMark, QFrame#Panel QLabel#BrandMark, QFrame#TopBar QLabel#BrandMark {{
    background: {ACCENT};
    color: white;
    border-radius: 8px;
    font-size: 13pt;
    font-weight: 800;
}}
QLabel#Clock {{
    font-family: {MONO};
    font-size: 15pt;
    font-weight: 600;
}}
QLabel#ClockDate {{ color: {MUTED}; font-size: 8.5pt; }}
QFrame#VSep {{ background: {BORDER}; max-width: 1px; min-width: 1px; }}
QPushButton, QFrame#Panel QPushButton, QFrame#TopBar QPushButton {{
    background: {PANEL_ALT};
    border: 1px solid {BORDER};
    border-radius: 7px;
    padding: 7px 16px;
}}
QPushButton:hover, QFrame#Panel QPushButton:hover, QFrame#TopBar QPushButton:hover {{ background: {PANEL_HOVER}; border-color: {BORDER_STRONG}; }}
QPushButton:pressed, QFrame#Panel QPushButton:pressed, QFrame#TopBar QPushButton:pressed {{ background: {BORDER}; }}
QPushButton:disabled, QFrame#Panel QPushButton:disabled, QFrame#TopBar QPushButton:disabled {{ color: {MUTED}; }}
QPushButton#Primary, QFrame#Panel QPushButton#Primary, QFrame#TopBar QPushButton#Primary {{ background: {ACCENT}; border-color: {ACCENT}; color: white; font-weight: 600; }}
QPushButton#Primary:hover, QFrame#Panel QPushButton#Primary:hover, QFrame#TopBar QPushButton#Primary:hover {{ background: {ACCENT_HOVER}; border-color: {ACCENT_HOVER}; }}
QPushButton#Primary:pressed, QFrame#Panel QPushButton#Primary:pressed, QFrame#TopBar QPushButton#Primary:pressed {{ background: #2f6fd8; }}
QPushButton#Primary:disabled, QFrame#Panel QPushButton#Primary:disabled, QFrame#TopBar QPushButton#Primary:disabled {{ background: {PANEL_ALT}; border-color: {BORDER}; color: {MUTED}; }}
QPushButton#Link, QFrame#Panel QPushButton#Link, QFrame#TopBar QPushButton#Link {{
    background: transparent;
    border: none;
    color: {MUTED};
    padding: 6px 4px;
}}
QPushButton#Link:hover, QFrame#Panel QPushButton#Link:hover, QFrame#TopBar QPushButton#Link:hover {{ color: {TEXT}; text-decoration: underline; }}
QToolButton, QFrame#Panel QToolButton, QFrame#TopBar QToolButton {{
    background: {PANEL_ALT};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 4px 10px;
}}
QToolButton:hover, QFrame#Panel QToolButton:hover, QFrame#TopBar QToolButton:hover {{ background: {PANEL_HOVER}; border-color: {BORDER_STRONG}; }}
QToolButton:pressed, QFrame#Panel QToolButton:pressed, QFrame#TopBar QToolButton:pressed {{ background: {BORDER}; }}
QToolButton::menu-indicator {{ image: none; width: 0; }}
QMenu {{
    background: {PANEL_ALT};
    border: 1px solid {BORDER_STRONG};
    border-radius: 8px;
    padding: 5px;
}}
QMenu::item {{ background: transparent; padding: 6px 26px 6px 12px; border-radius: 5px; }}
QMenu::item:selected {{ background: {PANEL_HOVER}; }}
QMenu::indicator {{ width: 0; }}
QMenu::item:checked {{ color: {ACCENT_HOVER}; font-weight: 600; }}
QToolTip {{
    background: {PANEL_ALT};
    color: {TEXT};
    border: 1px solid {BORDER_STRONG};
    padding: 6px 8px;
}}
QLineEdit {{
    background: {PANEL_ALT};
    border: 1px solid {BORDER};
    border-radius: 7px;
    padding: 9px 11px;
    selection-background-color: {ACCENT};
}}
QLineEdit:hover {{ border-color: {BORDER_STRONG}; }}
QLineEdit:focus {{ border-color: {ACCENT}; }}
QTableWidget {{
    background: transparent;
    alternate-background-color: rgba(255, 255, 255, 0.018);
    border: none;
    gridline-color: transparent;
    selection-background-color: {PANEL_HOVER};
    selection-color: {TEXT};
}}
QTableWidget::item {{ padding: 0 8px; border-bottom: 1px solid {BORDER}; }}
QTabWidget::pane {{ border: none; border-top: 1px solid {BORDER}; }}
QTabBar::tab {{
    background: transparent;
    color: {MUTED};
    border: none;
    border-bottom: 2px solid transparent;
    padding: 7px 16px;
    font-weight: 600;
}}
QTabBar::tab:hover {{ color: {TEXT}; }}
QTabBar::tab:selected {{ color: {TEXT}; border-bottom: 2px solid {ACCENT}; }}
QHeaderView::section {{
    background: transparent;
    color: {FAINT};
    border: none;
    border-bottom: 1px solid {BORDER};
    padding: 5px 8px;
    font-size: 8.5pt;
    font-weight: 700;
}}
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: {BORDER}; border-radius: 3px; min-height: 30px; }}
QScrollBar::handle:vertical:hover {{ background: {BORDER_STRONG}; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 2px; }}
QScrollBar::handle:horizontal {{ background: {BORDER}; border-radius: 3px; min-width: 30px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}
QSplitter::handle {{ background: {BG}; }}
QSplitter::handle:hover {{ background: {BORDER}; }}
QStatusBar {{ background: {PANEL}; color: {MUTED}; border-top: 1px solid {BORDER}; }}
QStatusBar::item {{ border: none; }}
QStatusBar QLabel {{ background: transparent; color: {MUTED}; padding: 2px 10px; }}
QMessageBox QLabel {{ background: transparent; }}
QDialog {{ background: {BG}; }}
"""
