"""PSAU-branded dark console theme (forest green and gold) for the gate screen."""

# PSAU palette, taken from the university logo: deep forest green + gold ring.
BG = "#07140f"
PANEL = "#0d2118"
PANEL_ALT = "#14301f"
PANEL_HOVER = "#1b3d2a"
BORDER = "#1f4531"
BORDER_STRONG = "#2e6045"
TEXT = "#f4f1e6"
MUTED = "#9db3a5"
FAINT = "#6b8576"
ACCENT = "#f5d31f"          # PSAU gold
ACCENT_HOVER = "#ffe24d"
ACCENT_PRESSED = "#d9b80f"
ON_ACCENT = "#10402f"       # PSAU green, for text on gold
VIDEO_BG = "#030a07"

RED = "#ef4444"
GREEN = "#34d27b"
AMBER = "#f59e0b"

MONO = '"Cascadia Mono", Consolas, monospace'

RESULT_COLORS = {"violation": RED, "clear": GREEN, "not_registered": AMBER, "no_plate": MUTED}
RESULT_LABELS = {"violation": "VIOLATION", "clear": "NO VIOLATION", "not_registered": "NOT REGISTERED",
                 "no_plate": "NO PLATE READ"}
RESULT_ICONS = {"violation": "⛔", "clear": "✔", "not_registered": "?", "no_plate": "–"}
# Dark, low-saturation fills behind status text (badges, banners, rows).
RESULT_TINTS = {"violation": "#3a1418", "clear": "#0f3a22", "not_registered": "#3a2c08", "no_plate": "#14301f"}


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
    background: #f7f5ec;
    color: #10402f;
    border: 2px solid #10402f;
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
    background: transparent;
    color: {ACCENT};
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
QPushButton#Primary, QFrame#Panel QPushButton#Primary, QFrame#TopBar QPushButton#Primary {{ background: {ACCENT}; border-color: {ACCENT}; color: {ON_ACCENT}; font-weight: 600; }}
QPushButton#Primary:hover, QFrame#Panel QPushButton#Primary:hover, QFrame#TopBar QPushButton#Primary:hover {{ background: {ACCENT_HOVER}; border-color: {ACCENT_HOVER}; }}
QPushButton#Primary:pressed, QFrame#Panel QPushButton#Primary:pressed, QFrame#TopBar QPushButton#Primary:pressed {{ background: {ACCENT_PRESSED}; border-color: {ACCENT_PRESSED}; }}
QPushButton#Primary:disabled, QFrame#Panel QPushButton#Primary:disabled, QFrame#TopBar QPushButton#Primary:disabled {{ background: {PANEL_ALT}; border-color: {BORDER}; color: {MUTED}; }}
QFrame#TopBar QPushButton {{ padding: 7px 12px; }}
QPushButton#Nav, QFrame#TopBar QPushButton#Nav {{ background: transparent; border: 1px solid {BORDER_STRONG}; color: {TEXT}; font-weight: 600; padding: 7px 12px; }}
QPushButton#Nav:hover, QFrame#TopBar QPushButton#Nav:hover {{ background: {PANEL_HOVER}; border-color: {ACCENT}; }}
QPushButton#Nav:pressed, QFrame#TopBar QPushButton#Nav:pressed {{ background: {BORDER}; }}
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
    selection-color: {ON_ACCENT};
}}
QLineEdit:hover {{ border-color: {BORDER_STRONG}; }}
QLineEdit:focus {{ border-color: {ACCENT}; }}
QTableWidget {{
    background: transparent;
    alternate-background-color: rgba(245, 211, 31, 0.025);
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


def brand_mark(size: int):
    """QLabel showing the PSAU logo badge; falls back to the letter "P" if the image is missing."""
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QPixmap
    from PySide6.QtWidgets import QLabel

    from ..config import bundle_dir

    mark = QLabel()
    mark.setObjectName("BrandMark")
    mark.setFixedSize(size, size)
    mark.setAlignment(Qt.AlignmentFlag.AlignCenter)
    pix = QPixmap(str(bundle_dir() / "platescanner" / "assets" / "logo_badge.png"))
    if pix.isNull():
        mark.setText("P")
    else:
        mark.setPixmap(pix.scaled(size, size, Qt.AspectRatioMode.KeepAspectRatio,
                                  Qt.TransformationMode.SmoothTransformation))
    return mark
