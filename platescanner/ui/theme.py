"""PSAU-branded console theme for the gate screen: forest green and gold, in a dark and a light variant.

The palette comes from the university logo (deep green strokes, gold ring). Call apply("dark" | "light")
once at start-up, before the stylesheet is built; the other modules read the colours as theme.<NAME>.
"""

MONO_FONT = '"Cascadia Mono", Consolas, monospace'

_DARK = dict(
    BG="#0d1311", PANEL="#141c19", PANEL_ALT="#1b2521", PANEL_HOVER="#232f2a",
    BORDER="#27332e", BORDER_STRONG="#364640",
    TEXT="#f4f1e6", MUTED="#9aa8a1", FAINT="#6f7d76",
    ACCENT="#e9c82a", ACCENT_HOVER="#f3d54b", ACCENT_PRESSED="#d9b80f", ON_ACCENT="#10402f",
    ACCENT_TEXT="#f3d54b", FOCUS="#e9c82a", TAB_LINE="#e9c82a", ALT_ROW="rgba(255, 255, 255, 0.02)",
    VIDEO_BG="#050806",
    RED="#ef4444", GREEN="#3cc57f", AMBER="#f59e0b",
    TINT_VIOLATION="#301a1c", TINT_CLEAR="#152b20", TINT_NOT_REGISTERED="#2f2812", TINT_NO_PLATE="#1b2521",
    # top bar
    TB_BG="#141c19", TB_EDGE="1px solid #27332e", TB_TEXT="#f4f1e6", TB_MUTED="#9aa8a1",
    TB_BTN="#1b2521", TB_BTN_HOVER="#232f2a", TB_BTN_BORDER="#364640",
    PLATE_BG="#f7f5ec", PLATE_INK="#10402f",
)

_LIGHT = dict(
    BG="#f2f0e6", PANEL="#ffffff", PANEL_ALT="#f7f5ee", PANEL_HOVER="#ebe8da",
    BORDER="#dcd8c6", BORDER_STRONG="#c2bda5",
    TEXT="#12261d", MUTED="#55695e", FAINT="#85948b",
    ACCENT="#f5d31f", ACCENT_HOVER="#ffe24d", ACCENT_PRESSED="#d9b80f", ON_ACCENT="#10402f",
    ACCENT_TEXT="#114232", FOCUS="#1a5c45", TAB_LINE="#114232", ALT_ROW="rgba(17, 66, 50, 0.035)",
    VIDEO_BG="#06120d",
    RED="#d93636", GREEN="#188a4a", AMBER="#b86e00",
    TINT_VIOLATION="#fde6e6", TINT_CLEAR="#e0f3e8", TINT_NOT_REGISTERED="#fff0d1", TINT_NO_PLATE="#efede2",
    TB_BG="#114232", TB_EDGE="3px solid #f5d31f", TB_TEXT="#ffffff", TB_MUTED="rgba(255, 255, 255, 0.72)",
    TB_BTN="rgba(255, 255, 255, 0.10)", TB_BTN_HOVER="rgba(255, 255, 255, 0.20)",
    TB_BTN_BORDER="rgba(255, 255, 255, 0.28)",
    PLATE_BG="#fffdf4", PLATE_INK="#114232",
)

PALETTES = {"dark": _DARK, "light": _LIGHT}
MODE = "dark"
STYLESHEET = ""

RESULT_LABELS = {"violation": "VIOLATION", "clear": "NO VIOLATION", "not_registered": "NOT REGISTERED",
                 "no_plate": "NO PLATE READ"}
RESULT_ICONS = {"violation": "⛔", "clear": "✔", "not_registered": "?", "no_plate": "–"}


def dot(color: str, text: str) -> str:
    """Rich-text status line with a coloured dot in front."""
    return f'<span style="color:{color}; font-size:11pt;">●</span>&nbsp; {text}'


def apply(mode: str = "dark") -> str:
    """Switch the module-level colours to the dark or light palette and rebuild STYLESHEET."""
    global MODE, STYLESHEET, MONO, RESULT_COLORS, RESULT_TINTS
    mode = mode if mode in PALETTES else "dark"
    globals().update(PALETTES[mode])
    MODE = mode
    MONO = MONO_FONT
    RESULT_COLORS = {"violation": RED, "clear": GREEN, "not_registered": AMBER, "no_plate": MUTED}
    RESULT_TINTS = {"violation": TINT_VIOLATION, "clear": TINT_CLEAR, "not_registered": TINT_NOT_REGISTERED,
                    "no_plate": TINT_NO_PLATE}
    STYLESHEET = _build_stylesheet()
    return mode


def _build_stylesheet() -> str:
    g = globals()
    BG, PANEL, PANEL_ALT, PANEL_HOVER = g["BG"], g["PANEL"], g["PANEL_ALT"], g["PANEL_HOVER"]
    BORDER = g["BORDER"]
    BORDER_STRONG = g["BORDER_STRONG"]
    TEXT = g["TEXT"]
    MUTED = g["MUTED"]
    FAINT = g["FAINT"]
    ACCENT = g["ACCENT"]
    ACCENT_HOVER = g["ACCENT_HOVER"]
    ACCENT_PRESSED = g["ACCENT_PRESSED"]
    ON_ACCENT = g["ON_ACCENT"]
    ACCENT_TEXT = g["ACCENT_TEXT"]
    FOCUS = g["FOCUS"]
    TAB_LINE = g["TAB_LINE"]
    ALT_ROW = g["ALT_ROW"]
    TB_BG = g["TB_BG"]
    TB_EDGE = g["TB_EDGE"]
    TB_TEXT = g["TB_TEXT"]
    TB_MUTED = g["TB_MUTED"]
    TB_BTN = g["TB_BTN"]
    TB_BTN_HOVER = g["TB_BTN_HOVER"]
    TB_BTN_BORDER = g["TB_BTN_BORDER"]
    PLATE_BG = g["PLATE_BG"]
    PLATE_INK = g["PLATE_INK"]
    MONO = g["MONO"]
    return f"""
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
    background: {PLATE_BG};
    color: {PLATE_INK};
    border: 2px solid {PLATE_INK};
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
    background: {TB_BG};
    border-bottom: {TB_EDGE};
}}
QFrame#TopBar QWidget {{ background: transparent; }}
QLabel#AppTitle {{ font-size: 12.5pt; font-weight: 700; }}
QLabel#AppSubtitle {{ color: {MUTED}; font-size: 8.5pt; }}
QFrame#TopBar QLabel {{ color: {TB_TEXT}; }}
QFrame#TopBar QLabel#AppSubtitle, QFrame#TopBar QLabel#ClockDate {{ color: {TB_MUTED}; }}
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
QFrame#TopBar QPushButton {{ padding: 7px 12px; background: {TB_BTN}; border-color: {TB_BTN_BORDER}; color: {TB_TEXT}; }}
QFrame#TopBar QPushButton:hover {{ background: {TB_BTN_HOVER}; }}
QFrame#TopBar QPushButton#Primary {{ background: {ACCENT}; border-color: {ACCENT}; color: {ON_ACCENT}; }}
QFrame#TopBar QPushButton#Primary:hover {{ background: {ACCENT_HOVER}; border-color: {ACCENT_HOVER}; }}
QPushButton#Nav, QFrame#TopBar QPushButton#Nav {{ background: transparent; border: 1px solid {TB_BTN_BORDER}; color: {TB_TEXT}; font-weight: 600; padding: 7px 12px; }}
QPushButton#Nav:hover, QFrame#TopBar QPushButton#Nav:hover {{ background: {TB_BTN_HOVER}; border-color: {ACCENT}; }}
QPushButton#Nav:pressed, QFrame#TopBar QPushButton#Nav:pressed {{ background: {BORDER}; }}
QPushButton#Link, QFrame#Panel QPushButton#Link, QFrame#TopBar QPushButton#Link {{
    background: transparent;
    border: none;
    color: {MUTED};
    padding: 6px 4px;
}}
QPushButton#Link:hover, QFrame#Panel QPushButton#Link:hover, QFrame#TopBar QPushButton#Link:hover {{ color: {TEXT}; text-decoration: underline; }}
QToolButton, QFrame#Panel QToolButton {{
    background: {PANEL_ALT};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 4px 10px;
}}
QToolButton:hover, QFrame#Panel QToolButton:hover, QFrame#TopBar QToolButton:hover {{ background: {PANEL_HOVER}; border-color: {BORDER_STRONG}; }}
QToolButton:pressed, QFrame#Panel QToolButton:pressed, QFrame#TopBar QToolButton:pressed {{ background: {BORDER}; }}
QFrame#TopBar QToolButton {{ background: {TB_BTN}; border: 1px solid {TB_BTN_BORDER}; border-radius: 6px; padding: 4px 10px; color: {TB_TEXT}; }}
QFrame#TopBar QToolButton:hover {{ background: {TB_BTN_HOVER}; }}
QToolButton::menu-indicator {{ image: none; width: 0; }}
QMenu, QFrame#Panel QMenu, QFrame#TopBar QMenu {{
    background: {PANEL_ALT};
    border: 1px solid {BORDER_STRONG};
    border-radius: 8px;
    padding: 5px;
}}
QMenu::item {{ background: transparent; padding: 6px 26px 6px 12px; border-radius: 5px; }}
QMenu::item:selected {{ background: {PANEL_HOVER}; }}
QMenu::separator {{ height: 1px; background: {BORDER_STRONG}; margin: 6px 8px; }}
QMenu::indicator {{ width: 0; }}
QMenu::item:checked {{ color: {ACCENT_TEXT}; font-weight: 600; }}
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
QLineEdit:focus {{ border-color: {FOCUS}; }}
QTableWidget {{
    background: transparent;
    alternate-background-color: {ALT_ROW};
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
QTabBar::tab:selected {{ color: {TEXT}; border-bottom: 2px solid {TAB_LINE}; }}
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


apply("dark")


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
