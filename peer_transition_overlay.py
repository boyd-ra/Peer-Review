from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional

from PySide6 import QtCore, QtGui, QtWidgets

from peer_cache import load_review_bundle_screenshot_bytes


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Peer review transition overlay")
    parser.add_argument("--window-x", type=int, required=True)
    parser.add_argument("--window-y", type=int, required=True)
    parser.add_argument("--window-width", type=int, required=True)
    parser.add_argument("--window-height", type=int, required=True)
    parser.add_argument("--bundle", type=str, default=None)
    parser.add_argument("--screenshot", type=str, default=None)
    parser.add_argument("--parent-pid", type=int, default=0)
    return parser.parse_args(argv)


class TransitionOverlayWindow(QtWidgets.QWidget):
    def __init__(
        self,
        *,
        window_rect: QtCore.QRect,
        bundle_path: Optional[Path],
        screenshot_path: Optional[Path],
        parent_pid: int,
    ) -> None:
        super().__init__(
            None,
            QtCore.Qt.WindowType.Tool
            | QtCore.Qt.WindowType.FramelessWindowHint
            | QtCore.Qt.WindowType.WindowStaysOnTopHint,
        )
        self.parent_pid = parent_pid
        self.screenshot_source_pixmap: Optional[QtGui.QPixmap] = None
        self.screenshot_display_pixmap: Optional[QtGui.QPixmap] = None

        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self.setGeometry(window_rect)
        self.setStyleSheet("background-color: black;")

        self.screenshot_label = QtWidgets.QLabel(self)
        self.screenshot_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.screenshot_label.setStyleSheet("QLabel { background-color: black; }")
        self.screenshot_label.setGeometry(self.rect())

        screenshot_bytes = b""
        if bundle_path is not None and bundle_path.exists():
            screenshot_bytes = load_review_bundle_screenshot_bytes(bundle_path) or b""
        if not screenshot_bytes and screenshot_path is not None and screenshot_path.exists():
            try:
                screenshot_bytes = screenshot_path.read_bytes()
            except OSError:
                screenshot_bytes = b""
        pixmap = QtGui.QPixmap()
        if screenshot_bytes and pixmap.loadFromData(screenshot_bytes):
            self.screenshot_source_pixmap = pixmap

        self.parent_watch_timer = QtCore.QTimer(self)
        self.parent_watch_timer.setInterval(1000)
        self.parent_watch_timer.timeout.connect(self._check_parent_process)
        if self.parent_pid > 0:
            self.parent_watch_timer.start()

        self._rescale_assets()

    def _rescale_assets(self) -> None:
        if self.screenshot_source_pixmap is not None and not self.screenshot_source_pixmap.isNull():
            self.screenshot_display_pixmap = self.screenshot_source_pixmap.scaled(
                self.size(),
                QtCore.Qt.AspectRatioMode.IgnoreAspectRatio,
                QtCore.Qt.TransformationMode.SmoothTransformation,
            )
            self.screenshot_label.setPixmap(self.screenshot_display_pixmap)
        else:
            self.screenshot_label.clear()

    def _check_parent_process(self) -> None:
        if self.parent_pid <= 0:
            return
        try:
            os.kill(self.parent_pid, 0)
        except OSError:
            self.close()

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self.screenshot_label.setGeometry(self.rect())
        self._rescale_assets()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self.parent_watch_timer.stop()
        super().closeEvent(event)


def main(argv: List[str]) -> int:
    args = parse_args(argv)
    app = QtWidgets.QApplication(sys.argv[:1])
    window = TransitionOverlayWindow(
        window_rect=QtCore.QRect(
            args.window_x,
            args.window_y,
            args.window_width,
            args.window_height,
        ),
        bundle_path=Path(args.bundle) if args.bundle else None,
        screenshot_path=Path(args.screenshot) if args.screenshot else None,
        parent_pid=args.parent_pid,
    )
    window.show()
    window.raise_()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
