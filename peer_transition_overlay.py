from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Peer review transition overlay")
    parser.add_argument("--window-x", type=int, required=True)
    parser.add_argument("--window-y", type=int, required=True)
    parser.add_argument("--window-width", type=int, required=True)
    parser.add_argument("--window-height", type=int, required=True)
    parser.add_argument("--axial-x", type=int, required=True)
    parser.add_argument("--axial-y", type=int, required=True)
    parser.add_argument("--axial-width", type=int, required=True)
    parser.add_argument("--axial-height", type=int, required=True)
    parser.add_argument("--screenshot", type=str, default=None)
    parser.add_argument("--movie", type=str, default=None)
    parser.add_argument("--parent-pid", type=int, default=0)
    return parser.parse_args(argv)


def rgba_to_pixmap(frame_rgba: np.ndarray) -> QtGui.QPixmap:
    frame = np.ascontiguousarray(frame_rgba, dtype=np.uint8)
    height, width = frame.shape[:2]
    image = QtGui.QImage(
        frame.data,
        width,
        height,
        frame.strides[0],
        QtGui.QImage.Format.Format_RGBA8888,
    ).copy()
    return QtGui.QPixmap.fromImage(image)


class TransitionOverlayWindow(QtWidgets.QWidget):
    def __init__(
        self,
        *,
        window_rect: QtCore.QRect,
        axial_rect: QtCore.QRect,
        screenshot_path: Optional[Path],
        movie_path: Optional[Path],
        parent_pid: int,
    ) -> None:
        super().__init__(
            None,
            QtCore.Qt.WindowType.Tool
            | QtCore.Qt.WindowType.FramelessWindowHint
            | QtCore.Qt.WindowType.WindowStaysOnTopHint,
        )
        self.parent_pid = parent_pid
        self.axial_rect = QtCore.QRect(axial_rect)
        self.screenshot_source_pixmap: Optional[QtGui.QPixmap] = None
        self.screenshot_display_pixmap: Optional[QtGui.QPixmap] = None
        self.movie_source_pixmaps: List[QtGui.QPixmap] = []
        self.movie_display_pixmaps: List[QtGui.QPixmap] = []
        self.movie_frame_index = 0
        self.movie_interval_ms = 120

        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self.setGeometry(window_rect)
        self.setStyleSheet("background-color: black;")

        self.screenshot_label = QtWidgets.QLabel(self)
        self.screenshot_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.screenshot_label.setStyleSheet("QLabel { background-color: black; }")
        self.screenshot_label.setGeometry(self.rect())

        self.movie_label = QtWidgets.QLabel(self)
        self.movie_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.movie_label.setStyleSheet("QLabel { background-color: black; }")
        self.movie_label.setGeometry(self.axial_rect)

        if screenshot_path is not None and screenshot_path.exists():
            pixmap = QtGui.QPixmap(str(screenshot_path))
            if not pixmap.isNull():
                self.screenshot_source_pixmap = pixmap

        if movie_path is not None and movie_path.exists():
            with np.load(movie_path, allow_pickle=False) as data:
                frames = data["frames"]
                if "interval_ms" in data:
                    interval_raw = np.asarray(data["interval_ms"]).reshape(-1)
                    if interval_raw.size > 0:
                        self.movie_interval_ms = max(60, int(interval_raw[0]))
            self.movie_source_pixmaps = [rgba_to_pixmap(frame_rgba) for frame_rgba in frames]

        self.parent_watch_timer = QtCore.QTimer(self)
        self.parent_watch_timer.setInterval(1000)
        self.parent_watch_timer.timeout.connect(self._check_parent_process)
        if self.parent_pid > 0:
            self.parent_watch_timer.start()

        self.movie_timer = QtCore.QTimer(self)
        self.movie_timer.setSingleShot(False)
        self.movie_timer.timeout.connect(self._advance_movie_frame)

        self._rescale_assets()
        if self.movie_display_pixmaps:
            self.movie_timer.setInterval(self.movie_interval_ms)
            self.movie_timer.start()

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

        if self.movie_source_pixmaps:
            target_size = self.axial_rect.size()
            self.movie_display_pixmaps = [
                pixmap.scaled(
                    target_size,
                    QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                    QtCore.Qt.TransformationMode.SmoothTransformation,
                )
                for pixmap in self.movie_source_pixmaps
            ]
            self.movie_frame_index = 0
            self.movie_label.setPixmap(self.movie_display_pixmaps[0])
        else:
            self.movie_display_pixmaps = []
            self.movie_label.clear()

    def _advance_movie_frame(self) -> None:
        if not self.movie_display_pixmaps:
            return
        self.movie_frame_index = (self.movie_frame_index + 1) % len(self.movie_display_pixmaps)
        self.movie_label.setPixmap(self.movie_display_pixmaps[self.movie_frame_index])

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
        self.movie_label.setGeometry(self.axial_rect)
        self._rescale_assets()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self.movie_timer.stop()
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
        axial_rect=QtCore.QRect(
            args.axial_x,
            args.axial_y,
            args.axial_width,
            args.axial_height,
        ),
        screenshot_path=Path(args.screenshot) if args.screenshot else None,
        movie_path=Path(args.movie) if args.movie else None,
        parent_pid=args.parent_pid,
    )
    window.show()
    window.raise_()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
