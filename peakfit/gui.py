"""Minimal PySide6 GUI scaffold for interactive peak fitting.

This module provides a basic desktop UI using PySide6 with an embedded
matplotlib canvas to render data+fit and residuals, a parameter table,
and buttons to load/save files and run/cancel fits.

This is an initial scaffold that reuses `peakfit.io` and `peakfit.fitting`.
Ensure `PySide6` and `matplotlib` are installed before running.
"""
from __future__ import annotations

import json
import os
from typing import Dict, Any

import numpy as np
import copy

try:
    from PySide6 import QtCore, QtWidgets
    from PySide6.QtCore import QLocale
    from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                                   QHBoxLayout, QPushButton, QFileDialog, QTableWidget,
                                   QTableWidgetItem, QCheckBox, QProgressBar, QLabel,
                                   QHeaderView, QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox,
                                   QPlainTextEdit)
    from PySide6.QtGui import QDoubleValidator, QColor, QPalette
except Exception as e:
    raise ImportError("PySide6 is required to run the GUI: " + str(e))

try:
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar
except Exception as e:
    raise ImportError("matplotlib (Qt backend) is required to run the GUI: " + str(e))

from .io import load_data, load_param_config, normalize_number_string
from .fitting import fit_with_lmfit, fit_with_scipy, _build_free_params, _params_from_vector
from .model import model as compute_model, model_components, resolve_accelerator_mode

# Built-in default parameter sets used to initialize the GUI when the user
# selects a kernel. These mirror the values used in the test fixtures /
# `params_gauss.json` (embedded here so the GUI can self-initialize).
GAUSS_DEFAULT_PARAMS = {
    'C':   {'value': 171400.0, 'vary': False, 'min': None, 'max': None},
    'D':   {'value': 100000.0, 'vary': False, 'min': None, 'max': None},
    'b':   {'value': 0.0, 'vary': True, 'min': None, 'max': None},
    'g0':  {'value': 4.5, 'vary': True, 'min': 1e-6, 'max': 10.0},
    'q':   {'value': 2.0, 'vary': True, 'min': -1000.0, 'max': 1000.0},
    'a':   {'value': 0.5431, 'vary': False, 'min': 0.1, 'max': 0.6},
    'L':   {'value': 10.0, 'vary': True, 'min': 1.0, 'max': 100.0},
    'alpha': {'value': 1.0, 'vary': False, 'min': 0.0, 'max': None},
    'N':   {'value': 1.0, 'vary': True, 'min': 0.0, 'max': 1e9},
    'y0':  {'value': 0.0, 'vary': True, 'min': -1e9, 'max': 1e9},
}

BESSEL_DEFAULT_PARAMS = {
    'C':   {'value': 171400.0, 'vary': False, 'min': None, 'max': None},
    'D':   {'value': 100000.0, 'vary': False, 'min': None, 'max': None},
    'b':   {'value': 0.0, 'vary': True, 'min': None, 'max': None},
    'g0':  {'value': 4.5, 'vary': True, 'min': 1e-6, 'max': 10.0},
    'q':   {'value': 2.0, 'vary': True, 'min': -1000.0, 'max': 1000.0},
    'a':   {'value': 0.5431, 'vary': False, 'min': 0.1, 'max': 0.6},
    'L':   {'value': 10.0, 'vary': True, 'min': 1.0, 'max': 100.0},
    'N':   {'value': 1.0, 'vary': True, 'min': 0.0, 'max': 1e9},
    'y0':  {'value': 0.0, 'vary': True, 'min': -1e9, 'max': 1e9},
}


def sci_format(x, decimals: int = 4):
    """Return scientific 'e' notation for large/small numbers, otherwise fixed format.

    Use `e` formatting when |x|>100 or |x|<0.01; otherwise use fixed-point with
    `decimals` digits after the decimal point.
    """
    try:
        if x is None:
            return ''
        xv = float(x)
    except Exception:
        return str(x)
    if not np.isfinite(xv):
        return str(x)
    if xv == 0:
        return f"0.{''.join(['0']*decimals)}e+00"
    if abs(xv) > 100 or abs(xv) < 0.01:
        return f"{xv:.{decimals}e}"
    return f"{xv:.{decimals}f}"


class ParamLineEdit(QLineEdit):
    """Numeric line edit tuned for quick overwrite in the parameter table."""

    def focusInEvent(self, event):
        super().focusInEvent(event)
        QtCore.QTimer.singleShot(0, self.selectAll)

    def mousePressEvent(self, event):
        had_focus = self.hasFocus()
        super().mousePressEvent(event)
        if not had_focus:
            QtCore.QTimer.singleShot(0, self.selectAll)

    def keyPressEvent(self, event):
        # If the cursor is at the end, Delete is usually a no-op; select all
        # so a single Delete press clears the field for rewriting.
        if event.key() == QtCore.Qt.Key_Delete and not self.hasSelectedText():
            if self.cursorPosition() >= len(self.text()):
                self.selectAll()
        super().keyPressEvent(event)


class FitWorker(QtCore.QObject):
    progress = QtCore.Signal(int)
    finished = QtCore.Signal(object)
    error = QtCore.Signal(str)

    def __init__(self, iw, y, param_config: Dict[str, Any], backend: str = 'lmfit', integrator_opts=None, minimizer_opts=None):
        super().__init__()
        self.iw = np.asarray(iw, dtype=float)
        self.y = np.asarray(y, dtype=float)
        self.param_config = param_config
        self.backend = backend
        self._cancel = False
        self.integrator_opts = integrator_opts or {}
        self.minimizer_opts = minimizer_opts or {}

    def cancel(self):
        self._cancel = True

    def run(self):
        def progress_cb(call_n, pvec=None):
            self.progress.emit(call_n)
            return not self._cancel

        accel_used = resolve_accelerator_mode(
            self.integrator_opts.get('integrator', 'grid'),
            self.integrator_opts.get('accelerator', 'auto'),
        )

        def _refine_param_config(base_cfg, fitted_vals):
            cfg = copy.deepcopy(base_cfg) if isinstance(base_cfg, dict) else {}
            if not isinstance(cfg, dict):
                cfg = {}
            if isinstance(fitted_vals, dict):
                for name, val in fitted_vals.items():
                    if name in cfg and isinstance(cfg.get(name), dict):
                        try:
                            cfg[name]['value'] = float(val)
                        except Exception:
                            cfg[name]['value'] = val
                    else:
                        try:
                            cfg[name] = {'value': float(val), 'vary': False}
                        except Exception:
                            cfg[name] = {'value': val, 'vary': False}
            return cfg

        def _score_fit(out_dict):
            if not isinstance(out_dict, dict):
                return (0, float('-inf'))
            converged = bool(out_dict.get('converged', False))
            try:
                r2_val = float(out_dict.get('r2', float('-inf')))
                if not np.isfinite(r2_val):
                    r2_val = float('-inf')
            except Exception:
                r2_val = float('-inf')
            return (1 if converged else 0, r2_val)

        def _maybe_numpy_refine(first_out):
            try:
                integrator = str(self.integrator_opts.get('integrator', 'grid')).strip().lower()
                selected = str(self.integrator_opts.get('accelerator', 'auto')).strip().lower()
                effective = resolve_accelerator_mode(integrator, selected)
                if integrator != 'grid' or effective != 'numba':
                    return first_out
                if self._cancel:
                    return first_out
                if not isinstance(first_out, dict):
                    return first_out
                fitted_first = first_out.get('fitted', None)
                if not isinstance(fitted_first, dict) or not fitted_first:
                    return first_out

                refine_cfg = _refine_param_config(self.param_config, fitted_first)
                refine_integrator_opts = dict(self.integrator_opts)
                refine_integrator_opts['accelerator'] = 'numpy'

                if self.backend == 'lmfit':
                    refine_minimizer_opts = dict(self.minimizer_opts or {})
                    try:
                        cur = int(refine_minimizer_opts.get('max_nfev', 300))
                    except Exception:
                        cur = 300
                    refine_minimizer_opts['max_nfev'] = max(80, min(cur, 500))
                    fitted2, errs2, result2, y_model2, r22, conv2, msg2, nfev2 = fit_with_lmfit(
                        self.iw,
                        self.y,
                        refine_cfg,
                        integrator_opts=refine_integrator_opts,
                        minimizer_opts=refine_minimizer_opts,
                        progress_callback=progress_cb,
                    )
                    out2 = dict(
                        fitted=fitted2,
                        errs=errs2,
                        result=result2,
                        y_model=y_model2,
                        r2=r22,
                        converged=conv2,
                        message=msg2,
                        nfev=nfev2,
                        accelerator_used='numba->numpy-refine',
                    )
                    try:
                        pcov2 = None
                        try:
                            pcov2 = getattr(result2, 'covar', None)
                        except Exception:
                            pcov2 = None
                        if pcov2 is None:
                            try:
                                pcov2 = getattr(result2, 'covariance', None)
                            except Exception:
                                pcov2 = None
                        if pcov2 is not None:
                            out2['pcov'] = pcov2
                    except Exception:
                        pass
                else:
                    refine_curvefit_opts = dict(self.minimizer_opts or {})
                    try:
                        cur = int(refine_curvefit_opts.get('maxfev', 300))
                    except Exception:
                        cur = 300
                    refine_curvefit_opts['maxfev'] = max(80, min(cur, 500))
                    params2, errs2, popt2, pcov2, y_model2, r22, conv2, msg2, nfev2 = fit_with_scipy(
                        self.iw,
                        self.y,
                        refine_cfg,
                        integrator_opts=refine_integrator_opts,
                        curvefit_opts=refine_curvefit_opts,
                        progress_callback=progress_cb,
                    )
                    out2 = dict(
                        fitted=params2,
                        errs=errs2,
                        popt=popt2,
                        pcov=pcov2,
                        y_model=y_model2,
                        r2=r22,
                        converged=conv2,
                        message=msg2,
                        nfev=nfev2,
                        accelerator_used='numba->numpy-refine',
                    )

                if _score_fit(out2) >= _score_fit(first_out):
                    return out2
                return first_out
            except Exception:
                return first_out

        try:
            if self.backend == 'lmfit':
                fitted, errs, result, y_model, r2, converged, message, nfev = fit_with_lmfit(
                    self.iw, self.y, self.param_config,
                    integrator_opts=self.integrator_opts, minimizer_opts=self.minimizer_opts,
                    progress_callback=progress_cb)
                out = dict(fitted=fitted, errs=errs, result=result, y_model=y_model, r2=r2, converged=converged, message=message, nfev=nfev, accelerator_used=accel_used)
                # If the lmfit result contains a covariance matrix attach it
                # explicitly in the out dict so the GUI can read it reliably.
                try:
                    pcov = None
                    try:
                        pcov = getattr(result, 'covar', None)
                    except Exception:
                        pcov = None
                    if pcov is None:
                        try:
                            pcov = getattr(result, 'covariance', None)
                        except Exception:
                            pcov = None
                    if pcov is not None:
                        out['pcov'] = pcov
                except Exception:
                    pass
            else:
                params_full, errs_full, popt, pcov, y_model, r2, converged, message, nfev = fit_with_scipy(
                    self.iw, self.y, self.param_config,
                    integrator_opts=self.integrator_opts, curvefit_opts=self.minimizer_opts,
                    progress_callback=progress_cb)
                out = dict(fitted=params_full, errs=errs_full, popt=popt, pcov=pcov, y_model=y_model, r2=r2, converged=converged, message=message, nfev=nfev, accelerator_used=accel_used)
            out = _maybe_numpy_refine(out)
            self.finished.emit(out)
        except Exception as e:
            self.error.emit(str(e))


class MainWindow(QMainWindow):
    def __init__(self, assets_dir: str | None = None):
        super().__init__()
        self.setWindowTitle('PeakFit GUI (scaffold)')
        self.resize(1100, 700)

        self.assets_dir = assets_dir or os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'assets'))

        self.iw = None
        self.y = None
        self.param_config: Dict[str, Dict[str, Any]] = {}
        self.fit_thread: QtCore.QThread | None = None
        self.fit_worker: FitWorker | None = None
        self.last_y_model = None
        self._legend_artist_map = {}
        self._hover_x = None
        self._hover_y = None
        self._hover_annotation = None
        self._hover_marker = None
        self._hover_last_idx = None
        self._main_plot_cache = {}
        self._resid_plot_cache = {}
        self._mpl_motion_cid = None
        self._mpl_pick_cid = None
        self._mpl_leave_cid = None

        self._init_ui()
        self._apply_app_style()
        try:
            self._configure_accelerator_menu()
            self._update_accel_indicator()
        except Exception:
            pass

    def _apply_app_style(self):
        """Apply a lightweight modern theme for readability and contrast."""
        try:
            app = QApplication.instance()
            if app is not None:
                try:
                    app.setStyle('Fusion')
                except Exception:
                    pass
                try:
                    pal = QPalette()
                    pal.setColor(QPalette.Window, QColor('#f4f7fb'))
                    pal.setColor(QPalette.WindowText, QColor('#1d2a38'))
                    pal.setColor(QPalette.Base, QColor('#ffffff'))
                    pal.setColor(QPalette.AlternateBase, QColor('#f8fbff'))
                    pal.setColor(QPalette.ToolTipBase, QColor('#ffffff'))
                    pal.setColor(QPalette.ToolTipText, QColor('#1d2a38'))
                    pal.setColor(QPalette.Text, QColor('#1d2a38'))
                    pal.setColor(QPalette.Button, QColor('#eef3f9'))
                    pal.setColor(QPalette.ButtonText, QColor('#1d2a38'))
                    pal.setColor(QPalette.BrightText, QColor('#ffffff'))
                    pal.setColor(QPalette.Highlight, QColor('#1f4e79'))
                    pal.setColor(QPalette.HighlightedText, QColor('#ffffff'))
                    pal.setColor(QPalette.Disabled, QPalette.Text, QColor('#6f8196'))
                    pal.setColor(QPalette.Disabled, QPalette.WindowText, QColor('#6f8196'))
                    pal.setColor(QPalette.Disabled, QPalette.ButtonText, QColor('#6f8196'))
                    app.setPalette(pal)
                except Exception:
                    pass

            self.setStyleSheet(
                """
                QMainWindow { background: #f4f7fb; }
                QWidget {
                    font-family: 'Segoe UI Variable', 'Segoe UI', 'Candara', sans-serif;
                    font-size: 10pt;
                    color: #1d2a38;
                }
                QLabel { color: #1d2a38; }
                QCheckBox {
                    color: #1d2a38;
                    spacing: 6px;
                    font-weight: 700;
                }
                /* Larger, more visible checkbox indicator */
                QCheckBox::indicator {
                    width: 18px;
                    height: 18px;
                    border: 1px solid #7a8da3;
                    border-radius: 4px;
                    background: #ffffff;
                }
                QCheckBox::indicator:checked {
                    background-color: #1f4e79;
                    border: 1px solid #153a60;
                }
                QCheckBox::indicator:hover {
                    border: 1px solid #1f4e79;
                }
                QPushButton {
                    background-color: #1f4e79;
                    color: #ffffff;
                    border: 1px solid #183a5a;
                    border-radius: 6px;
                    padding: 4px 10px;
                }
                QPushButton:hover { background-color: #2b6396; }
                QPushButton:disabled {
                    background-color: #9aa8b8;
                    color: #f7f9fc;
                    border-color: #8c99a8;
                }
                QLineEdit, QComboBox, QPlainTextEdit {
                    background: #ffffff;
                    color: #1d2a38;
                    border: 1px solid #c7d2df;
                    border-radius: 5px;
                    padding: 2px 6px;
                    selection-background-color: #1f4e79;
                    selection-color: #ffffff;
                }
                QSpinBox, QDoubleSpinBox {
                    background: #ffffff;
                    color: #1d2a38;
                    border: 1px solid #c7d2df;
                    border-radius: 5px;
                    padding: 1px 22px 1px 6px;
                    selection-background-color: #1f4e79;
                    selection-color: #ffffff;
                }
                QSpinBox::up-button, QDoubleSpinBox::up-button {
                    subcontrol-origin: border;
                    subcontrol-position: top right;
                    width: 18px;
                    border-left: 1px solid #c7d2df;
                    border-top-right-radius: 5px;
                }
                QSpinBox::down-button, QDoubleSpinBox::down-button {
                    subcontrol-origin: border;
                    subcontrol-position: bottom right;
                    width: 18px;
                    border-left: 1px solid #c7d2df;
                    border-bottom-right-radius: 5px;
                }
                QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {
                    image: none;
                    width: 0px;
                    height: 0px;
                    border-left: 4px solid transparent;
                    border-right: 4px solid transparent;
                    border-bottom: 6px solid #1d2a38;
                }
                QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {
                    image: none;
                    width: 0px;
                    height: 0px;
                    border-left: 4px solid transparent;
                    border-right: 4px solid transparent;
                    border-top: 6px solid #1d2a38;
                }
                QTableWidget {
                    background: #ffffff;
                    color: #1d2a38;
                    alternate-background-color: #f8fbff;
                    border: 1px solid #c7d2df;
                    gridline-color: #dde5ef;
                    selection-background-color: #1f4e79;
                    selection-color: #ffffff;
                }
                QTableWidget::item {
                    color: #1d2a38;
                    background: #ffffff;
                }
                QHeaderView::section {
                    background-color: #e8eef6;
                    color: #1d2a38;
                    border: 1px solid #d0d9e4;
                    padding: 4px;
                }
                QComboBox QAbstractItemView {
                    background: #ffffff;
                    color: #1d2a38;
                    selection-background-color: #1f4e79;
                    selection-color: #ffffff;
                    border: 1px solid #c7d2df;
                }
                QToolBar {
                    background: #eef3f9;
                    border: 1px solid #d6e0ec;
                    color: #1d2a38;
                    spacing: 4px;
                }
                QToolButton {
                    color: #1d2a38;
                    background: #eef3f9;
                    border: 1px solid transparent;
                    border-radius: 4px;
                    padding: 3px 6px;
                }
                QToolButton:hover {
                    background: #dce8f4;
                    border-color: #b9cbe0;
                }
                QProgressBar {
                    border: 1px solid #c7d2df;
                    border-radius: 6px;
                    color: #1d2a38;
                    text-align: center;
                    background: #eef3f9;
                }
                QProgressBar::chunk { background-color: #1f4e79; border-radius: 4px; }
                """
            )

            # Force readable matplotlib foreground/background regardless of OS theme.
            try:
                self.figure.set_facecolor('#ffffff')
                for ax in (getattr(self, 'ax_main', None), getattr(self, 'ax_resid', None)):
                    if ax is None:
                        continue
                    ax.set_facecolor('#ffffff')
                    ax.tick_params(colors='#1d2a38', labelcolor='#1d2a38')
                    ax.xaxis.label.set_color('#1d2a38')
                    ax.yaxis.label.set_color('#1d2a38')
                    for spine in ax.spines.values():
                        spine.set_color('#7a8da3')
                if getattr(self, 'canvas', None) is not None:
                    self.canvas.draw_idle()
            except Exception:
                pass
        except Exception:
            pass

    def _new_float_validator(self):
        v = QDoubleValidator()
        try:
            v.setNotation(QDoubleValidator.StandardNotation)
            v.setLocale(QLocale.c())
        except Exception:
            pass
        return v

    def _normalize_decimal_line_edit(self, edit: QLineEdit):
        try:
            t = edit.text()
            if ',' in t:
                edit.setText(t.replace(',', '.'))
        except Exception:
            pass

    def _selected_accelerator(self) -> str:
        try:
            if hasattr(self, 'accel_combo') and self.accel_combo is not None:
                txt = str(self.accel_combo.currentText()).strip().lower()
                if txt.startswith('numba'):
                    return 'numba'
                if txt.startswith('numpy'):
                    return 'numpy'
                return 'auto'
        except Exception:
            pass
        return 'auto'

    def _configure_accelerator_menu(self):
        try:
            if not hasattr(self, 'accel_combo') or self.accel_combo is None:
                return
            idx = self.accel_combo.findText('numba')
            numba_available = resolve_accelerator_mode('grid', 'numba') == 'numba'
            model = self.accel_combo.model()
            if idx >= 0 and model is not None:
                try:
                    item = model.item(idx)
                    if item is not None:
                        item.setEnabled(bool(numba_available))
                except Exception:
                    pass
            if (not numba_available) and self._selected_accelerator() == 'numba':
                self.accel_combo.setCurrentText('auto')
            tip = 'Acceleration mode for grid integrator: auto, numba, or numpy'
            if not numba_available:
                tip += ' (numba not installed in this environment)'
            self.accel_combo.setToolTip(tip)
        except Exception:
            pass

    def _update_accel_indicator(self):
        try:
            selected = self._selected_accelerator()
            integrator = self.integrator_combo.currentText() if hasattr(self, 'integrator_combo') else 'grid'
            effective = resolve_accelerator_mode(integrator, selected)
            integ = str(integrator).strip().lower()
            if integ == 'quad':
                self.accel_indicator.setText(f'Accelerator: {selected} -> {effective} (quad ignores accelerator)')
            elif selected == 'numba' and effective != 'numba':
                self.accel_indicator.setText(f'Accelerator: {selected} -> {effective} (numba unavailable)')
            else:
                self.accel_indicator.setText(f'Accelerator: {selected} -> {effective}')
        except Exception:
            pass

    def _effective_accelerator_label(self, integrator: str = None) -> str:
        try:
            integ = integrator if integrator is not None else (self.integrator_combo.currentText() if hasattr(self, 'integrator_combo') else 'grid')
            return resolve_accelerator_mode(integ, self._selected_accelerator())
        except Exception:
            return 'numpy'

    def _hide_hover_annotation(self):
        try:
            changed = False
            if self._hover_annotation is not None and self._hover_annotation.get_visible():
                self._hover_annotation.set_visible(False)
                changed = True
            if self._hover_marker is not None and self._hover_marker.get_visible():
                self._hover_marker.set_visible(False)
                changed = True
            if changed:
                self.canvas.draw_idle()
        except Exception:
            pass

    def _on_axes_leave(self, _event):
        self._hide_hover_annotation()

    def _on_canvas_motion(self, event):
        try:
            if event is None or event.inaxes != self.ax_main:
                self._hide_hover_annotation()
                return
            if self._hover_annotation is None or self._hover_x is None or self._hover_y is None:
                return
            if event.xdata is None or event.ydata is None:
                self._hide_hover_annotation()
                return

            hx = np.asarray(self._hover_x, dtype=float)
            hy = np.asarray(self._hover_y, dtype=float)
            if hx.size == 0 or hy.size == 0:
                self._hide_hover_annotation()
                return

            x_lo, x_hi = self.ax_main.get_xlim()
            y_lo, y_hi = self.ax_main.get_ylim()
            x_span = max(1e-12, abs(float(x_hi - x_lo)))
            y_span = max(1e-12, abs(float(y_hi - y_lo)))
            d2 = ((hx - float(event.xdata)) / x_span) ** 2 + ((hy - float(event.ydata)) / y_span) ** 2
            idx = int(np.argmin(d2))

            # Do not show a tooltip when the cursor is far from data.
            if float(d2[idx]) > 0.01:
                self._hide_hover_annotation()
                self._hover_last_idx = None
                return

            if self._hover_last_idx != idx or not self._hover_annotation.get_visible():
                x_val = float(hx[idx])
                y_val = float(hy[idx])
                self._hover_annotation.xy = (x_val, y_val)
                self._hover_annotation.set_text(f'x={x_val:.6g}\ny={y_val:.6g}')
                self._hover_annotation.set_visible(True)
                try:
                    if self._hover_marker is not None:
                        self._hover_marker.set_data([x_val], [y_val])
                        self._hover_marker.set_visible(True)
                except Exception:
                    pass
                self._hover_last_idx = idx
                self.canvas.draw_idle()
        except Exception:
            pass

    def _on_legend_pick(self, event):
        try:
            artist = getattr(event, 'artist', None)
            if artist is None:
                return
            if artist not in self._legend_artist_map:
                return
            target = self._legend_artist_map.get(artist)
            if target is None:
                return
            vis = not bool(target.get_visible())
            target.set_visible(vis)
            try:
                artist.set_alpha(1.0 if vis else 0.2)
            except Exception:
                pass
            self.canvas.draw_idle()
        except Exception:
            pass

    def _rescale_plot_y(self):
        """Rescale Y axes to data inside currently visible X windows."""
        try:
            x = np.asarray(self._main_plot_cache.get('x')) if self._main_plot_cache else None
            y_data = np.asarray(self._main_plot_cache.get('y_data')) if self._main_plot_cache else None
            y_model = self._main_plot_cache.get('y_model') if self._main_plot_cache else None
            if y_model is not None:
                y_model = np.asarray(y_model)
            if x is not None and y_data is not None and x.size > 0 and y_data.size == x.size:
                x_lo, x_hi = self.ax_main.get_xlim()
                if x_lo > x_hi:
                    x_lo, x_hi = x_hi, x_lo
                mask = (x >= float(x_lo)) & (x <= float(x_hi))
                if np.any(mask):
                    yvals = list(y_data[mask])
                    if y_model is not None and y_model.size == x.size:
                        yvals.extend(list(y_model[mask]))
                    yvals_arr = np.asarray(yvals, dtype=float)
                    yvals_arr = yvals_arr[np.isfinite(yvals_arr)]
                    if yvals_arr.size > 0:
                        ymin = float(np.min(yvals_arr))
                        ymax = float(np.max(yvals_arr))
                        if ymin == ymax:
                            ymin -= 1e-6
                            ymax += 1e-6
                        pad = max(1e-6, 0.05 * (ymax - ymin))
                        self.ax_main.set_ylim(ymin - pad, ymax + pad)

            xr = np.asarray(self._resid_plot_cache.get('x')) if self._resid_plot_cache else None
            resid = self._resid_plot_cache.get('resid') if self._resid_plot_cache else None
            if resid is not None:
                resid = np.asarray(resid)
            if xr is not None and resid is not None and xr.size > 0 and resid.size == xr.size:
                r_lo, r_hi = self.ax_resid.get_xlim()
                if r_lo > r_hi:
                    r_lo, r_hi = r_hi, r_lo
                rmask = (xr >= float(r_lo)) & (xr <= float(r_hi))
                if np.any(rmask):
                    vals = np.asarray(resid[rmask], dtype=float)
                    vals = vals[np.isfinite(vals)]
                    if vals.size > 0:
                        ymin = float(np.min(vals))
                        ymax = float(np.max(vals))
                        if ymin == ymax:
                            ymin -= 1e-6
                            ymax += 1e-6
                        pad = max(1e-6, 0.05 * (ymax - ymin))
                        self.ax_resid.set_ylim(ymin - pad, ymax + pad)

            self.canvas.draw_idle()
        except Exception:
            pass

    def _rescale_plot_all(self):
        """Reset both plots to full data extents."""
        try:
            x = np.asarray(self._main_plot_cache.get('x')) if self._main_plot_cache else None
            y_data = np.asarray(self._main_plot_cache.get('y_data')) if self._main_plot_cache else None
            y_model = self._main_plot_cache.get('y_model') if self._main_plot_cache else None
            if y_model is not None:
                y_model = np.asarray(y_model)

            if x is not None and y_data is not None and x.size > 0 and y_data.size == x.size:
                self.ax_main.set_xlim(float(np.min(x)), float(np.max(x)))
                yvals = list(y_data)
                if y_model is not None and y_model.size == x.size:
                    yvals.extend(list(y_model))
                yvals_arr = np.asarray(yvals, dtype=float)
                yvals_arr = yvals_arr[np.isfinite(yvals_arr)]
                if yvals_arr.size > 0:
                    ymin = float(np.min(yvals_arr))
                    ymax = float(np.max(yvals_arr))
                    if ymin == ymax:
                        ymin -= 1e-6
                        ymax += 1e-6
                    pad = max(1e-6, 0.05 * (ymax - ymin))
                    self.ax_main.set_ylim(ymin - pad, ymax + pad)

            xr = np.asarray(self._resid_plot_cache.get('x')) if self._resid_plot_cache else None
            resid = self._resid_plot_cache.get('resid') if self._resid_plot_cache else None
            if resid is not None:
                resid = np.asarray(resid)
            if xr is not None and xr.size > 0:
                self.ax_resid.set_xlim(float(np.min(xr)), float(np.max(xr)))
                if resid is not None and resid.size == xr.size:
                    vals = np.asarray(resid, dtype=float)
                    vals = vals[np.isfinite(vals)]
                    if vals.size > 0:
                        ymin = float(np.min(vals))
                        ymax = float(np.max(vals))
                        if ymin == ymax:
                            ymin -= 1e-6
                            ymax += 1e-6
                        pad = max(1e-6, 0.05 * (ymax - ymin))
                        self.ax_resid.set_ylim(ymin - pad, ymax + pad)

            self.canvas.draw_idle()
        except Exception:
            pass

    def _init_ui(self):
        # Use a QSplitter so the user can resize the control pane vs the plots.
        # Left pane: controls + parameter table
        left = QWidget()
        try:
            left.setMinimumWidth(80)
        except Exception:
            pass
        left_l = QVBoxLayout(left)
        try:
            left_l.setSpacing(6)
        except Exception:
            pass

        # Helper: buttons can expand, but not beyond ~2x their text width.
        def _shrink_btn(btn, pad: int = 18):
            try:
                fm = btn.fontMetrics()
                text_w = max(1, fm.horizontalAdvance(str(btn.text())))
                min_w = int(text_w + pad)
                max_w = int((2.0 * text_w) + pad)
                if max_w < min_w:
                    max_w = min_w
                btn.setMinimumWidth(min_w)
                btn.setMaximumWidth(max_w)
                btn.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
            except Exception:
                pass

        btn_row = QWidget()
        brl = QHBoxLayout(btn_row)
        self.btn_load_txt = QPushButton('Load TXT')
        self.btn_load_json = QPushButton('Load JSON')
        self.btn_save_fitted = QPushButton('Save fitted TXT')
        self.btn_export_params = QPushButton('Export params JSON')
        brl.addWidget(self.btn_load_txt)
        brl.addWidget(self.btn_load_json)
        brl.addWidget(self.btn_save_fitted)
        brl.addWidget(self.btn_export_params)
        try:
            # Main action buttons can span the available row width.
            self.btn_load_txt.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
            self.btn_load_json.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
            self.btn_save_fitted.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
            self.btn_export_params.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        except Exception:
            pass
        left_l.addWidget(btn_row)

        # show currently loaded data file
        self.loaded_file_label = QLabel('No file loaded')
        try:
            self.loaded_file_label.setToolTip('Currently loaded TXT data file')
        except Exception:
            pass
        left_l.addWidget(self.loaded_file_label)

        btn_fit_row = QWidget()
        bfr = QHBoxLayout(btn_fit_row)
        bfr.setContentsMargins(0, 0, 0, 0)
        self.btn_step = QPushButton('Step Fit')
        self.btn_fit = QPushButton('Fit (full)')
        self.btn_cancel = QPushButton('Cancel')
        bfr.addWidget(self.btn_step)
        bfr.addWidget(self.btn_fit)
        bfr.addWidget(self.btn_cancel)
        try:
            self.btn_step.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
            self.btn_fit.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
            self.btn_cancel.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        except Exception:
            pass
        left_l.addWidget(btn_fit_row)

        # Integrator & backend controls
        # Controls arranged in two rows so the left pane can be narrowed
        ctrl_container = QWidget()
        ctrl_v = QVBoxLayout(ctrl_container)
        ctrl_v.setContentsMargins(0, 0, 0, 0)

        # Row 1: backend + integrator + accelerator (grouped label+control pairs)
        ctrl_row1 = QWidget()
        ctrl1 = QHBoxLayout(ctrl_row1)
        ctrl1.setContentsMargins(0, 0, 0, 0)
        try:
            ctrl1.setSpacing(6)
        except Exception:
            pass

        # Backend group
        self.backend_combo = QComboBox()
        self.backend_combo.addItems(['lmfit', 'scipy'])
        try:
            self.backend_combo.setFixedWidth(100)
        except Exception:
            pass
        backend_w = QWidget()
        backend_l = QHBoxLayout(backend_w)
        backend_l.setContentsMargins(0, 0, 0, 0)
        backend_l.setSpacing(4)
        backend_l.addWidget(QLabel('Backend:'))
        backend_l.addWidget(self.backend_combo)
        try:
            backend_w.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred)
        except Exception:
            pass
        ctrl1.addWidget(backend_w)

        # Integrator group
        self.integrator_combo = QComboBox()
        self.integrator_combo.addItems(['grid', 'quad'])
        try:
            self.integrator_combo.setFixedWidth(80)
        except Exception:
            pass
        integ_w = QWidget()
        integ_l = QHBoxLayout(integ_w)
        integ_l.setContentsMargins(0, 0, 0, 0)
        integ_l.setSpacing(4)
        integ_l.addWidget(QLabel('Integrator:'))
        integ_l.addWidget(self.integrator_combo)
        try:
            integ_w.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred)
        except Exception:
            pass
        ctrl1.addWidget(integ_w)

        # Accelerator group
        self.accel_combo = QComboBox()
        self.accel_combo.addItems(['auto', 'numba', 'numpy'])
        self.accel_combo.setCurrentText('auto')
        try:
            self.accel_combo.setFixedWidth(80)
        except Exception:
            pass
        accel_w = QWidget()
        accel_l = QHBoxLayout(accel_w)
        accel_l.setContentsMargins(0, 0, 0, 0)
        accel_l.setSpacing(4)
        accel_l.addWidget(QLabel('Accelerator:'))
        accel_l.addWidget(self.accel_combo)
        try:
            accel_w.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred)
        except Exception:
            pass
        ctrl1.addWidget(accel_w)

        ctrl_v.addWidget(ctrl_row1)

        # Row 2: kernel + fit range controls
        ctrl_row1b = QWidget()
        ctrl1b = QHBoxLayout(ctrl_row1b)
        ctrl1b.setContentsMargins(0, 0, 0, 0)
        # Kernel selector group (label + combo + info button)
        self.kernel_combo = QComboBox()
        self.kernel_combo.addItems(['PCM_Fano_Bessel', 'PCM_Fano_Gauss'])
        # Compact single-character info button (ℹ) — keeps tooltip
        self.btn_kernel_info = QPushButton('ℹ')
        try:
            fm = self.btn_kernel_info.fontMetrics()
            w = max(20, fm.horizontalAdvance('ℹ') + 12)
            h = max(20, fm.height() + 8)
            self.btn_kernel_info.setFixedSize(int(w), int(h))
            self.btn_kernel_info.setStyleSheet('padding:0px; border-radius:%dpx;' % (int(h/2)))
        except Exception:
            try:
                self.btn_kernel_info.setFixedSize(26, 26)
            except Exception:
                pass
        kernel_w = QWidget()
        kernel_l = QHBoxLayout(kernel_w)
        kernel_l.setContentsMargins(0, 0, 0, 0)
        kernel_l.setSpacing(4)
        kernel_l.addWidget(QLabel('Kernel:'))
        try:
            self.kernel_combo.setFixedWidth(160)
        except Exception:
            pass
        kernel_l.addWidget(self.kernel_combo)
        kernel_l.addWidget(self.btn_kernel_info)
        try:
            kernel_w.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred)
        except Exception:
            pass
        ctrl1b.addWidget(kernel_w)
        # Group label + spin widgets so they remain attached when resizing
        self.fit_xmin_spin = QSpinBox()
        self.fit_xmin_spin.setRange(-999999999, 999999999)
        self.fit_xmin_spin.setSingleStep(1)
        self.fit_xmin_spin.setValue(0)
        try:
            self.fit_xmin_spin.setFixedWidth(88)
        except Exception:
            pass
        fit_xmin_w = QWidget()
        fit_xmin_l = QHBoxLayout(fit_xmin_w)
        fit_xmin_l.setContentsMargins(0, 0, 0, 0)
        fit_xmin_l.setSpacing(4)
        fit_xmin_l.addWidget(QLabel('Fit x start:'))
        fit_xmin_l.addWidget(self.fit_xmin_spin)
        try:
            fit_xmin_w.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred)
        except Exception:
            pass
        ctrl1b.addWidget(fit_xmin_w)

        self.fit_xmax_spin = QSpinBox()
        self.fit_xmax_spin.setRange(-999999999, 999999999)
        self.fit_xmax_spin.setSingleStep(1)
        self.fit_xmax_spin.setValue(0)
        try:
            self.fit_xmax_spin.setFixedWidth(88)
        except Exception:
            pass
        fit_xmax_w = QWidget()
        fit_xmax_l = QHBoxLayout(fit_xmax_w)
        fit_xmax_l.setContentsMargins(0, 0, 0, 0)
        fit_xmax_l.setSpacing(4)
        fit_xmax_l.addWidget(QLabel('Fit x end:'))
        fit_xmax_l.addWidget(self.fit_xmax_spin)
        try:
            fit_xmax_w.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred)
        except Exception:
            pass
        ctrl1b.addWidget(fit_xmax_w)
        self.btn_apply_range = QPushButton('Apply Range')
        try:
            _shrink_btn(self.btn_apply_range, pad=26)
            self.btn_apply_range.setMinimumWidth(max(self.btn_apply_range.minimumWidth(), 116))
        except Exception:
            pass
        ctrl1b.addWidget(self.btn_apply_range)
        # Rescale checkbox with indicator on the left and text right next to it.
        self.rescale_cb = QCheckBox('Rescale\nplot to range')
        self.rescale_cb.setChecked(False)
        self.rescale_cb.setToolTip('When checked, zoom the main plot to the selected fit x-range')
        try:
            self.rescale_cb.setFixedWidth(130)
        except Exception:
            pass
        ctrl1b.addWidget(self.rescale_cb)
        ctrl_v.addWidget(ctrl_row1b)
        try:
            ctrl1b.setSpacing(6)
        except Exception:
            pass

        # Row 3: LM method, max evals, autoscale (grouped label+control pairs)
        ctrl_row2 = QWidget()
        ctrl2 = QHBoxLayout(ctrl_row2)
        ctrl2.setContentsMargins(0, 0, 0, 0)
        try:
            ctrl2.setSpacing(6)
        except Exception:
            pass

        # LM method group
        self.lm_method_combo = QComboBox()
        self.lm_method_combo.addItems(['least_squares', 'leastsq', 'nelder', 'powell', 'lbfgsb'])
        try:
            self.lm_method_combo.setFixedWidth(150)
        except Exception:
            pass
        lm_w = QWidget()
        lm_l = QHBoxLayout(lm_w)
        lm_l.setContentsMargins(0, 0, 0, 0)
        lm_l.setSpacing(4)
        lm_l.addWidget(QLabel('LM method:'))
        lm_l.addWidget(self.lm_method_combo)
        try:
            lm_w.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred)
        except Exception:
            pass
        ctrl2.addWidget(lm_w)

        # Max evals group
        self.maxeval_spin = QSpinBox()
        self.maxeval_spin.setRange(1, 20000000)
        self.maxeval_spin.setSingleStep(100)
        self.maxeval_spin.setValue(10000)
        try:
            self.maxeval_spin.setFixedWidth(100)
        except Exception:
            pass
        max_w = QWidget()
        max_l = QHBoxLayout(max_w)
        max_l.setContentsMargins(0, 0, 0, 0)
        max_l.setSpacing(4)
        max_l.addWidget(QLabel('Max evals:'))
        max_l.addWidget(self.maxeval_spin)
        try:
            max_w.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred)
        except Exception:
            pass
        ctrl2.addWidget(max_w)

        # Autoscale group (checkbox)
        self.autoscale_cb = QCheckBox()
        self.autoscale_cb.setChecked(True)
        autoscale_w = QWidget()
        autoscale_l = QHBoxLayout(autoscale_w)
        autoscale_l.setContentsMargins(0, 0, 0, 0)
        autoscale_l.setSpacing(4)
        autoscale_l.addWidget(QLabel('Autoscale'))
        autoscale_l.addWidget(self.autoscale_cb)
        try:
            autoscale_w.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred)
        except Exception:
            pass
        ctrl2.addWidget(autoscale_w)

        # Post-fit diagnostics group (checkbox)
        self.postfit_diag_cb = QCheckBox()
        # Off by default to keep UI responsive after each fit.
        self.postfit_diag_cb.setChecked(False)
        postfit_w = QWidget()
        postfit_l = QHBoxLayout(postfit_w)
        postfit_l.setContentsMargins(0, 0, 0, 0)
        postfit_l.setSpacing(4)
        postfit_l.addWidget(QLabel('Post-fit diagnostics'))
        postfit_l.addWidget(self.postfit_diag_cb)
        try:
            postfit_w.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred)
        except Exception:
            pass
        ctrl2.addWidget(postfit_w)

        ctrl_v.addWidget(ctrl_row2)

        # Row 4: grid and preview options
        ctrl_row3 = QWidget()
        ctrl3 = QHBoxLayout(ctrl_row3)
        ctrl3.setContentsMargins(0, 0, 0, 0)
        # Grid size group
        self.grid_spin = QSpinBox()
        self.grid_spin.setRange(10, 10000)
        self.grid_spin.setSingleStep(10)
        self.grid_spin.setValue(4000)
        try:
            self.grid_spin.setFixedWidth(80)
        except Exception:
            pass
        grid_w = QWidget()
        grid_l = QHBoxLayout(grid_w)
        grid_l.setContentsMargins(0, 0, 0, 0)
        grid_l.setSpacing(4)
        grid_l.addWidget(QLabel('Grid size:'))
        grid_l.addWidget(self.grid_spin)
        try:
            grid_w.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred)
        except Exception:
            pass
        ctrl3.addWidget(grid_w)

        # ik_min group
        self.ik_min_spin = QDoubleSpinBox()
        self.ik_min_spin.setRange(-10.0, 10.0)
        self.ik_min_spin.setSingleStep(0.01)
        self.ik_min_spin.setValue(0.0)
        try:
            self.ik_min_spin.setMinimumContentsLength(7)
            self.ik_min_spin.setFixedWidth(86)
        except Exception:
            pass
        ikmin_w = QWidget()
        ikmin_l = QHBoxLayout(ikmin_w)
        ikmin_l.setContentsMargins(0, 0, 0, 0)
        ikmin_l.setSpacing(4)
        ikmin_l.addWidget(QLabel('ik_min:'))
        ikmin_l.addWidget(self.ik_min_spin)
        try:
            ikmin_w.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred)
        except Exception:
            pass
        ctrl3.addWidget(ikmin_w)

        # ik_max group
        self.ik_max_spin = QDoubleSpinBox()
        self.ik_max_spin.setRange(-10.0, 10.0)
        self.ik_max_spin.setSingleStep(0.01)
        self.ik_max_spin.setValue(1.0)
        try:
            self.ik_max_spin.setMinimumContentsLength(7)
            self.ik_max_spin.setFixedWidth(86)
        except Exception:
            pass
        ikmax_w = QWidget()
        ikmax_l = QHBoxLayout(ikmax_w)
        ikmax_l.setContentsMargins(0, 0, 0, 0)
        ikmax_l.setSpacing(4)
        ikmax_l.addWidget(QLabel('ik_max:'))
        ikmax_l.addWidget(self.ik_max_spin)
        try:
            ikmax_w.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred)
        except Exception:
            pass
        ctrl3.addWidget(ikmax_w)

        # Preview grid group
        self.preview_spin = QSpinBox()
        self.preview_spin.setRange(10, 2000)
        self.preview_spin.setValue(100)
        try:
            self.preview_spin.setFixedWidth(80)
        except Exception:
            pass
        preview_w = QWidget()
        preview_l = QHBoxLayout(preview_w)
        preview_l.setContentsMargins(0, 0, 0, 0)
        preview_l.setSpacing(4)
        preview_l.addWidget(QLabel('Preview grid:'))
        preview_l.addWidget(self.preview_spin)
        try:
            preview_w.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred)
        except Exception:
            pass
        ctrl3.addWidget(preview_w)
        self.btn_preview = QPushButton('Preview')
        ctrl3.addWidget(self.btn_preview)
        self.btn_normalize = QPushButton('Normalize')
        ctrl3.addWidget(self.btn_normalize)
        try:
            _shrink_btn(self.btn_preview, pad=26)
            _shrink_btn(self.btn_normalize, pad=26)
            self.btn_preview.setMinimumWidth(max(self.btn_preview.minimumWidth(), 92))
            self.btn_normalize.setMinimumWidth(max(self.btn_normalize.minimumWidth(), 102))
        except Exception:
            pass
        ctrl_v.addWidget(ctrl_row3)
        try:
            ctrl3.setSpacing(6)
        except Exception:
            pass

        left_l.addWidget(ctrl_container)
        # Tooltips for quick help
        try:
            self.backend_combo.setToolTip('Choose fitting backend: lmfit (rich features) or scipy (curve_fit)')
            self.integrator_combo.setToolTip('Integrator: "grid" is fast/approximate, "quad" is accurate but slow')
            self.accel_combo.setToolTip('Acceleration mode for grid integrator: auto, numba, or numpy')
            self.lm_method_combo.setToolTip('Minimizer method used by lmfit; try least_squares or leastsq')
            self.maxeval_spin.setToolTip('Maximum function evaluations for a full fit')
            self.autoscale_cb.setToolTip('Autoscale initial guesses for N and y0 from the loaded data')
            self.postfit_diag_cb.setToolTip('When enabled, run sensitivity and finite-difference correlation diagnostics after fit (slower)')
            self.grid_spin.setToolTip('Grid size used by the grid integrator (larger = more accurate/slower)')
            self.ik_min_spin.setToolTip('Lower integration limit (ik)')
            self.ik_max_spin.setToolTip('Upper integration limit (ik)')
            self.preview_spin.setToolTip('Grid size used for preview plots')
            self.btn_normalize.setToolTip('Normalize y to [0,1] using (y-min)/(max-min)')
            self.kernel_combo.setToolTip('Choose kernel/integrand shape used in the model')
            self.btn_kernel_info.setToolTip('Show formula and parameter names for the selected kernel')
            self.btn_preview.setToolTip('Render a preview of the model with the current parameter values')
            self.btn_step.setToolTip('Run a short fit step (5 function evaluations)')
            self.btn_fit.setToolTip('Run the full fit until convergence or max evals')
            self.btn_cancel.setToolTip('Cancel the running fit')
        except Exception:
            pass

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)
        left_l.addWidget(self.progress)

        self.accel_indicator = QLabel('Accelerator: auto')
        try:
            self.accel_indicator.setToolTip('Effective accelerator used by model evaluation')
        except Exception:
            pass
        left_l.addWidget(self.accel_indicator)

        # Fit status: use a scrollable plain-text box for long messages
        self.fit_status = QPlainTextEdit()
        self.fit_status.setReadOnly(True)
        try:
            self.fit_status.setFixedHeight(80)
        except Exception:
            pass
        self.fit_status.setPlainText('Idle')
        left_l.addWidget(self.fit_status)

        left_l.addWidget(QLabel('Parameters'))
        try:
            # Combined row: Gaussian checkbox + count and Lorentzian checkbox + count in one line
            combo_row = QWidget()
            combo_layout = QHBoxLayout(combo_row)
            combo_layout.setContentsMargins(0, 0, 0, 0)
            # Gaussian controls
            self.gauss_cb = QCheckBox('Gaussians')
            self.gauss_cb.setChecked(False)
            self.gauss_cb.setToolTip('Enable Gaussian components')
            self.gauss_cb.toggled.connect(self._toggle_gaussian)
            combo_layout.addWidget(self.gauss_cb)
            combo_layout.addWidget(QLabel('Count:'))
            self.gauss_count_spin = QSpinBox()
            self.gauss_count_spin.setRange(1, 10)
            self.gauss_count_spin.setValue(1)
            try:
                self.gauss_count_spin.setFixedWidth(70)
            except Exception:
                pass
            self.gauss_count_spin.setEnabled(False)
            self.gauss_count_spin.valueChanged.connect(lambda v: (self._ensure_gaussian_params(), self._populate_param_table()))
            combo_layout.addWidget(self.gauss_count_spin)
            # Spacer
            combo_layout.addStretch(1)
            # Lorentzian controls
            self.lorentz_cb = QCheckBox('Lorentzians')
            self.lorentz_cb.setChecked(False)
            self.lorentz_cb.setToolTip('Enable Lorentzian components')
            self.lorentz_cb.toggled.connect(self._toggle_lorentz)
            combo_layout.addWidget(self.lorentz_cb)
            combo_layout.addWidget(QLabel('Count:'))
            self.lorentz_count_spin = QSpinBox()
            self.lorentz_count_spin.setRange(1, 10)
            self.lorentz_count_spin.setValue(1)
            try:
                self.lorentz_count_spin.setFixedWidth(70)
            except Exception:
                pass
            self.lorentz_count_spin.setEnabled(False)
            self.lorentz_count_spin.valueChanged.connect(lambda v: (self._ensure_lorentz_params(), self._populate_param_table()))
            combo_layout.addWidget(self.lorentz_count_spin)
            left_l.addWidget(combo_row)
        except Exception:
            pass
        # Columns: Name, Value, Std, Vary (small), Min, Max, Corr
        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(['Name', 'Value', 'Std', 'Vary', 'Min', 'Max', 'Corr'])
        header = self.table.horizontalHeader()
        # Make the name column narrower and let Value expand; allow interactive resize
        # Make all columns user-resizable so the user can drag widths freely
        header.setSectionResizeMode(0, QHeaderView.Interactive)
        header.setSectionResizeMode(1, QHeaderView.Interactive)
        header.setSectionResizeMode(2, QHeaderView.Interactive)
        header.setSectionResizeMode(3, QHeaderView.Interactive)
        header.setSectionResizeMode(4, QHeaderView.Interactive)
        header.setSectionResizeMode(5, QHeaderView.Interactive)
        header.setSectionResizeMode(6, QHeaderView.Interactive)
        # set preferred widths (tweakable)
        try:
            self.table.setColumnWidth(0, 60)   # Name (narrower)
            self.table.setColumnWidth(1, 180)  # Value (narrower)
            self.table.setColumnWidth(2, 80)   # Std
            self.table.setColumnWidth(3, 48)   # Vary
            self.table.setColumnWidth(4, 80)   # Min
            self.table.setColumnWidth(5, 80)   # Max
            self.table.setColumnWidth(6, 80)   # Corr
        except Exception:
            pass
        left_l.addWidget(self.table)

        # Right pane: matplotlib preview + toolbar (zoom/pan/home)
        right = QWidget()
        right_l = QVBoxLayout(right)
        right_l.setContentsMargins(0, 0, 0, 0)

        plot_btn_row = QWidget()
        plot_btn_layout = QHBoxLayout(plot_btn_row)
        plot_btn_layout.setContentsMargins(0, 0, 0, 0)
        self.btn_plot_rescale_y = QPushButton('Rescale Y')
        self.btn_plot_rescale_all = QPushButton('Rescale All')
        self.btn_plot_rescale_y.setToolTip('Rescale Y axes for current visible X range')
        self.btn_plot_rescale_all.setToolTip('Reset both X and Y axes to show all plotted data')
        plot_btn_layout.addWidget(self.btn_plot_rescale_y)
        plot_btn_layout.addWidget(self.btn_plot_rescale_all)
        try:
            _shrink_btn(self.btn_plot_rescale_y)
            _shrink_btn(self.btn_plot_rescale_all)
        except Exception:
            pass
        plot_btn_layout.addStretch(1)

        self.figure = Figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        gs = self.figure.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.2)
        self.ax_main = self.figure.add_subplot(gs[0, 0])
        self.ax_resid = self.figure.add_subplot(gs[1, 0], sharex=self.ax_main)

        try:
            self._mpl_motion_cid = self.canvas.mpl_connect('motion_notify_event', self._on_canvas_motion)
            self._mpl_pick_cid = self.canvas.mpl_connect('pick_event', self._on_legend_pick)
            self._mpl_leave_cid = self.canvas.mpl_connect('axes_leave_event', self._on_axes_leave)
        except Exception:
            pass

        right_l.addWidget(plot_btn_row)
        right_l.addWidget(self.toolbar)
        right_l.addWidget(self.canvas, 1)

        from PySide6.QtWidgets import QSplitter
        splitter = QSplitter(QtCore.Qt.Horizontal)
        splitter.addWidget(left)
        splitter.addWidget(right)
        # reasonable initial sizes (left pane small)
        try:
            splitter.setSizes([360, 800])
            splitter.setHandleWidth(12)
            splitter.setStyleSheet('QSplitter::handle { background-color: #DDDDDD; } QSplitter::handle:horizontal { width: 12px; }')
            try:
                splitter.setOpaqueResize(False)
            except Exception:
                pass
        except Exception:
            pass

        self.setCentralWidget(splitter)

        # Connect
        self.btn_load_txt.clicked.connect(self.load_data)
        self.btn_load_json.clicked.connect(self.load_params)
        self.btn_save_fitted.clicked.connect(self.save_fitted)
        self.btn_export_params.clicked.connect(self.export_params)
        self.btn_step.clicked.connect(lambda: self.start_fit(mode='step'))
        self.btn_fit.clicked.connect(lambda: self.start_fit(mode='full'))
        self.btn_cancel.clicked.connect(self.cancel_fit)
        self.btn_preview.clicked.connect(self.preview_model)
        self.btn_plot_rescale_y.clicked.connect(self._rescale_plot_y)
        self.btn_plot_rescale_all.clicked.connect(self._rescale_plot_all)
        try:
            self.btn_apply_range.clicked.connect(lambda: self._apply_fit_range())
        except Exception:
            pass
        try:
            self.rescale_cb.stateChanged.connect(lambda *_: self._render_preview(y_model=getattr(self, 'last_y_model', None)))
        except Exception:
            pass
        try:
            self.btn_normalize.clicked.connect(self.normalize_data)
        except Exception:
            pass
        try:
            self.kernel_combo.currentTextChanged.connect(self._on_kernel_changed)
        except Exception:
            pass
        try:
            self.btn_kernel_info.clicked.connect(self._show_kernel_info)
        except Exception:
            pass
        try:
            self.accel_combo.currentTextChanged.connect(lambda *_: self._update_accel_indicator())
            self.integrator_combo.currentTextChanged.connect(lambda *_: self._update_accel_indicator())
        except Exception:
            pass

    def load_data(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Open data file', '.', 'Text Files (*.txt);;All Files (*)')
        if not path:
            return
        try:
            iw, y = load_data(path)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, 'Error', f'Could not load data: {e}')
            return
        self.iw = iw
        self.y = y
        # if fit-range controls exist, set their sensible ranges to the data
        try:
            xv = np.asarray(self.iw, dtype=float)
            xmin = float(np.min(xv))
            xmax = float(np.max(xv))
            if hasattr(self, 'fit_xmin_spin'):
                imin = int(np.floor(xmin))
                imax = int(np.ceil(xmax))
                # clamp to spinbox allowable range
                imin = max(-999999999, imin)
                imax = min(999999999, imax)
                self.fit_xmin_spin.setRange(imin, imax)
                self.fit_xmin_spin.setValue(imin)
            if hasattr(self, 'fit_xmax_spin'):
                imin = int(np.floor(xmin))
                imax = int(np.ceil(xmax))
                imin = max(-999999999, imin)
                imax = min(999999999, imax)
                self.fit_xmax_spin.setRange(imin, imax)
                self.fit_xmax_spin.setValue(imax)
        except Exception:
            pass
        # remember and display the loaded file path (basename)
        try:
            self.current_data_path = path
            # show filename plus number of points and x-range to aid debugging
            try:
                npts = len(self.iw) if self.iw is not None else 0
                self.loaded_file_label.setText(f"{os.path.basename(path)} ({npts} pts, {xmin:.6g}-{xmax:.6g})")
                print(f"Loaded {npts} points from {path}; x range {xmin}-{xmax}", flush=True)
            except Exception:
                self.loaded_file_label.setText(os.path.basename(path) if path is not None else '')
        except Exception:
            pass
        self._render_preview()

    def load_params(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Open params JSON', 'configs', 'JSON Files (*.json);;All Files (*)')
        if not path:
            return
        try:
            cfg = load_param_config(path)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, 'Error', f'Could not load params: {e}')
            return
        self.param_config = cfg
        self._populate_param_table()

    def preview_model(self):
        if self.iw is None or self.y is None:
            QtWidgets.QMessageBox.warning(self, 'Warning', 'Load data first')
            return
        # If no parameter configuration is loaded, attempt to auto-initialize
        # parameters from the project's configs (prefer `params_gauss.json`
        # or `params_bessel.json`) or fall back to embedded defaults.
        if not self.param_config:
            try:
                self._ensure_param_config_initialized()
            except Exception:
                pass
            if not self.param_config:
                QtWidgets.QMessageBox.warning(self, 'Warning', 'Load parameter JSON first')
                return
        # read current table values into param_config
        self._read_table_into_config()

        # build plain params dict for model()
        params = {name: info.get('value', 0.0) for name, info in self.param_config.items()}
        integrator = self.integrator_combo.currentText() if hasattr(self, 'integrator_combo') else 'grid'
        grid_size = int(self.preview_spin.value()) if hasattr(self, 'preview_spin') else 100
        ik_min = float(self.ik_min_spin.value()) if hasattr(self, 'ik_min_spin') else 0.0
        ik_max = float(self.ik_max_spin.value()) if hasattr(self, 'ik_max_spin') else 1.0

        kernel = self.kernel_combo.currentText() if hasattr(self, 'kernel_combo') else 'PCM_Fano_Bessel'
        try:
            y_model = compute_model(self.iw, params, integrator=integrator, grid_size=grid_size, ik_min=ik_min, ik_max=ik_max, kernel=kernel, accelerator=self._selected_accelerator())
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, 'Error', f'Preview generation failed: {e}')
            return

        self.last_y_model = np.asarray(y_model)
        self._render_preview(y_model=self.last_y_model)

    def _populate_param_table(self):
        self.table.setRowCount(0)
        for name, info in self.param_config.items():
            row = self.table.rowCount()
            self.table.insertRow(row)
            # Name
            itm = QTableWidgetItem(name)
            itm.setFlags(itm.flags() & ~QtCore.Qt.ItemIsEditable)
            self.table.setItem(row, 0, itm)
            # Value (editable numeric)
            # format value: scientific 'e' style only if outside [0.01, 100]
            val_raw = info.get('value', '')
            try:
                val_str = sci_format(float(val_raw), decimals=4)
            except Exception:
                val_str = str(val_raw)
            val_edit = ParamLineEdit(val_str)
            val_edit.setValidator(self._new_float_validator())
            val_edit.editingFinished.connect(lambda e=val_edit: self._normalize_decimal_line_edit(e))
            self.table.setCellWidget(row, 1, val_edit)
            # Std
            std_itm = QTableWidgetItem('')
            std_itm.setFlags(std_itm.flags() & ~QtCore.Qt.ItemIsEditable)
            self.table.setItem(row, 2, std_itm)
            # Vary
            cb = QCheckBox()
            cb.setChecked(bool(info.get('vary', True)))
            self.table.setCellWidget(row, 3, cb)
            # Min (editable numeric)
            mn = info.get('min', '')
            try:
                mn_str = sci_format(float(mn), decimals=4) if mn != '' and mn is not None else ''
            except Exception:
                mn_str = str(mn)
            min_edit = ParamLineEdit(mn_str)
            min_edit.setValidator(self._new_float_validator())
            min_edit.editingFinished.connect(lambda e=min_edit: self._normalize_decimal_line_edit(e))
            self.table.setCellWidget(row, 4, min_edit)
            # Max (editable numeric)
            mx = info.get('max', '')
            try:
                mx_str = sci_format(float(mx), decimals=4) if mx != '' and mx is not None else ''
            except Exception:
                mx_str = str(mx)
            max_edit = ParamLineEdit(mx_str)
            max_edit.setValidator(self._new_float_validator())
            max_edit.editingFinished.connect(lambda e=max_edit: self._normalize_decimal_line_edit(e))
            self.table.setCellWidget(row, 5, max_edit)
            # Correlation (filled after fit)
            corr_itm = QTableWidgetItem('')
            corr_itm.setFlags(corr_itm.flags() & ~QtCore.Qt.ItemIsEditable)
            self.table.setItem(row, 6, corr_itm)

    def _read_table_into_config(self):
        # Update self.param_config from table widgets
        cfg = {}
        for r in range(self.table.rowCount()):
            name = self.table.item(r, 0).text()
            # value may be a cell widget (QLineEdit) or a plain item
            val_widget = self.table.cellWidget(r, 1)
            if val_widget is not None:
                val = val_widget.text()
            else:
                val = self.table.item(r, 1).text() if self.table.item(r, 1) is not None else ''
            std = self.table.item(r, 2).text() if self.table.item(r, 2) is not None else ''
            cb: QCheckBox = self.table.cellWidget(r, 3)
            mn_widget = self.table.cellWidget(r, 4)
            if mn_widget is not None:
                mn = mn_widget.text()
            else:
                mn = self.table.item(r, 4).text() if self.table.item(r, 4) is not None else ''
            mx_widget = self.table.cellWidget(r, 5)
            if mx_widget is not None:
                mx = mx_widget.text()
            else:
                mx = self.table.item(r, 5).text() if self.table.item(r, 5) is not None else ''
            entry = {}
            try:
                sval = normalize_number_string(val)
                entry['value'] = float(sval) if sval != '' else float('nan')
            except Exception:
                entry['value'] = float('nan')
            entry['vary'] = bool(cb.isChecked()) if cb is not None else True
            try:
                smn = normalize_number_string(mn)
                entry['min'] = float(smn) if smn != '' else None
            except Exception:
                entry['min'] = None
            try:
                smx = normalize_number_string(mx)
                entry['max'] = float(smx) if smx != '' else None
            except Exception:
                entry['max'] = None
            cfg[name] = entry
        self.param_config = cfg
        return cfg

    def start_fit(self, mode: str = 'full'):
        if self.iw is None or self.y is None:
            QtWidgets.QMessageBox.warning(self, 'Warning', 'Load data first')
            return
        if not self.param_config:
            QtWidgets.QMessageBox.warning(self, 'Warning', 'Load parameter JSON first')
            return
        # Read GUI table into param_config
        self._read_table_into_config()

        # compute free parameter summary (for debug / UI reasons)
        try:
            free_names, p0, bounds = _build_free_params(self.param_config)
            # show number of free params in progress format
            self.progress.setFormat(f'Free params: {len(free_names)}')
        except Exception:
            pass

        # Validate parameter bounds (avoid min == max for actually free parameters)
        try:
            free_names, _, _ = _build_free_params(self.param_config)
        except Exception:
            free_names = []
        for name, info in self.param_config.items():
            mn = info.get('min', None)
            mx = info.get('max', None)
            # only consider parameters that are free according to _build_free_params
            if name in free_names and mn is not None and mx is not None:
                try:
                    if float(mn) == float(mx):
                        QtWidgets.QMessageBox.critical(self, 'Invalid bounds', f'Parameter {name} has min == max while varying. Fix bounds or disable varying.')
                        return
                except Exception:
                    pass

        # Prepare worker (respect UI backend and integrator settings)
        backend = self.backend_combo.currentText() if hasattr(self, 'backend_combo') else 'lmfit'
        integrator_opts = {
            'integrator': self.integrator_combo.currentText() if hasattr(self, 'integrator_combo') else 'grid',
            'grid_size': int(self.grid_spin.value()) if hasattr(self, 'grid_spin') else 4000,
            'ik_min': float(self.ik_min_spin.value()) if hasattr(self, 'ik_min_spin') else 0.0,
            'ik_max': float(self.ik_max_spin.value()) if hasattr(self, 'ik_max_spin') else 1.0,
            'accelerator': self._selected_accelerator(),
            # Always use the parameters shown in the table as the starting
            # guess for both step and full fits (do not autoscale p0 here).
            'autoscale': False,
            'kernel': self.kernel_combo.currentText() if hasattr(self, 'kernel_combo') else 'PCM_Fano_Bessel',
        }

        try:
            self._update_accel_indicator()
        except Exception:
            pass

        # For step-mode fits, disable autoscaling so successive short fits
        # start from the most-recent parameter values (don't reinitialize p0).
        if mode == 'step':
            integrator_opts['autoscale'] = False

        # Prepare worker with integrator options
        # Decide minimizer/curvefit options based on mode (step vs full)
        # Build minimizer / curve_fit options from UI controls
        minimizer_opts = {}
        # user-specified maximum evaluations
        maxeval = int(self.maxeval_spin.value()) if hasattr(self, 'maxeval_spin') else 2000
        # step mode uses a small number of function evaluations; increase
        # proportional to number of free parameters so step fits make
        # measurable progress for both backends.
        if mode == 'step':
            base = 5
            step_n = max(base, base * max(1, len(free_names)))
        else:
            step_n = maxeval

        backend = backend
        if backend == 'lmfit':
            # LMFit: set method and max_nfev
            method = self.lm_method_combo.currentText() if hasattr(self, 'lm_method_combo') else 'least_squares'
            minimizer_opts['method'] = method
            minimizer_opts['max_nfev'] = step_n
        else:
            # SciPy curve_fit expects 'maxfev'
            minimizer_opts['maxfev'] = step_n

        # disable fit buttons while running
        try:
            self.btn_step.setEnabled(False)
            self.btn_fit.setEnabled(False)
        except Exception:
            pass

        # Respect the user-selected fit x-range (if any) and pass the sliced
        # data to the fitter so the backend only sees the requested points.
        try:
            iw_use = np.asarray(self.iw)
            y_use = np.asarray(self.y)
            if hasattr(self, 'fit_xmin_spin') and hasattr(self, 'fit_xmax_spin'):
                x0 = float(self.fit_xmin_spin.value())
                x1 = float(self.fit_xmax_spin.value())
                if x0 > x1:
                    x0, x1 = x1, x0
                mask = (iw_use >= x0) & (iw_use <= x1)
                if np.count_nonzero(mask) == 0:
                    QtWidgets.QMessageBox.warning(self, 'Warning', 'Selected fit range contains no data points; aborting fit')
                    return
                iw_use = iw_use[mask]
                y_use = y_use[mask]
        except Exception:
            iw_use = np.asarray(self.iw)
            y_use = np.asarray(self.y)

        self.fit_worker = FitWorker(iw_use, y_use, self.param_config, backend=backend, integrator_opts=integrator_opts, minimizer_opts=minimizer_opts)
        self.fit_thread = QtCore.QThread()
        self.fit_worker.moveToThread(self.fit_thread)
        self.fit_thread.started.connect(self.fit_worker.run)
        self.fit_worker.progress.connect(self._on_progress)
        self.fit_worker.finished.connect(self._on_fit_finished)
        self.fit_worker.error.connect(self._on_fit_error)
        # Ensure the thread is asked to quit on both success and error to avoid
        # leaving a running QThread object when the worker finishes with an
        # exception (which previously led to "QThread: Destroyed while thread '' is still running").
        self.fit_worker.finished.connect(lambda *_: self.fit_thread.quit())
        self.fit_worker.error.connect(lambda *_: self.fit_thread.quit())
        self.fit_worker.finished.connect(self.fit_worker.deleteLater)
        self.fit_thread.finished.connect(self.fit_thread.deleteLater)

        self.progress.setVisible(True)
        try:
            selected_accel = self._selected_accelerator()
            effective_accel = self._effective_accelerator_label(integrator_opts.get('integrator', 'grid'))
            if selected_accel != effective_accel:
                self.fit_status.setPlainText(f"Running (requested {selected_accel}, using {effective_accel})")
            else:
                self.fit_status.setPlainText(f"Running ({effective_accel})")
        except Exception:
            pass
        self.fit_thread.start()

        # clear last fit result until a new one completes
        self.last_fit_out = None

    def cancel_fit(self):
        if self.fit_worker is not None:
            self.fit_worker.cancel()
            self.progress.setVisible(False)
            try:
                self.fit_status.setPlainText('Cancelling...')
            except Exception:
                pass

    def _on_progress(self, n: int):
        # show an indeterminate progress; optionally display iteration count
        self.progress.setFormat(f'Iterations: {n}')
        self.progress.setVisible(True)

    def _on_fit_finished(self, out: dict):
        self.progress.setVisible(False)
        # re-enable fit buttons
        try:
            self.btn_step.setEnabled(True)
            self.btn_fit.setEnabled(True)
        except Exception:
            pass
        # store last fit output for export
        self.last_fit_out = out
        # pulled fitted/errs early for diagnostics and table updates
        fitted = out.get('fitted', {})
        errs = out.get('errs', {})
        # Persist fitted values into the in-memory param_config so subsequent
        # step fits begin from the updated values (prevents autoscale from
        # overriding incremental progress when autoscale is disabled above).
        try:
            if isinstance(self.param_config, dict) and fitted:
                for name, val in fitted.items():
                    if name in self.param_config:
                        try:
                            self.param_config[name]['value'] = float(val)
                        except Exception:
                            self.param_config[name]['value'] = val
        except Exception:
            pass
        # Update fit status (converged + message + nfev + R^2)
        converged = out.get('converged')
        msg = out.get('message', '')
        nfev = out.get('nfev', None)
        r2 = out.get('r2', None)
        accel_used = out.get('accelerator_used', self._effective_accelerator_label()) if isinstance(out, dict) else self._effective_accelerator_label()
        try:
            selected_accel = self._selected_accelerator()
            if selected_accel != accel_used:
                self.accel_indicator.setText(f'Accelerator: {selected_accel} -> {accel_used}')
            else:
                self.accel_indicator.setText(f'Accelerator: {accel_used}')
        except Exception:
            pass
        try:
            r2s = ('R2=' + ('{:.4g}'.format(float(r2)) if r2 is not None else 'nan'))
        except Exception:
            r2s = 'R2=nan'
        try:
            if converged:
                self.fit_status.setPlainText(f'Converged [{accel_used}] (nfev={nfev}, {r2s})')
            else:
                self.fit_status.setPlainText(f'Not converged [{accel_used}]: {msg} (nfev={nfev}, {r2s})')
        except Exception:
            pass
        try:
            run_postfit_diagnostics = bool(self.postfit_diag_cb.isChecked()) if hasattr(self, 'postfit_diag_cb') else False
        except Exception:
            run_postfit_diagnostics = False
        if not run_postfit_diagnostics:
            try:
                old = self.fit_status.toPlainText()
                self.fit_status.setPlainText(old + ' — diagnostics: off')
            except Exception:
                pass
        # Best-effort sensitivity check: flag free parameters that hardly change
        if run_postfit_diagnostics:
            try:
                free_names, _, _ = _build_free_params(self.param_config)
                current_params = {}
                for name in self.param_config:
                    if name in fitted:
                        current_params[name] = float(fitted[name])
                    else:
                        current_params[name] = float(self.param_config[name].get('value', 0.0))
                insensitive = []
                y_arr = np.asarray(self.y)
                y_range = float(np.max(y_arr) - np.min(y_arr)) if y_arr.size else 0.0
                sens_threshold = max(1e-8, 1e-6 * (y_range if y_range > 0 else float(np.std(y_arr))))
                test_grid = min(200, int(self.preview_spin.value())) if hasattr(self, 'preview_spin') else 100
                kern = self.kernel_combo.currentText() if hasattr(self, 'kernel_combo') else 'PCM_Fano_Bessel'
                for name in free_names:
                    pval = float(current_params.get(name, 0.0))
                    dp = abs(pval * 0.01) if abs(pval) > 0 else 1e-6
                    p_plus = dict(current_params)
                    p_minus = dict(current_params)
                    p_plus[name] = pval + dp
                    p_minus[name] = pval - dp
                    accel = self._selected_accelerator()
                    try:
                        y_plus = compute_model(self.iw, p_plus, integrator='grid', grid_size=test_grid, kernel=kern, accelerator=accel)
                        y_minus = compute_model(self.iw, p_minus, integrator='grid', grid_size=test_grid, kernel=kern, accelerator=accel)
                    except Exception:
                        y_plus = compute_model(self.iw, p_plus, kernel=kern, accelerator=accel)
                        y_minus = compute_model(self.iw, p_minus, kernel=kern, accelerator=accel)
                    rms_change = float(np.sqrt(np.mean((np.asarray(y_plus) - np.asarray(y_minus)) ** 2)))
                    if rms_change < sens_threshold:
                        insensitive.append(name)
                if insensitive:
                    try:
                        old = self.fit_status.toPlainText()
                        self.fit_status.setPlainText(old + ' — Insensitive: ' + ','.join(insensitive))
                    except Exception:
                        pass
            except Exception:
                pass
        # Compute parameter correlations (if covariance available) and report high correlations.
        # If a covariance matrix is not available from the backend, compute a
        # finite-difference Jacobian estimate here so correlations update for
        # any backend and also during step fits.
        try:
            free_names, _, _ = _build_free_params(self.param_config)
        except Exception:
            free_names = []

        pcov = out.get('pcov', None) if out is not None else None
        if pcov is None and out is not None and out.get('result', None) is not None:
            res = out.get('result')
            pcov = None
            try:
                pcov = getattr(res, 'covar', None)
            except Exception:
                pcov = None
            if pcov is None:
                try:
                    pcov = getattr(res, 'covariance', None)
                except Exception:
                    pcov = None

        # If we still don't have pcov, compute a finite-difference estimate
        if pcov is None and run_postfit_diagnostics:
            try:
                fnames = free_names
                nfree = len(fnames)
                if nfree > 0:
                    pvec = np.array([float(fitted.get(n, self.param_config[n].get('value', 0.0))) for n in fnames], dtype=float)
                    ndata = np.asarray(self.iw).size
                    Jfd = np.zeros((ndata, nfree), dtype=float)
                    eps = 1e-6
                    fd_grid = int(min(200, int(self.preview_spin.value()))) if hasattr(self, 'preview_spin') else 100
                    kern = self.kernel_combo.currentText() if hasattr(self, 'kernel_combo') else 'PCM_Fano_Bessel'
                    accel = self._selected_accelerator()
                    for j in range(nfree):
                        pj = pvec[j]
                        dp = eps * max(1.0, abs(pj))
                        if dp == 0:
                            dp = eps
                        p_plus = pvec.copy(); p_minus = pvec.copy()
                        p_plus[j] += dp
                        p_minus[j] -= dp
                        params_plus = _params_from_vector(fnames, p_plus, self.param_config)
                        params_minus = _params_from_vector(fnames, p_minus, self.param_config)
                        try:
                            y_plus = compute_model(self.iw, params_plus, integrator='grid', grid_size=fd_grid, kernel=kern, accelerator=accel)
                            y_minus = compute_model(self.iw, params_minus, integrator='grid', grid_size=fd_grid, kernel=kern, accelerator=accel)
                        except Exception:
                            y_plus = compute_model(self.iw, params_plus, kernel=kern, accelerator=accel)
                            y_minus = compute_model(self.iw, params_minus, kernel=kern, accelerator=accel)
                        deriv = (np.asarray(y_plus) - np.asarray(y_minus)) / (2.0 * dp)
                        Jfd[:, j] = deriv
                    JTJ = Jfd.T.dot(Jfd)
                    params_base = _params_from_vector(fnames, pvec, self.param_config)
                    try:
                        y_base = compute_model(self.iw, params_base, integrator='grid', grid_size=fd_grid, kernel=kern, accelerator=accel)
                    except Exception:
                        y_base = compute_model(self.iw, params_base, kernel=kern, accelerator=accel)
                    resid_vec = np.asarray(self.y) - np.asarray(y_base)
                    dof = max(1, ndata - nfree)
                    s2 = float(np.sum(resid_vec ** 2) / dof) if dof > 0 else float(np.sum(resid_vec ** 2))
                    try:
                        pcov = np.linalg.pinv(JTJ) * s2
                    except Exception:
                        pcov = None
            except Exception:
                pcov = None

        # If we have a covariance matrix (either from backend or FD), compute correlations
        if pcov is not None:
            try:
                pcov = np.asarray(pcov)
                # Map names: if pcov covers free params directly use that order,
                # otherwise try to map into the full param list.
                param_keys = list(self.param_config.keys())
                n_p = pcov.shape[0]
                if n_p == len(free_names):
                    names_for_matrix = free_names
                elif n_p == len(param_keys):
                    names_for_matrix = param_keys
                else:
                    names_for_matrix = param_keys[:n_p]
                idx_map = {n: i for i, n in enumerate(names_for_matrix)}
                free_indices = [idx_map[n] for n in free_names if n in idx_map]
                if len(free_indices) > 0:
                    pcov_sub = pcov[np.ix_(free_indices, free_indices)]
                    diag = np.sqrt(np.diag(pcov_sub))
                    with np.errstate(divide='ignore', invalid='ignore'):
                        corr = pcov_sub / (diag[:, None] * diag[None, :])
                    corr = np.nan_to_num(corr)
                    thresh = 0.9
                    corr_map = {}
                    present_names = [n for n in free_names if n in idx_map]
                    for i, name in enumerate(present_names):
                        if corr.shape[0] > 1:
                            maxabs = float(np.max(np.abs(np.delete(corr[i, :], i))))
                        else:
                            maxabs = 0.0
                        corr_map[name] = maxabs
                    high = []
                    for i in range(len(present_names)):
                        for j in range(i+1, len(present_names)):
                            if abs(corr[i, j]) >= thresh:
                                high.append(f"{present_names[i]}-{present_names[j]}:{corr[i,j]:.3g}")
                    if high:
                        try:
                            old = self.fit_status.toPlainText()
                            self.fit_status.setPlainText(old + ' — HighCorr: ' + ','.join(high))
                        except Exception:
                            pass
                    try:
                        for r in range(self.table.rowCount()):
                            pname = self.table.item(r, 0).text()
                            if pname in corr_map:
                                val = '{:.4f}'.format(corr_map[pname])
                                if self.table.item(r, 6) is not None:
                                    self.table.item(r, 6).setText(val)
                                else:
                                    itm = QTableWidgetItem(val)
                                    itm.setFlags(itm.flags() & ~QtCore.Qt.ItemIsEditable)
                                    self.table.setItem(r, 6, itm)
                            else:
                                if self.table.item(r, 6) is not None:
                                    self.table.item(r, 6).setText('—')
                    except Exception:
                        pass
            except Exception:
                pass
        else:
            try:
                for r in range(self.table.rowCount()):
                    if self.table.item(r, 6) is not None:
                        self.table.item(r, 6).setText('—')
            except Exception:
                pass
        # Update param table with fitted values and stds
        fitted = out.get('fitted', {})
        errs = out.get('errs', {})
        # Update table
        for r in range(self.table.rowCount()):
            name = self.table.item(r, 0).text()
            if name in fitted:
                val_widget = self.table.cellWidget(r, 1)
                val_text = ''
                try:
                    val_text = sci_format(float(fitted[name]), decimals=4)
                except Exception:
                    val_text = str(fitted[name])
                if val_widget is not None:
                    val_widget.setText(val_text)
                else:
                    if self.table.item(r, 1) is not None:
                        self.table.item(r, 1).setText(val_text)
            if name in errs and errs[name] is not None:
                try:
                    self.table.item(r, 2).setText(sci_format(float(errs[name]), decimals=4))
                except Exception:
                    self.table.item(r, 2).setText(str(errs[name]))

        # Highlight min/max cells when fitted values hit parameter bounds
        try:
            for r in range(self.table.rowCount()):
                pname = self.table.item(r, 0).text()
                try:
                    pval = float(fitted.get(pname, self.param_config.get(pname, {}).get('value', float('nan'))))
                except Exception:
                    # skip non-numeric values
                    continue
                pmin = self.param_config.get(pname, {}).get('min', None)
                pmax = self.param_config.get(pname, {}).get('max', None)
                try:
                    tol = max(1e-8, 1e-6 * max(abs(pval), 1.0))
                except Exception:
                    tol = 1e-8
                # cell widgets for Min/Max (QLineEdit); may be None
                min_w = self.table.cellWidget(r, 4)
                max_w = self.table.cellWidget(r, 5)
                # clear previous highlighting
                try:
                    if min_w is not None:
                        min_w.setStyleSheet('')
                except Exception:
                    pass
                try:
                    if max_w is not None:
                        max_w.setStyleSheet('')
                except Exception:
                    pass
                # Only highlight min/max boxes when the parameter is set to vary
                try:
                    vary_cb = self.table.cellWidget(r, 3)
                    vary_enabled = bool(vary_cb.isChecked()) if vary_cb is not None else False
                except Exception:
                    vary_enabled = False
                # apply highlight when value is at or very near bound and vary is enabled
                try:
                    if vary_enabled and pmin is not None and np.isfinite(float(pmin)) and pval <= float(pmin) + tol:
                        if min_w is not None:
                            min_w.setStyleSheet('background-color: rgba(255,0,0,0.15);')
                except Exception:
                    pass
                try:
                    if vary_enabled and pmax is not None and np.isfinite(float(pmax)) and pval >= float(pmax) - tol:
                        if max_w is not None:
                            max_w.setStyleSheet('background-color: rgba(255,0,0,0.15);')
                except Exception:
                    pass
        except Exception:
            pass

        # Update plots: compute the model on the full x-grid so the overlay
        # shows the fitted function across all loaded data, even if the
        # backend was only given a sliced subset.
        try:
            fitted_for_plot = {}
            if isinstance(fitted, dict) and fitted:
                fitted_for_plot = {n: float(v) for n, v in fitted.items()}
            else:
                # fall back to current param_config values
                fitted_for_plot = {n: float(self.param_config[n].get('value', 0.0)) for n in self.param_config}
            kern = self.kernel_combo.currentText() if hasattr(self, 'kernel_combo') else 'PCM_Fano_Bessel'
            grid_sz = int(self.grid_spin.value()) if hasattr(self, 'grid_spin') else 4000
            integrator = self.integrator_combo.currentText() if hasattr(self, 'integrator_combo') else 'grid'
            try:
                y_backend = out.get('y_model') if isinstance(out, dict) else None
                if y_backend is not None and np.asarray(y_backend).size == np.asarray(self.iw).size:
                    self.last_y_model = np.asarray(y_backend)
                else:
                    y_model_full = compute_model(self.iw, fitted_for_plot, integrator=integrator, grid_size=grid_sz, ik_min=float(self.ik_min_spin.value()) if hasattr(self, 'ik_min_spin') else 0.0, ik_max=float(self.ik_max_spin.value()) if hasattr(self, 'ik_max_spin') else 1.0, kernel=kern, accelerator=self._selected_accelerator())
                    self.last_y_model = np.asarray(y_model_full)
            except Exception:
                # fallback to backend-provided y_model if full recompute fails
                y_model = out.get('y_model')
                self.last_y_model = np.asarray(y_model) if y_model is not None else None
        except Exception:
            try:
                y_model = out.get('y_model')
                self.last_y_model = np.asarray(y_model) if y_model is not None else None
            except Exception:
                self.last_y_model = None
        if self.last_y_model is not None:
            self._render_preview(y_model=self.last_y_model)

    def _on_fit_error(self, msg: str):
        self.progress.setVisible(False)
        QtWidgets.QMessageBox.critical(self, 'Fit error', msg)
        try:
            self.btn_step.setEnabled(True)
            self.btn_fit.setEnabled(True)
        except Exception:
            pass
        try:
            self.fit_status.setPlainText(f'Error: {msg}')
        except Exception:
            pass

    def closeEvent(self, event):
        # Attempt to gracefully stop any running fit thread before closing.
        try:
            if getattr(self, 'fit_thread', None) is not None and self.fit_thread.isRunning():
                if getattr(self, 'fit_worker', None) is not None:
                    try:
                        self.fit_worker.cancel()
                    except Exception:
                        pass
                try:
                    self.fit_thread.quit()
                except Exception:
                    pass
                # wait briefly for thread to finish
                try:
                    self.fit_thread.wait(3000)
                except Exception:
                    pass
        except Exception:
            pass
        super().closeEvent(event)

    def _render_preview(self, y_model=None):
        # Draw data+fit and residuals directly on a matplotlib canvas.
        if self.iw is None or self.y is None:
            return

        try:
            x_full = np.asarray(self.iw, dtype=float)
            y_full = np.asarray(self.y, dtype=float)
        except Exception:
            x_full = np.asarray(self.iw)
            y_full = np.asarray(self.y)

        if x_full.size == 0 or y_full.size == 0:
            self.ax_main.clear()
            self.ax_resid.clear()
            self.canvas.draw_idle()
            return

        plot_idx = None
        if x_full.size > 6000:
            plot_idx = np.linspace(0, x_full.size - 1, 6000, dtype=int)
            x_plot_arr = x_full[plot_idx]
            y_plot_arr = y_full[plot_idx]
        else:
            x_plot_arr = x_full
            y_plot_arr = y_full

        self.ax_main.clear()
        self.ax_resid.clear()
        self._legend_artist_map = {}
        self._hover_last_idx = None
        try:
            self._hover_x = np.asarray(x_plot_arr, dtype=float)
            self._hover_y = np.asarray(y_plot_arr, dtype=float)
        except Exception:
            self._hover_x = None
            self._hover_y = None
        try:
            self._hover_annotation = self.ax_main.annotate(
                '',
                xy=(0, 0),
                xytext=(12, 12),
                textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#666666', alpha=0.9),
            )
            self._hover_annotation.set_visible(False)
        except Exception:
            self._hover_annotation = None
        try:
            self._hover_marker = self.ax_main.plot(
                [],
                [],
                marker='o',
                markersize=9,
                markerfacecolor='none',
                markeredgecolor='#ff7f0e',
                markeredgewidth=1.8,
                linestyle='None',
                zorder=8,
            )[0]
            self._hover_marker.set_visible(False)
        except Exception:
            self._hover_marker = None

        # Highlight points that hit dataset min/max to help diagnose clipping.
        try:
            y_min = float(np.min(y_full))
            y_max = float(np.max(y_full))
            edge_mask = np.isclose(y_plot_arr, y_min) | np.isclose(y_plot_arr, y_max)
            if np.any(~edge_mask):
                self.ax_main.scatter(x_plot_arr[~edge_mask], y_plot_arr[~edge_mask], s=18, c='tab:blue', alpha=0.8, label='data')
            if np.any(edge_mask):
                self.ax_main.scatter(x_plot_arr[edge_mask], y_plot_arr[edge_mask], s=24, marker='s', c='red', alpha=0.25, edgecolors='none', label='data limits')
        except Exception:
            self.ax_main.scatter(x_plot_arr, y_plot_arr, s=18, c='tab:blue', alpha=0.8, label='data')

        plotted_model = False
        try:
            params_for_model = {name: info.get('value', 0.0) for name, info in self.param_config.items()} if isinstance(self.param_config, dict) else None
        except Exception:
            params_for_model = None

        if params_for_model is not None and ((getattr(self, 'gauss_cb', None) and self.gauss_cb.isChecked()) or (getattr(self, 'lorentz_cb', None) and self.lorentz_cb.isChecked())):
            try:
                grid_sz = int(self.preview_spin.value()) if hasattr(self, 'preview_spin') else 100
                integrator = self.integrator_combo.currentText() if hasattr(self, 'integrator_combo') else 'grid'
                ik_min = float(self.ik_min_spin.value()) if hasattr(self, 'ik_min_spin') else 0.0
                ik_max = float(self.ik_max_spin.value()) if hasattr(self, 'ik_max_spin') else 1.0
                base_comp, gauss_total, gauss_components, lorentz_total, lorentz_components = model_components(
                    np.asarray(self.iw, dtype=float),
                    params_for_model,
                    integrator=integrator,
                    grid_size=grid_sz,
                    ik_min=ik_min,
                    ik_max=ik_max,
                    kernel=(self.kernel_combo.currentText() if hasattr(self, 'kernel_combo') else None),
                    accelerator=self._selected_accelerator(),
                )
                base_arr = np.asarray(base_comp, dtype=float)
                gauss_total_arr = np.asarray(gauss_total, dtype=float)
                lorentz_total_arr = np.asarray(lorentz_total, dtype=float)
                sum_arr = base_arr.copy()
                if getattr(self, 'gauss_cb', None) and self.gauss_cb.isChecked():
                    sum_arr = sum_arr + gauss_total_arr
                if getattr(self, 'lorentz_cb', None) and self.lorentz_cb.isChecked():
                    sum_arr = sum_arr + lorentz_total_arr

                base_plot = base_arr[plot_idx] if plot_idx is not None and base_arr.size == x_full.size else base_arr
                sum_plot = sum_arr[plot_idx] if plot_idx is not None and sum_arr.size == x_full.size else sum_arr
                if base_plot.size == x_plot_arr.size:
                    self.ax_main.plot(x_plot_arr, base_plot, linestyle='--', linewidth=1.6, label='integral component')
                if getattr(self, 'gauss_cb', None) and self.gauss_cb.isChecked() and gauss_components is not None:
                    for j, comp in enumerate(gauss_components):
                        try:
                            arr = np.asarray(comp, dtype=float)
                            if np.any(np.abs(arr) > 1e-12):
                                arr_plot = arr[plot_idx] if plot_idx is not None and arr.size == x_full.size else arr
                                if arr_plot.size == x_plot_arr.size:
                                    self.ax_main.plot(x_plot_arr, arr_plot, linestyle=':', linewidth=1.4, label=f'gaussian {j+1}')
                        except Exception:
                            pass
                if getattr(self, 'lorentz_cb', None) and self.lorentz_cb.isChecked() and lorentz_components is not None:
                    for j, comp in enumerate(lorentz_components):
                        try:
                            arr = np.asarray(comp, dtype=float)
                            if np.any(np.abs(arr) > 1e-12):
                                arr_plot = arr[plot_idx] if plot_idx is not None and arr.size == x_full.size else arr
                                if arr_plot.size == x_plot_arr.size:
                                    self.ax_main.plot(x_plot_arr, arr_plot, linestyle=':', linewidth=1.4, label=f'lorentzian {j+1}')
                        except Exception:
                            pass
                if sum_plot.size == x_plot_arr.size:
                    self.ax_main.plot(x_plot_arr, sum_plot, linewidth=2.2, label='fit')
                plotted_model = True
            except Exception:
                plotted_model = False

        if not plotted_model and y_model is not None:
            try:
                y_model_arr = np.asarray(y_model, dtype=float)
                if plot_idx is not None and y_model_arr.size == x_full.size:
                    y_model_arr = y_model_arr[plot_idx]
                if y_model_arr.size == x_plot_arr.size:
                    self.ax_main.plot(x_plot_arr, y_model_arr, linewidth=2.0, label='fit')
            except Exception:
                pass

        # Determine selected fit-range mask (if any).
        try:
            x_arr = np.asarray(self.iw, dtype=float)
            y_arr = np.asarray(self.y, dtype=float)
        except Exception:
            x_arr = np.asarray(self.iw)
            y_arr = np.asarray(self.y)

        mask = None
        x0 = None
        x1 = None
        try:
            if hasattr(self, 'fit_xmin_spin') and hasattr(self, 'fit_xmax_spin'):
                x0 = float(self.fit_xmin_spin.value())
                x1 = float(self.fit_xmax_spin.value())
                if x0 > x1:
                    x0, x1 = x1, x0
                mask = (x_arr >= x0) & (x_arr <= x1)
                if x0 < x1:
                    self.ax_main.axvspan(x0, x1, facecolor='lightsalmon', alpha=0.2)
        except Exception:
            mask = None

        # Residuals: prefer explicit full-length y_model, then cached model,
        # then recompute from latest fit output/current params.
        y_model_arr = None
        try:
            if y_model is not None and np.asarray(y_model).size == x_arr.size:
                y_model_arr = np.asarray(y_model, dtype=float)
        except Exception:
            y_model_arr = None

        if y_model_arr is None:
            try:
                if getattr(self, 'last_y_model', None) is not None and np.asarray(self.last_y_model).size == x_arr.size:
                    y_model_arr = np.asarray(self.last_y_model, dtype=float)
            except Exception:
                y_model_arr = None

        if y_model_arr is None:
            params_try = None
            if getattr(self, 'last_fit_out', None) is not None:
                out = self.last_fit_out
                params_try = out.get('fitted') or out.get('params') or None
            if params_try is None:
                try:
                    params_try = {name: info.get('value', 0.0) for name, info in self.param_config.items()}
                except Exception:
                    params_try = None
            if params_try is not None:
                try:
                    grid_sz = int(self.grid_spin.value()) if hasattr(self, 'grid_spin') else 4000
                    integrator = self.integrator_combo.currentText() if hasattr(self, 'integrator_combo') else 'grid'
                    ik_min = float(self.ik_min_spin.value()) if hasattr(self, 'ik_min_spin') else 0.0
                    ik_max = float(self.ik_max_spin.value()) if hasattr(self, 'ik_max_spin') else 1.0
                    y_full_try = compute_model(
                        self.iw,
                        params_try,
                        integrator=integrator,
                        grid_size=grid_sz,
                        ik_min=ik_min,
                        ik_max=ik_max,
                        kernel=(self.kernel_combo.currentText() if hasattr(self, 'kernel_combo') else None),
                        accelerator=self._selected_accelerator(),
                    )
                    if np.asarray(y_full_try).size == x_arr.size:
                        y_model_arr = np.asarray(y_full_try, dtype=float)
                except Exception:
                    y_model_arr = None

        resid_arr = None
        if y_model_arr is not None:
            try:
                resid_arr = y_arr - y_model_arr
                if plot_idx is not None and resid_arr.size == x_arr.size:
                    res_x = x_arr[plot_idx]
                    res_y = resid_arr[plot_idx]
                else:
                    res_x = x_arr
                    res_y = resid_arr
                if res_x.size > 0:
                    self.ax_resid.scatter(res_x, res_y, s=12, c='0.3', label='residuals')
            except Exception:
                resid_arr = None

        if resid_arr is None:
            self.ax_resid.scatter(x_plot_arr, np.zeros_like(x_plot_arr), s=12, c='0.3', label='residuals')

        # Rescale main plot axes only when the checkbox is enabled.
        try:
            if hasattr(self, 'rescale_cb') and self.rescale_cb.isChecked() and mask is not None and np.any(mask):
                self.ax_main.set_xlim(float(x0), float(x1))
                yvals = []
                try:
                    yvals.extend(list(y_arr[mask]))
                except Exception:
                    pass
                try:
                    if y_model_arr is not None:
                        yvals.extend(list(y_model_arr[mask]))
                except Exception:
                    pass
                yvals_arr = np.asarray(yvals, dtype=float)
                yvals_arr = yvals_arr[np.isfinite(yvals_arr)]
                if yvals_arr.size > 0:
                    ymin = float(np.min(yvals_arr))
                    ymax = float(np.max(yvals_arr))
                    if ymin == ymax:
                        ymin -= 1e-6
                        ymax += 1e-6
                    pad = max(1e-6, 0.05 * (ymax - ymin))
                    self.ax_main.set_ylim(ymin - pad, ymax + pad)
        except Exception:
            pass

        # Residual y autoscale in fit region, with optional x-range restriction.
        try:
            if mask is not None and np.any(mask) and resid_arr is not None:
                resid_mask = np.asarray(resid_arr[mask], dtype=float)
                resid_mask = resid_mask[np.isfinite(resid_mask)]
                if resid_mask.size > 0:
                    ymin = float(np.min(resid_mask))
                    ymax = float(np.max(resid_mask))
                    if ymin == ymax:
                        ymin -= 1e-6
                        ymax += 1e-6
                    pad = max(1e-6, 0.05 * (ymax - ymin))
                    self.ax_resid.set_ylim(ymin - pad, ymax + pad)
            if mask is not None and np.any(mask) and getattr(self, 'rescale_cb', None) and self.rescale_cb.isChecked():
                self.ax_resid.set_xlim(float(x0), float(x1))
        except Exception:
            pass

        # Cache arrays used by manual rescale controls.
        try:
            self._main_plot_cache = {
                'x': np.asarray(x_arr, dtype=float),
                'y_data': np.asarray(y_arr, dtype=float),
                'y_model': (np.asarray(y_model_arr, dtype=float) if y_model_arr is not None else None),
            }
        except Exception:
            self._main_plot_cache = {}
        try:
            self._resid_plot_cache = {
                'x': np.asarray(x_arr, dtype=float),
                'resid': (np.asarray(resid_arr, dtype=float) if resid_arr is not None else None),
            }
        except Exception:
            self._resid_plot_cache = {}

        self.ax_main.set_title('Data and Fit')
        self.ax_main.set_ylabel('Intensity')
        self.ax_main.grid(True, alpha=0.25)
        self.ax_resid.set_title('Residuals')
        self.ax_resid.set_ylabel('Residuals')
        self.ax_resid.set_xlabel('X')
        self.ax_resid.axhline(0.0, color='black', linewidth=0.8, alpha=0.5)
        self.ax_resid.grid(True, alpha=0.25)
        try:
            handles, labels = self.ax_main.get_legend_handles_labels()
            if handles:
                legend = self.ax_main.legend(loc='upper center', bbox_to_anchor=(0.5, 1.18), ncol=min(4, len(handles)), fontsize=8)
                leg_handles = getattr(legend, 'legend_handles', None)
                if leg_handles is None:
                    leg_handles = getattr(legend, 'legendHandles', [])
                for leg_h, orig_h in zip(leg_handles, handles):
                    try:
                        leg_h.set_picker(True)
                        if hasattr(leg_h, 'set_pickradius'):
                            leg_h.set_pickradius(8)
                    except Exception:
                        pass
                    self._legend_artist_map[leg_h] = orig_h
        except Exception:
            pass

        try:
            self.figure.tight_layout()
        except Exception:
            pass
        self.canvas.draw_idle()
        self.last_y_model = None if y_model is None else np.asarray(y_model)

    def save_fitted(self):
        # Save currently shown fit data (if available)
        if self.iw is None:
            QtWidgets.QMessageBox.warning(self, 'Warning', 'No data to save')
            return
        path, _ = QFileDialog.getSaveFileName(self, 'Save fitted data', 'fitted.txt', 'Text Files (*.txt);;All Files (*)')
        if not path:
            return
        # Ask whether to save the full spectrum or only the selected range
        save_full = False
        try:
            dlg = QtWidgets.QMessageBox(self)
            dlg.setWindowTitle('Save Range')
            dlg.setText('Save full spectrum or only the selected range?')
            full_btn = dlg.addButton('Full spectrum', QtWidgets.QMessageBox.AcceptRole)
            sel_btn = dlg.addButton('Selected range', QtWidgets.QMessageBox.AcceptRole)
            cancel_btn = dlg.addButton(QtWidgets.QMessageBox.Cancel)
            dlg.exec()
            clicked = dlg.clickedButton()
            if clicked == cancel_btn:
                return
            elif clicked == full_btn:
                save_full = True
            else:
                save_full = False
        except Exception:
            save_full = False
        try:
            # determine which points to export depending on the user's choice
            try:
                x_arr = np.asarray(self.iw, dtype=float)
                if save_full:
                    mask = np.ones_like(x_arr, dtype=bool)
                else:
                    if hasattr(self, 'fit_xmin_spin') and hasattr(self, 'fit_xmax_spin'):
                        x0 = float(self.fit_xmin_spin.value())
                        x1 = float(self.fit_xmax_spin.value())
                        if x0 > x1:
                            x0, x1 = x1, x0
                        mask = (x_arr >= x0) & (x_arr <= x1)
                        if not np.any(mask):
                            QtWidgets.QMessageBox.warning(self, 'Warning', 'Selected range contains no data points; nothing to save')
                            return
                    else:
                        mask = np.ones_like(x_arr, dtype=bool)
            except Exception:
                x_arr = np.asarray(self.iw, dtype=float)
                mask = np.ones_like(x_arr, dtype=bool)

            x_sel = x_arr[mask]
            y_sel = np.asarray(self.y, dtype=float)[mask]

            # determine fitted values and per-component contributions for the selected points
            base_full = None
            gauss_total_full = None
            gauss_components_full = None
            lorentz_total_full = None
            lorentz_components_full = None
            y_fit = None

            # prefer using last fit output parameters when available, else use current table values
            params_plot = None
            if getattr(self, 'last_fit_out', None) is not None:
                out = self.last_fit_out
                params_plot = out.get('fitted') or out.get('params') or None
            if params_plot is None:
                try:
                    params_plot = {name: info.get('value', 0.0) for name, info in self.param_config.items()}
                except Exception:
                    params_plot = None

            integrator = self.integrator_combo.currentText() if hasattr(self, 'integrator_combo') else 'grid'
            grid_size = int(self.grid_spin.value()) if hasattr(self, 'grid_spin') else 4000
            ik_min = float(self.ik_min_spin.value()) if hasattr(self, 'ik_min_spin') else 0.0
            ik_max = float(self.ik_max_spin.value()) if hasattr(self, 'ik_max_spin') else 1.0
            kernel_name = self.kernel_combo.currentText() if hasattr(self, 'kernel_combo') else 'PCM_Fano_Bessel'

            if params_plot is not None:
                try:
                    base_full, gauss_total_full, gauss_components_full, lorentz_total_full, lorentz_components_full = model_components(self.iw, params_plot, integrator=integrator, grid_size=grid_size, ik_min=ik_min, ik_max=ik_max, kernel=kernel_name, accelerator=self._selected_accelerator())
                    if np.asarray(base_full).size == x_arr.size and np.asarray(gauss_total_full).size == x_arr.size and np.asarray(lorentz_total_full).size == x_arr.size:
                        y_fit = (np.asarray(base_full) + np.asarray(gauss_total_full) + np.asarray(lorentz_total_full))[mask]
                except Exception:
                    base_full = None
                    gauss_total_full = None
                    gauss_components_full = None
                    lorentz_total_full = None
                    lorentz_components_full = None

            # fallback: use last_y_model array if recompute failed
            if y_fit is None and getattr(self, 'last_y_model', None) is not None:
                try:
                    y_model_arr = np.asarray(self.last_y_model, dtype=float)
                    if y_model_arr.size == x_arr.size:
                        y_fit = y_model_arr[mask]
                except Exception:
                    y_fit = None

            # final fallback: try compute_model from current param_config values
            if y_fit is None:
                try:
                    params = {name: info.get('value', 0.0) for name, info in self.param_config.items()}
                    y_full = compute_model(self.iw, params, integrator=integrator, grid_size=grid_size, ik_min=ik_min, ik_max=ik_max, kernel=kernel_name, accelerator=self._selected_accelerator())
                    y_fit = np.asarray(y_full, dtype=float)[mask]
                    # try to compute components as well
                    try:
                        base_full, gauss_total_full, gauss_components_full, lorentz_total_full, lorentz_components_full = model_components(self.iw, params, integrator=integrator, grid_size=grid_size, ik_min=ik_min, ik_max=ik_max, kernel=kernel_name, accelerator=self._selected_accelerator())
                    except Exception:
                        base_full = None
                        gauss_total_full = None
                        gauss_components_full = None
                        lorentz_total_full = None
                        lorentz_components_full = None
                except Exception:
                    y_fit = np.full_like(x_sel, np.nan, dtype=float)

            # prepare parameter block (prefer last fit output fitted values and errs)
            params_out = {}
            errs_out = {}
            if getattr(self, 'last_fit_out', None) is not None:
                out = self.last_fit_out
                params_out = out.get('fitted') or out.get('params') or {}
                errs_out = out.get('errs') or {}
            if not params_out:
                # fall back to current param_config values
                try:
                    params_out = {name: info.get('value', None) for name, info in self.param_config.items()}
                except Exception:
                    params_out = {}

            with open(path, 'w', encoding='utf-8') as f:
                # header comment block with metadata and parameters
                f.write(f"# Exported by PeakFit GUI\n")
                try:
                    src = getattr(self, 'current_data_path', '')
                    f.write(f"# Source: {src}\n")
                except Exception:
                    pass
                try:
                    if x_sel.size > 0:
                        f.write(f"# Range: {float(x_sel[0])} to {float(x_sel[-1])} (points={len(x_sel)})\n")
                except Exception:
                    pass
                f.write("# Parameters:\n")
                for k, v in params_out.items():
                    try:
                        val_str = sci_format(float(v), decimals=6)
                    except Exception:
                        val_str = str(v)
                    std = errs_out.get(k, None)
                    if std is not None:
                        try:
                            std_str = sci_format(float(std), decimals=6)
                            f.write(f"# {k} = {val_str} +/- {std_str}\n")
                        except Exception:
                            f.write(f"# {k} = {val_str} +/- {std}\n")
                    else:
                        f.write(f"# {k} = {val_str}\n")

                # column header: include integral and individual gaussian and lorentzian component columns
                try:
                    # Respect GUI-enabled counts when available (user requested number of components)
                    if getattr(self, 'gauss_cb', None) and self.gauss_cb.isChecked() and hasattr(self, 'gauss_count_spin'):
                        n_gauss = int(self.gauss_count_spin.value())
                    else:
                        # fallback: count non-empty gaussian components
                        if gauss_components_full is not None and len(gauss_components_full) > 0:
                            n_gauss = len(gauss_components_full)
                        elif gauss_total_full is not None:
                            n_gauss = 1
                        else:
                            n_gauss = 0
                except Exception:
                    n_gauss = 0
                try:
                    if getattr(self, 'lorentz_cb', None) and self.lorentz_cb.isChecked() and hasattr(self, 'lorentz_count_spin'):
                        n_lorentz = int(self.lorentz_count_spin.value())
                    else:
                        if lorentz_components_full is not None and len(lorentz_components_full) > 0:
                            # only include lorentzian columns if non-zero contributions exist
                            # count components heuristically
                            try:
                                n_lorentz = sum(1 for c in lorentz_components_full if np.any(np.abs(np.asarray(c)) > 1e-12))
                                if n_lorentz == 0 and lorentz_total_full is not None and np.any(np.abs(np.asarray(lorentz_total_full)) > 1e-12):
                                    n_lorentz = 1
                            except Exception:
                                n_lorentz = len(lorentz_components_full)
                        elif lorentz_total_full is not None and np.any(np.abs(np.asarray(lorentz_total_full)) > 1e-12):
                            n_lorentz = 1
                        else:
                            n_lorentz = 0
                except Exception:
                    n_lorentz = 0
                gauss_names = [f"gaussian_{i+1}" for i in range(n_gauss)]
                lorentz_names = [f"lorentzian_{i+1}" for i in range(n_lorentz)]
                cols = ["x", "experimental", "fitted", "residual", "integral"] + gauss_names + lorentz_names
                f.write("# " + "\t".join(cols) + "\n")

                # prepare per-selected arrays for integral and gaussians (fill NaN when not available)
                try:
                    if base_full is not None and np.asarray(base_full).size == x_arr.size:
                        base_sel = np.asarray(base_full)[mask]
                    else:
                        base_sel = np.full_like(x_sel, float('nan'), dtype=float)
                except Exception:
                    base_sel = np.full_like(x_sel, float('nan'), dtype=float)

                gauss_sel_lists = []
                lorentz_sel_lists = []
                # Build gauss columns according to requested or detected count
                try:
                    if n_gauss > 0:
                        if gauss_components_full is not None and len(gauss_components_full) >= n_gauss:
                            for i in range(n_gauss):
                                gauss_sel_lists.append(np.asarray(gauss_components_full[i])[mask])
                        else:
                            # try to include available components then pad with NaN columns
                            if gauss_components_full is not None and len(gauss_components_full) > 0:
                                for comp in gauss_components_full:
                                    gauss_sel_lists.append(np.asarray(comp)[mask])
                                while len(gauss_sel_lists) < n_gauss:
                                    gauss_sel_lists.append(np.full_like(x_sel, float('nan'), dtype=float))
                            elif gauss_total_full is not None and np.asarray(gauss_total_full).size == x_arr.size:
                                gauss_sel_lists.append(np.asarray(gauss_total_full)[mask])
                                while len(gauss_sel_lists) < n_gauss:
                                    gauss_sel_lists.append(np.full_like(x_sel, float('nan'), dtype=float))
                            else:
                                gauss_sel_lists = [np.full_like(x_sel, float('nan'), dtype=float) for _ in range(n_gauss)]
                    else:
                        gauss_sel_lists = []
                except Exception:
                    gauss_sel_lists = [np.full_like(x_sel, float('nan'), dtype=float) for _ in range(n_gauss)]
                # Build lorentz columns according to requested or detected count
                try:
                    if n_lorentz > 0:
                        if lorentz_components_full is not None and len(lorentz_components_full) >= n_lorentz:
                            for i in range(n_lorentz):
                                lorentz_sel_lists.append(np.asarray(lorentz_components_full[i])[mask])
                        else:
                            if lorentz_components_full is not None and len(lorentz_components_full) > 0:
                                for comp in lorentz_components_full:
                                    lorentz_sel_lists.append(np.asarray(comp)[mask])
                                while len(lorentz_sel_lists) < n_lorentz:
                                    lorentz_sel_lists.append(np.full_like(x_sel, float('nan'), dtype=float))
                            elif lorentz_total_full is not None and np.asarray(lorentz_total_full).size == x_arr.size:
                                lorentz_sel_lists.append(np.asarray(lorentz_total_full)[mask])
                                while len(lorentz_sel_lists) < n_lorentz:
                                    lorentz_sel_lists.append(np.full_like(x_sel, float('nan'), dtype=float))
                            else:
                                lorentz_sel_lists = [np.full_like(x_sel, float('nan'), dtype=float) for _ in range(n_lorentz)]
                    else:
                        lorentz_sel_lists = []
                except Exception:
                    lorentz_sel_lists = [np.full_like(x_sel, float('nan'), dtype=float) for _ in range(n_lorentz)]

                try:
                    resid_arr = (np.asarray(y_sel, dtype=float) - np.asarray(y_fit, dtype=float))
                except Exception:
                    resid_arr = np.full_like(x_sel, float('nan'), dtype=float)

                for idx in range(len(x_sel)):
                    xi = x_sel[idx]
                    yi = y_sel[idx]
                    try:
                        yf_val = float(y_fit[idx])
                    except Exception:
                        yf_val = float('nan')
                    try:
                        yr = float(resid_arr[idx])
                    except Exception:
                        yr = float('nan')
                    try:
                        base_val = float(base_sel[idx])
                    except Exception:
                        base_val = float('nan')
                    row_vals = [xi, yi, yf_val, yr, base_val]
                    for glist in gauss_sel_lists:
                        try:
                            row_vals.append(float(glist[idx]))
                        except Exception:
                            row_vals.append(float('nan'))
                    for llist in lorentz_sel_lists:
                        try:
                            row_vals.append(float(llist[idx]))
                        except Exception:
                            row_vals.append(float('nan'))
                    try:
                        f.write("\t".join(f"{v:.6g}" for v in row_vals) + "\n")
                    except Exception:
                        # fallback to a simpler write if formatting fails
                        f.write("\t".join(str(v) for v in row_vals) + "\n")

            QtWidgets.QMessageBox.information(self, 'Saved', f'Saved data to {path}')
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, 'Error', str(e))

    def export_params(self):
        if not self.param_config:
            QtWidgets.QMessageBox.warning(self, 'Warning', 'No parameter configuration to export')
            return
        # Read current table into config and export
        cfg = self._read_table_into_config()
        path, _ = QFileDialog.getSaveFileName(self, 'Export params JSON', 'params_export.json', 'JSON Files (*.json);;All Files (*)')
        if not path:
            return
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(cfg, f, indent=2)
            QtWidgets.QMessageBox.information(self, 'Exported', f'Exported params to {path}')
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, 'Error', str(e))

    def export_html_report(self):
        QtWidgets.QMessageBox.information(
            self,
            'Not available',
            'GUI HTML export is disabled in this build. Use Save fitted TXT/Export params JSON, or use scripts/fit_peak.py for HTML output.',
        )

    def _show_kernel_info(self):
        try:
            k = self.kernel_combo.currentText() if hasattr(self, 'kernel_combo') else 'PCM_Fano_Bessel'
            kl = str(k).strip().lower()
            if kl in ('pcm_fano_bessel', 'sinc', 'bessel'):
                txt = ("Kernel: PCM_Fano_Bessel (original Bessel/sinc-like)\n"
                       "Fitting function: Y = y0 + N/((q^2 + 1)*L^3) * integral( kernel(ik, iw) d(ik) )\n"
                       "w0 = sqrt(C + D*cos(ik*pi/2)) - b\n"
                       "eps = 2*(iw - w0)/g0\n"
                       "den = (sin(x) - x*cos(x))**2 / ik**4   with x = (ik*pi/a)*L\n"
                       "num = (eps + q)**2 / (1 + eps**2)\n"
                       "Parameters: C, D, b, g0, q, a, L, N, y0")
            else:
                  txt = ("Kernel: PCM_Fano_Gauss (Gaussian-like envelope)\n"
                      "Fitting function: Y = y0 + N*(L^3)/(q^2 + 1) * alpha^(-4/3) * integral( kernel(ik, iw) d(ik) )\n"
                      "w0 = sqrt(C + D*cos(ik*pi/2)) - b\n"
                      "eps = 2*(iw - w0)/g0\n"
                      "den = ik^2 * exp((-2*pi^2 * ik^2 * L^2)/(alpha*a^2))\n"
                      "num = (eps + q)**2 / (1 + eps**2)\n"
                      "Parameters: C, D, b, g0, q, a, L, alpha, N, y0")
            QtWidgets.QMessageBox.information(self, 'Kernel info', txt)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, 'Error', f'Failed to show kernel info: {e}')
        except Exception:
            pass

    def _on_kernel_changed(self, txt: str):
        sel = str(txt).strip().lower()
        # Initialize parameter set for the selected kernel. This will
        # populate the parameter table with sensible starting values.
        try:
            if 'gauss' in sel:
                params_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'configs', 'params_gauss.json'))
                if os.path.exists(params_path):
                    try:
                        cfg = load_param_config(params_path)
                        self.param_config = cfg
                    except Exception:
                        self.param_config = copy.deepcopy(GAUSS_DEFAULT_PARAMS)
                else:
                    self.param_config = copy.deepcopy(GAUSS_DEFAULT_PARAMS)
            else:
                params_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'configs', 'params_bessel.json'))
                if os.path.exists(params_path):
                    try:
                        cfg = load_param_config(params_path)
                        self.param_config = cfg
                    except Exception:
                        # fallback to default_params.json
                        dp = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'configs', 'default_params.json'))
                        if os.path.exists(dp):
                            try:
                                self.param_config = load_param_config(dp)
                            except Exception:
                                self.param_config = copy.deepcopy(BESSEL_DEFAULT_PARAMS)
                        else:
                            self.param_config = copy.deepcopy(BESSEL_DEFAULT_PARAMS)
                else:
                    dp = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'configs', 'default_params.json'))
                    if os.path.exists(dp):
                        try:
                            self.param_config = load_param_config(dp)
                        except Exception:
                            self.param_config = copy.deepcopy(BESSEL_DEFAULT_PARAMS)
                    else:
                        self.param_config = copy.deepcopy(BESSEL_DEFAULT_PARAMS)
            try:
                # If the GUI has the Gaussian or Lorentz-enable checkbox checked, ensure
                # the corresponding parameters are present in the loaded config
                if getattr(self, 'gauss_cb', None) and self.gauss_cb.isChecked():
                    try:
                        self._ensure_gaussian_params()
                    except Exception:
                        pass
                if getattr(self, 'lorentz_cb', None) and self.lorentz_cb.isChecked():
                    try:
                        self._ensure_lorentz_params()
                    except Exception:
                        pass
                self._populate_param_table()
            except Exception:
                pass
        except Exception:
            pass

    def _ensure_param_config_initialized(self):
        """Ensure `self.param_config` is populated. Prefer project config
        files in `configs/` where available; otherwise use embedded defaults.
        Returns the resulting config dict."""
        if isinstance(self.param_config, dict) and self.param_config:
            return self.param_config
        sel = self.kernel_combo.currentText() if hasattr(self, 'kernel_combo') else 'PCM_Fano_Bessel'
        sel = str(sel).strip().lower()
        try:
            if 'gauss' in sel:
                params_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'configs', 'params_gauss.json'))
                if os.path.exists(params_path):
                    try:
                        cfg = load_param_config(params_path)
                        self.param_config = cfg
                    except Exception:
                        self.param_config = copy.deepcopy(GAUSS_DEFAULT_PARAMS)
                else:
                    self.param_config = copy.deepcopy(GAUSS_DEFAULT_PARAMS)
            else:
                params_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'configs', 'params_bessel.json'))
                if os.path.exists(params_path):
                    try:
                        cfg = load_param_config(params_path)
                        self.param_config = cfg
                    except Exception:
                        dp = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'configs', 'default_params.json'))
                        if os.path.exists(dp):
                            try:
                                self.param_config = load_param_config(dp)
                            except Exception:
                                self.param_config = copy.deepcopy(BESSEL_DEFAULT_PARAMS)
                        else:
                            self.param_config = copy.deepcopy(BESSEL_DEFAULT_PARAMS)
                else:
                    dp = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'configs', 'default_params.json'))
                    if os.path.exists(dp):
                        try:
                            self.param_config = load_param_config(dp)
                        except Exception:
                            self.param_config = copy.deepcopy(BESSEL_DEFAULT_PARAMS)
                    else:
                        self.param_config = copy.deepcopy(BESSEL_DEFAULT_PARAMS)
            try:
                self._populate_param_table()
            except Exception:
                pass
        except Exception:
            pass
        # If the extra Gaussian or Lorentzian checkbox is on, ensure its params exist
        try:
            if getattr(self, 'gauss_cb', None) and self.gauss_cb.isChecked():
                try:
                    self._ensure_gaussian_params()
                except Exception:
                    pass
            if getattr(self, 'lorentz_cb', None) and self.lorentz_cb.isChecked():
                try:
                    self._ensure_lorentz_params()
                except Exception:
                    pass
        except Exception:
            pass
        return self.param_config

    def _ensure_gaussian_params(self):
        """Add default Gaussian parameters to `self.param_config` if missing.

        Parameters added: `N_g` (amplitude), `x0` (center), `gg` (FWHM).
        Supports multiple components when `self.gauss_count_spin` is present
        (adds suffixed parameters `N_g1`, `x0_1`, `gg_1`, ...).
        """
        try:
            if not isinstance(self.param_config, dict):
                self.param_config = {}
            # Number of gaussian components requested (default 1)
            try:
                count = int(self.gauss_count_spin.value()) if hasattr(self, 'gauss_count_spin') else 1
            except Exception:
                count = 1
            # sensible defaults and bounds for each gaussian component
            # Use canonical indexed names: Ng1, x0g1, gg1, Ng2, x0g2, gg2, ...
            for i in range(1, count + 1):
                key_area = f'Ng{i}'
                key_x0 = f'x0g{i}'
                key_gg = f'gg{i}'
                defaults = {
                    key_area: {'value': 0.1 if i == 1 else 0.0, 'vary': True, 'min': 0.0, 'max': 1.0},
                    key_x0: {'value': 480.0, 'vary': True, 'min': 400.0, 'max': 500.0},
                    key_gg: {'value': 10.0, 'vary': True, 'min': 5.0, 'max': 100.0},
                }
                for k, v in defaults.items():
                    if k not in self.param_config:
                        self.param_config[k] = v
            # Migrate legacy unsuffixed gaussian keys to the new indexed names
            try:
                # legacy area: 'N_g' or 'Ng' -> 'Ng1'
                if 'N_g' in self.param_config and 'Ng1' not in self.param_config:
                    try:
                        self.param_config['Ng1'] = self.param_config.pop('N_g')
                    except Exception:
                        pass
                if 'Ng' in self.param_config and 'Ng1' not in self.param_config:
                    try:
                        self.param_config['Ng1'] = self.param_config.pop('Ng')
                    except Exception:
                        pass
                # legacy center: 'x0' -> 'x0g1'
                if 'x0' in self.param_config and 'x0g1' not in self.param_config:
                    try:
                        self.param_config['x0g1'] = self.param_config.pop('x0')
                    except Exception:
                        pass
                # legacy width: 'gg' -> 'gg1'
                if 'gg' in self.param_config and 'gg1' not in self.param_config:
                    try:
                        self.param_config['gg1'] = self.param_config.pop('gg')
                    except Exception:
                        pass
            except Exception:
                pass
            # Remove any higher-indexed gaussian params if the count was reduced
            try:
                # detect existing gaussian-like keys and remove those > count
                keys = list(self.param_config.keys())
                for k in keys:
                    kn = str(k).lower().lstrip('i').replace('_', '')
                    # match patterns like ng{n}, x0g{n}, gg{n}
                    import re as _re
                    m = _re.match(r'^(?:ng|x0g|gg)(\d+)$', kn)
                    if m:
                        idx = int(m.group(1))
                        if idx > count:
                            try:
                                del self.param_config[k]
                            except Exception:
                                pass
            except Exception:
                pass
        except Exception:
            pass

    def _ensure_lorentz_params(self):
        """Add default Lorentzian parameters to `self.param_config` if missing.

        Parameters added: `Nl` (amplitude), `x0l` (center), `gl` (width).
        Supports multiple components when `self.lorentz_count_spin` is present
        (adds suffixed parameters `Nl1`, `x0l1`, `gl1`, ...).
        """
        try:
            if not isinstance(self.param_config, dict):
                self.param_config = {}
            # Number of lorentz components requested (default 1)
            try:
                count = int(self.lorentz_count_spin.value()) if hasattr(self, 'lorentz_count_spin') else 1
            except Exception:
                count = 1
            # sensible defaults and bounds for each lorentz component
            for i in range(1, count + 1):
                key_area = f'Nl{i}'
                key_x0 = f'x0l{i}'
                key_gl = f'gl{i}'
                defaults = {
                    key_area: {'value': 0.1 if i == 1 else 0.0, 'vary': True, 'min': 0.0, 'max': 1.0},
                    key_x0: {'value': 480.0, 'vary': True, 'min': 400.0, 'max': 500.0},
                    key_gl: {'value': 10.0, 'vary': True, 'min': 1.0, 'max': 200.0},
                }
                for k, v in defaults.items():
                    if k not in self.param_config:
                        self.param_config[k] = v
            # Migrate legacy unsuffixed lorentz keys to the new indexed names
            try:
                if 'N_l' in self.param_config and 'Nl1' not in self.param_config:
                    try:
                        self.param_config['Nl1'] = self.param_config.pop('N_l')
                    except Exception:
                        pass
                if 'Nl' in self.param_config and 'Nl1' not in self.param_config:
                    try:
                        self.param_config['Nl1'] = self.param_config.pop('Nl')
                    except Exception:
                        pass
                if 'x0l' in self.param_config and 'x0l1' not in self.param_config:
                    try:
                        self.param_config['x0l1'] = self.param_config.pop('x0l')
                    except Exception:
                        pass
                if 'gl' in self.param_config and 'gl1' not in self.param_config:
                    try:
                        self.param_config['gl1'] = self.param_config.pop('gl')
                    except Exception:
                        pass
            except Exception:
                pass
            # Remove any higher-indexed lorentz params if the count was reduced
            try:
                keys = list(self.param_config.keys())
                for k in keys:
                    kn = str(k).lower().lstrip('i').replace('_', '')
                    import re as _re
                    m = _re.match(r'^(?:nl|x0l|gl)(\d+)$', kn)
                    if m:
                        idx = int(m.group(1))
                        if idx > count:
                            try:
                                del self.param_config[k]
                            except Exception:
                                pass
            except Exception:
                pass
        except Exception:
            pass

    def _toggle_lorentz(self, checked: bool):
        try:
            if checked:
                try:
                    self._ensure_lorentz_params()
                except Exception:
                    pass
                try:
                    if hasattr(self, 'lorentz_count_spin'):
                        self.lorentz_count_spin.setEnabled(True)
                except Exception:
                    pass
            else:
                # remove lorentzian params if present
                try:
                    keys = list(self.param_config.keys())
                    for k in keys:
                        kn = str(k).lower()
                        kns = kn.lstrip('i').replace('_', '')
                        if kns.startswith('nl') or kns.startswith('x0l') or kns.startswith('gl'):
                            try:
                                del self.param_config[k]
                            except Exception:
                                pass
                except Exception:
                    pass
                try:
                    if hasattr(self, 'lorentz_count_spin'):
                        self.lorentz_count_spin.setEnabled(False)
                except Exception:
                    pass
            try:
                self._populate_param_table()
            except Exception:
                pass
        except Exception:
            pass

    def _toggle_gaussian(self, checked: bool):
        try:
            if checked:
                try:
                    self._ensure_gaussian_params()
                except Exception:
                    pass
                try:
                    if hasattr(self, 'gauss_count_spin'):
                        self.gauss_count_spin.setEnabled(True)
                except Exception:
                    pass
            else:
                # remove gaussian params if present
                # delete any gaussian-related keys (ng, ng1, x0, x01, gg, gg1, ...)
                try:
                    keys = list(self.param_config.keys())
                    for k in keys:
                        kn = str(k).lower()
                        kns = kn.lstrip('i').replace('_', '')
                        if kns.startswith('ng') or kns.startswith('x0g') or (kns.startswith('x0') and not kns.startswith('x0l')) or kns.startswith('gg'):
                            try:
                                del self.param_config[k]
                            except Exception:
                                pass
                except Exception:
                    pass
                try:
                    if hasattr(self, 'gauss_count_spin'):
                        self.gauss_count_spin.setEnabled(False)
                except Exception:
                    pass
            try:
                self._populate_param_table()
            except Exception:
                pass
        except Exception:
            pass

    def _apply_fit_range(self):
        try:
            # validate and refresh preview to show highlighted range
            if hasattr(self, 'fit_xmin_spin') and hasattr(self, 'fit_xmax_spin'):
                x0 = int(self.fit_xmin_spin.value())
                x1 = int(self.fit_xmax_spin.value())
                if x0 > x1:
                    # swap to sensible order
                    self.fit_xmin_spin.setValue(int(x1))
                    self.fit_xmax_spin.setValue(int(x0))
            # re-render preview to show the selected range shading
            self._render_preview(y_model=getattr(self, 'last_y_model', None))
        except Exception:
            pass

    def normalize_data(self):
        try:
            if self.iw is None or self.y is None:
                QtWidgets.QMessageBox.warning(self, 'Warning', 'Load data first')
                return
            # Ask whether to normalize the full spectrum or use selected range
            use_full = False
            try:
                dlg = QtWidgets.QMessageBox(self)
                dlg.setWindowTitle('Normalize Range')
                dlg.setText('Normalize full spectrum or using the selected range?')
                full_btn = dlg.addButton('Full spectrum', QtWidgets.QMessageBox.AcceptRole)
                sel_btn = dlg.addButton('Selected range', QtWidgets.QMessageBox.AcceptRole)
                cancel_btn = dlg.addButton(QtWidgets.QMessageBox.Cancel)
                dlg.exec()
                clicked = dlg.clickedButton()
                if clicked == cancel_btn:
                    return
                elif clicked == full_btn:
                    use_full = True
                else:
                    use_full = False
            except Exception:
                use_full = True

            y_arr = np.asarray(self.y, dtype=float)
            if use_full:
                mask = np.ones_like(y_arr, dtype=bool)
            else:
                try:
                    x_arr = np.asarray(self.iw, dtype=float)
                    x0 = float(self.fit_xmin_spin.value()) if hasattr(self, 'fit_xmin_spin') else float(np.min(x_arr))
                    x1 = float(self.fit_xmax_spin.value()) if hasattr(self, 'fit_xmax_spin') else float(np.max(x_arr))
                    if x0 > x1:
                        x0, x1 = x1, x0
                    mask = (x_arr >= x0) & (x_arr <= x1)
                    if not np.any(mask):
                        QtWidgets.QMessageBox.warning(self, 'Warning', 'Selected range contains no data points; nothing to normalize')
                        return
                except Exception:
                    mask = np.ones_like(y_arr, dtype=bool)

            ymin = float(np.min(y_arr[mask]))
            ymax = float(np.max(y_arr[mask]))
            if ymax == ymin:
                QtWidgets.QMessageBox.warning(self, 'Warning', 'Selected data has zero dynamic range; cannot normalize')
                return
            # Apply scaling computed from the selected region to the whole array
            new_y = (y_arr - ymin) / (ymax - ymin)
            self.y = new_y
            # If we have a last model displayed, scale it the same way so overlay stays useful
            try:
                if getattr(self, 'last_y_model', None) is not None:
                    ly = np.asarray(self.last_y_model, dtype=float)
                    if ly.size == y_arr.size:
                        self.last_y_model = (ly - ymin) / (ymax - ymin)
                    else:
                        self.last_y_model = None
            except Exception:
                self.last_y_model = None
            # Update plot
            try:
                self._render_preview(y_model=self.last_y_model if getattr(self, 'last_y_model', None) is not None else None)
                if hasattr(self, 'rescale_cb') and self.rescale_cb.isChecked():
                    self._rescale_plot_y()
                try:
                    self.fit_status.setPlainText('Data normalized to [0,1]')
                except Exception:
                    pass
            except Exception:
                pass
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, 'Error', f'Normalization failed: {e}')


def main():
    try:
        QLocale.setDefault(QLocale.c())
    except Exception:
        pass
    app = QApplication([])
    win = MainWindow()
    win.show()
    app.exec()


if __name__ == '__main__':
    main()
