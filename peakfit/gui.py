"""Minimal PySide6 GUI scaffold for interactive peak fitting.

This module provides a basic desktop UI using PySide6 with a QWebEngineView
to render Plotly-based plots (data+fit and residuals), a parameter table,
and buttons to load/save files and run/cancel fits.

This is an initial scaffold that reuses `peakfit.io` and `peakfit.fitting`.
Ensure `PySide6` and `PySide6-QtWebEngine` are installed before running.
"""
from __future__ import annotations

import json
import os
import tempfile
from typing import Dict, Any

import numpy as np
import plotly.graph_objects as go
import copy

try:
    from PySide6 import QtCore, QtWidgets
    from PySide6.QtCore import QUrl
    from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                                   QHBoxLayout, QPushButton, QFileDialog, QTableWidget,
                                   QTableWidgetItem, QCheckBox, QProgressBar, QLabel,
                                   QHeaderView, QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox,
                                   QPlainTextEdit)
    from PySide6.QtGui import QDoubleValidator
    from PySide6.QtWebEngineWidgets import QWebEngineView
    from PySide6.QtWebEngineCore import QWebEnginePage
except Exception as e:
    raise ImportError("PySide6 and QtWebEngine are required to run the GUI: " + str(e))

from .io import load_data, load_param_config, normalize_number_string
from .fitting import fit_with_lmfit, fit_with_scipy, _build_free_params, _params_from_vector
from .model import model as compute_model

# Built-in default parameter sets used to initialize the GUI when the user
# selects a kernel. These mirror the values used in the test fixtures /
# `params_gauss.json` (embedded here so the GUI can self-initialize).
GAUSS_DEFAULT_PARAMS = {
    'alpha': {'value': 9.67, 'vary': False, 'min': 1.0, 'max': 100.0},
    'C':   {'value': 171400.0, 'vary': False, 'min': 171400.0, 'max': 171410.0},
    'D':   {'value': 100000.0, 'vary': False, 'min': 100000.0, 'max': 100010.0},
    'b':   {'value': 0.0, 'vary': False, 'min': -10.0, 'max': 10.0},
    'g0':  {'value': 4.5, 'vary': False, 'min': 1.0, 'max': 10.0},
    'q':   {'value': -1.0e12, 'vary': False, 'min': -1.0e12, 'max': 1.0e12},
    'a':   {'value': 0.5431, 'vary': False, 'min': 0.54, 'max': 0.55},
    'L':   {'value': 10.0, 'vary': False, 'min': 5.0, 'max': 50.0},
    'N':   {'value': 1.0, 'vary': True, 'min': 1e-20, 'max': 1e+20},
    'y0':  {'value': 0.0, 'vary': True, 'min': -10000.0, 'max': 10000.0},
}

BESSEL_DEFAULT_PARAMS = {
    'C':   {'value': 171400.0, 'vary': False, 'min': 171400.0, 'max': 171410.0},
    'D':   {'value': 100000.0, 'vary': False, 'min': 100000.0, 'max': 100010.0},
    'b':   {'value': 0.0, 'vary': False, 'min': -10.0, 'max': 10.0},
    'g0':  {'value': 4.5, 'vary': False, 'min': 1.0, 'max': 10.0},
    'q':   {'value': -1.0e12, 'vary': False, 'min': -1.0e12, 'max': 1.0e12},
    'a':   {'value': 0.5431, 'vary': False, 'min': 0.54, 'max': 0.55},
    'L':   {'value': 10.0, 'vary': False, 'min': 5.0, 'max': 50.0},
    'N':   {'value': 1.0, 'vary': True, 'min': 1e-20, 'max': 1e+20},
    'y0':  {'value': 0.0, 'vary': True, 'min': -10000.0, 'max': 10000.0},
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

        try:
            if self.backend == 'lmfit':
                fitted, errs, result, y_model, r2, converged, message, nfev = fit_with_lmfit(
                    self.iw, self.y, self.param_config,
                    integrator_opts=self.integrator_opts, minimizer_opts=self.minimizer_opts,
                    progress_callback=progress_cb)
                out = dict(fitted=fitted, errs=errs, result=result, y_model=y_model, r2=r2, converged=converged, message=message, nfev=nfev)
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
                out = dict(fitted=params_full, errs=errs_full, popt=popt, pcov=pcov, y_model=y_model, r2=r2, converged=converged, message=message, nfev=nfev)
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

        self._init_ui()

    def _init_ui(self):
        # Use a QSplitter so the user can resize the control pane vs the plots.
        # Left pane: controls + parameter table
        left = QWidget()
        try:
            left.setMinimumWidth(80)
        except Exception:
            pass
        left_l = QVBoxLayout(left)

        btn_row = QWidget()
        brl = QHBoxLayout(btn_row)
        self.btn_load_txt = QPushButton('Load TXT')
        self.btn_load_json = QPushButton('Load JSON')
        self.btn_save_fitted = QPushButton('Save fitted TXT')
        self.btn_export_params = QPushButton('Export params JSON')
        self.btn_export_html = QPushButton('Export HTML')
        brl.addWidget(self.btn_load_txt)
        brl.addWidget(self.btn_load_json)
        brl.addWidget(self.btn_save_fitted)
        brl.addWidget(self.btn_export_params)
        brl.addWidget(self.btn_export_html)
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
        left_l.addWidget(btn_fit_row)

        # Integrator & backend controls
        # Controls arranged in two rows so the left pane can be narrowed
        ctrl_container = QWidget()
        ctrl_v = QVBoxLayout(ctrl_container)
        ctrl_v.setContentsMargins(0, 0, 0, 0)

        # Row 1: backend + integrator
        ctrl_row1 = QWidget()
        ctrl1 = QHBoxLayout(ctrl_row1)
        ctrl1.setContentsMargins(0, 0, 0, 0)
        ctrl1.addWidget(QLabel('Backend:'))
        self.backend_combo = QComboBox()
        self.backend_combo.addItems(['lmfit', 'scipy'])
        ctrl1.addWidget(self.backend_combo)
        ctrl1.addWidget(QLabel('Integrator:'))
        self.integrator_combo = QComboBox()
        self.integrator_combo.addItems(['grid', 'quad'])
        ctrl1.addWidget(self.integrator_combo)
        ctrl_v.addWidget(ctrl_row1)

        # Row 2: kernel + fit range controls
        ctrl_row1b = QWidget()
        ctrl1b = QHBoxLayout(ctrl_row1b)
        ctrl1b.setContentsMargins(0, 0, 0, 0)
        ctrl1b.addWidget(QLabel('Kernel:'))
        self.kernel_combo = QComboBox()
        self.kernel_combo.addItems(['PCM_Fano_Bessel', 'PCM_Fano_Gauss'])
        ctrl1b.addWidget(self.kernel_combo)
        self.btn_kernel_info = QPushButton('Info')
        ctrl1b.addWidget(self.btn_kernel_info)
        ctrl1b.addWidget(QLabel('Fit x start:'))
        self.fit_xmin_spin = QSpinBox()
        self.fit_xmin_spin.setRange(-999999999, 999999999)
        self.fit_xmin_spin.setSingleStep(1)
        self.fit_xmin_spin.setValue(0)
        ctrl1b.addWidget(self.fit_xmin_spin)
        ctrl1b.addWidget(QLabel('Fit x end:'))
        self.fit_xmax_spin = QSpinBox()
        self.fit_xmax_spin.setRange(-999999999, 999999999)
        self.fit_xmax_spin.setSingleStep(1)
        self.fit_xmax_spin.setValue(0)
        ctrl1b.addWidget(self.fit_xmax_spin)
        self.btn_apply_range = QPushButton('Apply Range')
        ctrl1b.addWidget(self.btn_apply_range)
        # Option to rescale main plot to the selected fit range
        self.rescale_cb = QCheckBox('Rescale plot to range')
        self.rescale_cb.setChecked(False)
        self.rescale_cb.setToolTip('When checked, zoom the main plot to the selected fit x-range')
        ctrl1b.addWidget(self.rescale_cb)
        ctrl_v.addWidget(ctrl_row1b)

        # Row 3: LM method, max evals, autoscale
        ctrl_row2 = QWidget()
        ctrl2 = QHBoxLayout(ctrl_row2)
        ctrl2.setContentsMargins(0, 0, 0, 0)
        ctrl2.addWidget(QLabel('LM method:'))
        self.lm_method_combo = QComboBox()
        self.lm_method_combo.addItems(['least_squares', 'leastsq', 'nelder', 'powell', 'lbfgsb'])
        ctrl2.addWidget(self.lm_method_combo)
        ctrl2.addWidget(QLabel('Max evals:'))
        self.maxeval_spin = QSpinBox()
        self.maxeval_spin.setRange(1, 20000000)
        self.maxeval_spin.setSingleStep(100)
        self.maxeval_spin.setValue(10000)
        ctrl2.addWidget(self.maxeval_spin)
        ctrl2.addWidget(QLabel('Autoscale'))
        self.autoscale_cb = QCheckBox()
        self.autoscale_cb.setChecked(True)
        ctrl2.addWidget(self.autoscale_cb)
        ctrl_v.addWidget(ctrl_row2)

        # Row 4: grid and preview options
        ctrl_row3 = QWidget()
        ctrl3 = QHBoxLayout(ctrl_row3)
        ctrl3.setContentsMargins(0, 0, 0, 0)
        ctrl3.addWidget(QLabel('Grid size:'))
        self.grid_spin = QSpinBox()
        self.grid_spin.setRange(10, 10000)
        self.grid_spin.setSingleStep(10)
        self.grid_spin.setValue(4000)
        ctrl3.addWidget(self.grid_spin)
        ctrl3.addWidget(QLabel('ik_min:'))
        self.ik_min_spin = QDoubleSpinBox()
        self.ik_min_spin.setRange(-10.0, 10.0)
        self.ik_min_spin.setSingleStep(0.01)
        self.ik_min_spin.setValue(0.0)
        ctrl3.addWidget(self.ik_min_spin)
        ctrl3.addWidget(QLabel('ik_max:'))
        self.ik_max_spin = QDoubleSpinBox()
        self.ik_max_spin.setRange(-10.0, 10.0)
        self.ik_max_spin.setSingleStep(0.01)
        self.ik_max_spin.setValue(1.0)
        ctrl3.addWidget(self.ik_max_spin)
        ctrl3.addWidget(QLabel('Preview grid:'))
        self.preview_spin = QSpinBox()
        self.preview_spin.setRange(10, 2000)
        self.preview_spin.setValue(100)
        ctrl3.addWidget(self.preview_spin)
        self.btn_preview = QPushButton('Preview')
        ctrl3.addWidget(self.btn_preview)
        self.btn_normalize = QPushButton('Normalize')
        ctrl3.addWidget(self.btn_normalize)
        ctrl_v.addWidget(ctrl_row3)

        left_l.addWidget(ctrl_container)
        # Tooltips for quick help
        try:
            self.backend_combo.setToolTip('Choose fitting backend: lmfit (rich features) or scipy (curve_fit)')
            self.integrator_combo.setToolTip('Integrator: "grid" is fast/approximate, "quad" is accurate but slow')
            self.lm_method_combo.setToolTip('Minimizer method used by lmfit; try least_squares or leastsq')
            self.maxeval_spin.setToolTip('Maximum function evaluations for a full fit')
            self.autoscale_cb.setToolTip('Autoscale initial guesses for N and y0 from the loaded data')
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

        # Right pane: plot (QWebEngineView)
        # use a debug page to capture JS console messages from the embedded plot
        class _DebugPage(QWebEnginePage):
            def javaScriptConsoleMessage(self, level, msg, line, sourceID):
                try:
                    print(f"JS[{level}] {msg} (line {line} source {sourceID})", flush=True)
                except Exception:
                    pass

        self.web = QWebEngineView()
        dbg_page = _DebugPage(self.web)
        self.web.setPage(dbg_page)

        from PySide6.QtWidgets import QSplitter
        splitter = QSplitter(QtCore.Qt.Horizontal)
        splitter.addWidget(left)
        splitter.addWidget(self.web)
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
        self.btn_export_html.clicked.connect(self.export_html_report)
        self.btn_step.clicked.connect(lambda: self.start_fit(mode='step'))
        self.btn_fit.clicked.connect(lambda: self.start_fit(mode='full'))
        self.btn_cancel.clicked.connect(self.cancel_fit)
        self.btn_preview.clicked.connect(self.preview_model)
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
            y_model = compute_model(self.iw, params, integrator=integrator, grid_size=grid_size, ik_min=ik_min, ik_max=ik_max, kernel=kernel)
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
            val_edit = QLineEdit(val_str)
            val_edit.setValidator(QDoubleValidator())
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
            min_edit = QLineEdit(mn_str)
            min_edit.setValidator(QDoubleValidator())
            self.table.setCellWidget(row, 4, min_edit)
            # Max (editable numeric)
            mx = info.get('max', '')
            try:
                mx_str = sci_format(float(mx), decimals=4) if mx != '' and mx is not None else ''
            except Exception:
                mx_str = str(mx)
            max_edit = QLineEdit(mx_str)
            max_edit.setValidator(QDoubleValidator())
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
                vs = normalize_number_string(val)
                entry['value'] = float(vs)
            except Exception:
                entry['value'] = float('nan')
            entry['vary'] = bool(cb.isChecked()) if cb is not None else True
            try:
                ms = normalize_number_string(mn)
                entry['min'] = float(ms)
            except Exception:
                entry['min'] = None
            try:
                xs = normalize_number_string(mx)
                entry['max'] = float(xs)
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
            # Always use the parameters shown in the table as the starting
            # guess for both step and full fits (do not autoscale p0 here).
            'autoscale': False,
            'kernel': self.kernel_combo.currentText() if hasattr(self, 'kernel_combo') else 'PCM_Fano_Bessel',
        }

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
            self.fit_status.setPlainText('Running')
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
        try:
            r2s = ('R2=' + ('{:.4g}'.format(float(r2)) if r2 is not None else 'nan'))
        except Exception:
            r2s = 'R2=nan'
        try:
            if converged:
                self.fit_status.setPlainText(f'Converged (nfev={nfev}, {r2s})')
            else:
                self.fit_status.setPlainText(f'Not converged: {msg} (nfev={nfev}, {r2s})')
        except Exception:
            pass
        # Best-effort sensitivity check: flag free parameters that hardly change
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
                try:
                    y_plus = compute_model(self.iw, p_plus, integrator='grid', grid_size=test_grid, kernel=kern)
                    y_minus = compute_model(self.iw, p_minus, integrator='grid', grid_size=test_grid, kernel=kern)
                except Exception:
                    y_plus = compute_model(self.iw, p_plus, kernel=kern)
                    y_minus = compute_model(self.iw, p_minus, kernel=kern)
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
        if pcov is None:
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
                            y_plus = compute_model(self.iw, params_plus, integrator='grid', grid_size=fd_grid, kernel=kern)
                            y_minus = compute_model(self.iw, params_minus, integrator='grid', grid_size=fd_grid, kernel=kern)
                        except Exception:
                            y_plus = compute_model(self.iw, params_plus, kernel=kern)
                            y_minus = compute_model(self.iw, params_minus, kernel=kern)
                        deriv = (np.asarray(y_plus) - np.asarray(y_minus)) / (2.0 * dp)
                        Jfd[:, j] = deriv
                    JTJ = Jfd.T.dot(Jfd)
                    params_base = _params_from_vector(fnames, pvec, self.param_config)
                    try:
                        y_base = compute_model(self.iw, params_base, integrator='grid', grid_size=fd_grid, kernel=kern)
                    except Exception:
                        y_base = compute_model(self.iw, params_base, kernel=kern)
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
                y_model_full = compute_model(self.iw, fitted_for_plot, integrator=integrator, grid_size=grid_sz, ik_min=float(self.ik_min_spin.value()) if hasattr(self, 'ik_min_spin') else 0.0, ik_max=float(self.ik_max_spin.value()) if hasattr(self, 'ik_max_spin') else 1.0, kernel=kern)
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
        # Build Plotly figures and render in QWebEngineView
        if self.iw is None or self.y is None:
            return
        main_fig = go.Figure()
        x_data = list(map(float, np.asarray(self.iw)))
        y_data = list(map(float, np.asarray(self.y)))
        main_fig.add_trace(go.Scatter(x=x_data, y=y_data, mode='markers', name='data', marker=dict(size=6)))
        if y_model is not None:
            y_model_list = list(map(float, np.asarray(y_model)))
            main_fig.add_trace(go.Scatter(x=x_data, y=y_model_list, mode='lines', name='fit', line=dict(width=2)))
        # If a fit-range is specified, highlight it on the main plot
        try:
            if hasattr(self, 'fit_xmin_spin') and hasattr(self, 'fit_xmax_spin'):
                x0 = float(self.fit_xmin_spin.value())
                x1 = float(self.fit_xmax_spin.value())
                if x0 < x1:
                    main_fig.add_vrect(x0=x0, x1=x1, fillcolor='LightSalmon', opacity=0.2, layer='below', line_width=0)
        except Exception:
            pass
        main_fig.update_layout(title='Data and Fit', margin=dict(l=40, r=10, t=40, b=40), height=520, showlegend=False)

        # Determine selected fit-range mask (if any) and whether to rescale
        try:
            x_arr = np.asarray(self.iw, dtype=float)
            y_arr = np.asarray(self.y, dtype=float)
        except Exception:
            x_arr = np.asarray(self.iw)
            y_arr = np.asarray(self.y)

        mask = None
        try:
            if hasattr(self, 'fit_xmin_spin') and hasattr(self, 'fit_xmax_spin'):
                x0 = float(self.fit_xmin_spin.value())
                x1 = float(self.fit_xmax_spin.value())
                if x0 > x1:
                    x0, x1 = x1, x0
                mask = (x_arr >= x0) & (x_arr <= x1)
        except Exception:
            mask = None

        # Residuals: restrict to selected mask if present
        res_fig = go.Figure()
        try:
            if y_model is not None:
                y_model_arr = np.asarray(y_model)
                resid_arr = y_arr - y_model_arr
                if mask is not None:
                    res_x = list(map(float, x_arr[mask]))
                    res_y = list(map(float, resid_arr[mask]))
                else:
                    res_x = list(map(float, x_arr))
                    res_y = list(map(float, resid_arr))
                if len(res_x) > 0:
                    res_fig.add_trace(go.Scatter(x=res_x, y=res_y, mode='markers', name='residuals', marker=dict(size=4)))
            else:
                # no model: plot nothing or empty residuals within mask
                if mask is not None:
                    res_x = list(map(float, x_arr[mask]))
                    res_y = [0.0] * len(res_x)
                    if len(res_x) > 0:
                        res_fig.add_trace(go.Scatter(x=res_x, y=res_y, mode='markers', name='residuals', marker=dict(size=4)))
        except Exception:
            try:
                # fallback: plot full-range zero residuals
                res_fig.add_trace(go.Scatter(x=x_data, y=[0.0] * len(x_data), mode='markers', name='residuals', marker=dict(size=4)))
            except Exception:
                pass

        # If requested, rescale main plot to selected x-range and y-limits of masked data/model
        try:
            if hasattr(self, 'rescale_cb') and self.rescale_cb.isChecked() and mask is not None and np.any(mask):
                # x-limits
                main_fig.update_xaxes(range=[float(x0), float(x1)])
                # y-limits based on data/model inside mask
                yvals = []
                try:
                    yvals.extend(list(y_arr[mask]))
                except Exception:
                    pass
                try:
                    if y_model is not None:
                        yvals.extend(list(y_model_arr[mask]))
                except Exception:
                    pass
                if len(yvals) > 0:
                    ymin = float(np.min(yvals))
                    ymax = float(np.max(yvals))
                    if ymin == ymax:
                        # small padding when flat
                        ymin -= 1e-6
                        ymax += 1e-6
                    pad = max(1e-6, 0.05 * (ymax - ymin))
                    main_fig.update_yaxes(range=[ymin - pad, ymax + pad])
        except Exception:
            pass

        res_fig.update_layout(title='Residuals', margin=dict(l=40, r=10, t=30, b=30), height=240, showlegend=False)
        # Force residual x-axis to the selected range when mask present
        try:
            if mask is not None and np.any(mask):
                res_fig.update_xaxes(range=[float(x0), float(x1)])
        except Exception:
            pass

        main_div = main_fig.to_html(full_html=False, include_plotlyjs=False)
        resid_div = res_fig.to_html(full_html=False, include_plotlyjs=False)

        # Ensure explicit pixel heights for the embedded plot containers. The
        # default fragment uses `height:100%` which collapses inside our page
        # and can make the traces invisible. Replace that with the figure's
        # explicit height (pixels) so the QWebEngineView renders markers.
        try:
            main_h = int(main_fig.layout.height) if getattr(main_fig.layout, 'height', None) is not None else 520
        except Exception:
            main_h = 520
        try:
            resid_h = int(res_fig.layout.height) if getattr(res_fig.layout, 'height', None) is not None else 240
        except Exception:
            resid_h = 240
        main_div = main_div.replace('height:100%;', f'height:{main_h}px;')
        resid_div = resid_div.replace('height:100%;', f'height:{resid_h}px;')

        # Math block (reuse short description)
        math_block = '<div><strong>Model:</strong> see documentation</div>'

        # Compose final HTML: inline local plotly if available (avoids race 'Plotly is not defined')
        plotly_head = None
        katex_css = 'https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css'
        katex_js = 'https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.js'
        katex_autorender = 'https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/contrib/auto-render.min.js'
        base = QUrl('')
        if self.assets_dir:
            local_plotly = os.path.join(self.assets_dir, 'plotly.min.js')
            if os.path.isfile(local_plotly):
                try:
                    with open(local_plotly, 'r', encoding='utf-8') as f:
                        plotly_js = f.read()
                    plotly_head = f"<script type=\"text/javascript\">{plotly_js}</script>"
                    katex_css = 'katex.min.css'
                    katex_js = 'katex.min.js'
                    katex_autorender = 'auto-render.min.js'
                    base = QUrl.fromLocalFile(os.path.abspath(self.assets_dir) + os.sep)
                except Exception:
                    plotly_head = None

        if plotly_head is None:
            plotly_head = '<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>'

        html = f"""<!doctype html>
<html>
  <head>
    <meta charset="utf-8">
    <title>PeakFit Preview</title>
    <link rel="stylesheet" href="{katex_css}">
    {plotly_head}
    <script defer src="{katex_js}"></script>
    <script defer src="{katex_autorender}"></script>
    <style> body {{ font-family: Arial, sans-serif; margin:10px; }} table {{ width:100%; border-collapse:collapse; }} td, th {{ border:1px solid #ddd; padding:6px; }}</style>
  </head>
    <body>
        <div class="main">{main_div}</div>
        <div class="resid">{resid_div}</div>
        <div class="math">{math_block}</div>
    </body>
</html>"""

        # remember last model for saving
        self.last_y_model = None if y_model is None else np.asarray(y_model)

        # Load HTML into view.
        self.web.setHtml(html, base)

    def save_fitted(self):
        # Save currently shown fit data (if available)
        if self.iw is None:
            QtWidgets.QMessageBox.warning(self, 'Warning', 'No data to save')
            return
        path, _ = QFileDialog.getSaveFileName(self, 'Save fitted data', 'fitted.txt', 'Text Files (*.txt);;All Files (*)')
        if not path:
            return
        try:
            # determine selected range mask (only export points inside selected fit range)
            try:
                x_arr = np.asarray(self.iw, dtype=float)
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

            # determine fitted values for the selected points
            y_fit = None
            if getattr(self, 'last_y_model', None) is not None:
                try:
                    y_model_arr = np.asarray(self.last_y_model, dtype=float)
                    if y_model_arr.size == x_arr.size:
                        y_fit = y_model_arr[mask]
                except Exception:
                    y_fit = None

            if y_fit is None:
                # try to recompute model from current param_config
                try:
                    params = {name: info.get('value', 0.0) for name, info in self.param_config.items()}
                    integrator = self.integrator_combo.currentText() if hasattr(self, 'integrator_combo') else 'grid'
                    grid_size = int(self.grid_spin.value()) if hasattr(self, 'grid_spin') else 4000
                    ik_min = float(self.ik_min_spin.value()) if hasattr(self, 'ik_min_spin') else 0.0
                    ik_max = float(self.ik_max_spin.value()) if hasattr(self, 'ik_max_spin') else 1.0
                    y_full = compute_model(self.iw, params, integrator=integrator, grid_size=grid_size, ik_min=ik_min, ik_max=ik_max, kernel=self.kernel_combo.currentText() if hasattr(self, 'kernel_combo') else 'PCM_Fano_Bessel')
                    y_fit = np.asarray(y_full, dtype=float)[mask]
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

                # column header
                f.write("# x\texperimental\tfitted\tresidual\n")
                # data rows (include residual = experimental - fitted)
                try:
                    resid_arr = (np.asarray(y_sel, dtype=float) - np.asarray(y_fit, dtype=float))
                except Exception:
                    resid_arr = [float('nan')] * len(x_sel)
                for xi, yi, yf, yr in zip(x_sel, y_sel, y_fit, resid_arr):
                    try:
                        f.write(f"{xi:.6g}\t{yi:.6g}\t{yf:.6g}\t{yr:.6g}\n")
                    except Exception:
                        f.write(f"{xi}\t{yi}\t{yf}\t{yr}\n")

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
        # Export a full HTML report using the plotting helper (if available)
        try:
            from .plotting import plot_fit
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, 'Error', f'Plotting helper not available: {e}')
            return

        if self.iw is None or self.y is None:
            QtWidgets.QMessageBox.warning(self, 'Warning', 'Load data first')
            return

        path, _ = QFileDialog.getSaveFileName(self, 'Export HTML report', 'fit_report.html', 'HTML Files (*.html);;All Files (*)')
        if not path:
            return

        # prepare parameters and errs from last fit if present
        params = None
        errs = None
        r2 = None
        converged = None
        msg = None
        if getattr(self, 'last_fit_out', None) is not None:
            out = self.last_fit_out
            params = out.get('fitted') or out.get('params') or None
            errs = out.get('errs') or None
            r2 = out.get('r2')
            converged = out.get('converged')
            msg = out.get('message')

        try:
            # Respect the selected fit range: slice data/model to the selected x-range
            try:
                x_arr = np.asarray(self.iw, dtype=float)
                y_arr = np.asarray(self.y, dtype=float)
                if hasattr(self, 'fit_xmin_spin') and hasattr(self, 'fit_xmax_spin'):
                    x0 = float(self.fit_xmin_spin.value())
                    x1 = float(self.fit_xmax_spin.value())
                    if x0 > x1:
                        x0, x1 = x1, x0
                    mask = (x_arr >= x0) & (x_arr <= x1)
                    if not np.any(mask):
                        QtWidgets.QMessageBox.warning(self, 'Warning', 'Selected range contains no data points; nothing to export')
                        return
                else:
                    mask = np.ones_like(x_arr, dtype=bool)
            except Exception:
                x_arr = np.asarray(self.iw)
                y_arr = np.asarray(self.y)
                mask = np.ones_like(x_arr, dtype=bool)

            iw_sel = x_arr[mask]
            y_sel = y_arr[mask]

            # Prepare parameters for plotting (prefer last fit output, fallback to param_config)
            params_plot = params if params is not None else ({name: info.get('value', 0.0) for name, info in self.param_config.items()} if isinstance(self.param_config, dict) else None)

            # Compute fitted model across full x grid and slice to match selection so plots align
            y_full = None
            kernel = self.kernel_combo.currentText() if hasattr(self, 'kernel_combo') else None
            integrator = self.integrator_combo.currentText() if hasattr(self, 'integrator_combo') else 'grid'
            grid_size = int(self.grid_spin.value()) if hasattr(self, 'grid_spin') else 4000
            ik_min = float(self.ik_min_spin.value()) if hasattr(self, 'ik_min_spin') else 0.0
            ik_max = float(self.ik_max_spin.value()) if hasattr(self, 'ik_max_spin') else 1.0
            try:
                if params_plot is not None:
                    y_full = compute_model(self.iw, params_plot, integrator=integrator, grid_size=grid_size, ik_min=ik_min, ik_max=ik_max, kernel=kernel)
            except Exception:
                y_full = None

            if y_full is None:
                # fallback to last_y_model when recompute fails
                if getattr(self, 'last_y_model', None) is not None and np.asarray(self.last_y_model).size == np.asarray(self.iw).size:
                    y_full = np.asarray(self.last_y_model)
                else:
                    y_full = np.zeros_like(x_arr)

            y_model_sel = np.asarray(y_full)[mask]

            plot_fit(iw_sel, y_sel, y_model_sel,
                     params=params_plot, param_errs=errs, r2=r2,
                     output_html=path, show=False, converged=converged, convergence_message=msg,
                     assets_dir=self.assets_dir, kernel=kernel)
            QtWidgets.QMessageBox.information(self, 'Exported', f'HTML report saved to {path}')
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, 'Error', f'Failed to export HTML: {e}')

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
        return self.param_config

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
            y_arr = np.asarray(self.y, dtype=float)
            ymin = float(np.min(y_arr))
            ymax = float(np.max(y_arr))
            if ymax == ymin:
                QtWidgets.QMessageBox.warning(self, 'Warning', 'Data has zero dynamic range; cannot normalize')
                return
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
                try:
                    self.fit_status.setPlainText('Data normalized to [0,1]')
                except Exception:
                    pass
            except Exception:
                pass
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, 'Error', f'Normalization failed: {e}')


def main():
    app = QApplication([])
    win = MainWindow()
    win.show()
    app.exec()


if __name__ == '__main__':
    main()
