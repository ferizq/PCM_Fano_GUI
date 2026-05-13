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

from .io import load_data, load_param_config
from .fitting import fit_with_lmfit, fit_with_scipy, _build_free_params, _params_from_vector
from .model import model as compute_model


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

        self.btn_step = QPushButton('Step Fit')
        self.btn_fit = QPushButton('Fit (full)')
        self.btn_cancel = QPushButton('Cancel')
        left_l.addWidget(self.btn_step)
        left_l.addWidget(self.btn_fit)
        left_l.addWidget(self.btn_cancel)

        # Integrator & backend controls
        # Controls arranged in two rows so the left pane can be narrowed
        ctrl_container = QWidget()
        ctrl_v = QVBoxLayout(ctrl_container)
        ctrl_v.setContentsMargins(0, 0, 0, 0)

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
        ctrl1.addWidget(QLabel('LM method:'))
        self.lm_method_combo = QComboBox()
        self.lm_method_combo.addItems(['least_squares', 'leastsq', 'nelder', 'powell', 'lbfgsb'])
        ctrl1.addWidget(self.lm_method_combo)
        ctrl1.addWidget(QLabel('Max evals:'))
        self.maxeval_spin = QSpinBox()
        self.maxeval_spin.setRange(1, 20000000)
        self.maxeval_spin.setSingleStep(100)
        self.maxeval_spin.setValue(2000)
        ctrl1.addWidget(self.maxeval_spin)
        ctrl1.addWidget(QLabel('Autoscale'))
        self.autoscale_cb = QCheckBox()
        self.autoscale_cb.setChecked(True)
        ctrl1.addWidget(self.autoscale_cb)
        ctrl_v.addWidget(ctrl_row1)

        ctrl_row2 = QWidget()
        ctrl2 = QHBoxLayout(ctrl_row2)
        ctrl2.setContentsMargins(0, 0, 0, 0)
        ctrl2.addWidget(QLabel('Grid size:'))
        self.grid_spin = QSpinBox()
        self.grid_spin.setRange(10, 5000)
        self.grid_spin.setSingleStep(10)
        self.grid_spin.setValue(400)
        ctrl2.addWidget(self.grid_spin)
        ctrl2.addWidget(QLabel('ik_min:'))
        self.ik_min_spin = QDoubleSpinBox()
        self.ik_min_spin.setRange(-10.0, 10.0)
        self.ik_min_spin.setSingleStep(0.01)
        self.ik_min_spin.setValue(0.0)
        ctrl2.addWidget(self.ik_min_spin)
        ctrl2.addWidget(QLabel('ik_max:'))
        self.ik_max_spin = QDoubleSpinBox()
        self.ik_max_spin.setRange(-10.0, 10.0)
        self.ik_max_spin.setSingleStep(0.01)
        self.ik_max_spin.setValue(1.0)
        ctrl2.addWidget(self.ik_max_spin)
        ctrl2.addWidget(QLabel('Preview grid:'))
        self.preview_spin = QSpinBox()
        self.preview_spin.setRange(10, 2000)
        self.preview_spin.setValue(100)
        ctrl2.addWidget(self.preview_spin)
        self.btn_preview = QPushButton('Preview')
        ctrl2.addWidget(self.btn_preview)
        ctrl_v.addWidget(ctrl_row2)

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

        try:
            y_model = compute_model(self.iw, params, integrator=integrator, grid_size=grid_size, ik_min=ik_min, ik_max=ik_max)
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
                entry['value'] = float(val)
            except Exception:
                entry['value'] = float('nan')
            entry['vary'] = bool(cb.isChecked()) if cb is not None else True
            try:
                entry['min'] = float(mn)
            except Exception:
                entry['min'] = None
            try:
                entry['max'] = float(mx)
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
            'grid_size': int(self.grid_spin.value()) if hasattr(self, 'grid_spin') else 400,
            'ik_min': float(self.ik_min_spin.value()) if hasattr(self, 'ik_min_spin') else 0.0,
            'ik_max': float(self.ik_max_spin.value()) if hasattr(self, 'ik_max_spin') else 1.0,
            'autoscale': bool(self.autoscale_cb.isChecked()) if hasattr(self, 'autoscale_cb') else True,
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

        self.fit_worker = FitWorker(self.iw, self.y, self.param_config, backend=backend, integrator_opts=integrator_opts, minimizer_opts=minimizer_opts)
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
            for name in free_names:
                pval = float(current_params.get(name, 0.0))
                dp = abs(pval * 0.01) if abs(pval) > 0 else 1e-6
                p_plus = dict(current_params)
                p_minus = dict(current_params)
                p_plus[name] = pval + dp
                p_minus[name] = pval - dp
                y_plus = compute_model(self.iw, p_plus, integrator='grid', grid_size=test_grid)
                y_minus = compute_model(self.iw, p_minus, integrator='grid', grid_size=test_grid)
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
                            y_plus = compute_model(self.iw, params_plus, integrator='grid', grid_size=fd_grid)
                            y_minus = compute_model(self.iw, params_minus, integrator='grid', grid_size=fd_grid)
                        except Exception:
                            y_plus = compute_model(self.iw, params_plus)
                            y_minus = compute_model(self.iw, params_minus)
                        deriv = (np.asarray(y_plus) - np.asarray(y_minus)) / (2.0 * dp)
                        Jfd[:, j] = deriv
                    JTJ = Jfd.T.dot(Jfd)
                    params_base = _params_from_vector(fnames, pvec, self.param_config)
                    try:
                        y_base = compute_model(self.iw, params_base, integrator='grid', grid_size=fd_grid)
                    except Exception:
                        y_base = compute_model(self.iw, params_base)
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

        # Update plots
        y_model = out.get('y_model')
        if y_model is not None:
            self.last_y_model = np.asarray(y_model)
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
        main_fig.update_layout(title='Data and Fit', margin=dict(l=40, r=10, t=40, b=40), height=520, showlegend=False)

        resid_arr = np.asarray(self.y) - (np.asarray(y_model) if y_model is not None else np.zeros_like(np.asarray(self.y)))
        resid = list(map(float, resid_arr))
        res_fig = go.Figure()
        res_fig.add_trace(go.Scatter(x=x_data, y=resid, mode='markers', name='residuals', marker=dict(size=4)))
        res_fig.update_layout(title='Residuals', margin=dict(l=40, r=10, t=30, b=30), height=240, showlegend=False)

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
            with open(path, 'w', encoding='utf-8') as f:
                if getattr(self, 'last_y_model', None) is not None:
                    for x, ym in zip(self.iw, self.last_y_model):
                        f.write(f"{x}\t{ym}\n")
                else:
                    for x, y in zip(self.iw, self.y):
                        f.write(f"{x}\t{y}\n")
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
            plot_fit(self.iw, self.y, self.last_y_model if getattr(self, 'last_y_model', None) is not None else np.zeros_like(self.y),
                     params=params, param_errs=errs, r2=r2,
                     output_html=path, show=False, converged=converged, convergence_message=msg,
                     assets_dir=self.assets_dir)
            QtWidgets.QMessageBox.information(self, 'Exported', f'HTML report saved to {path}')
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, 'Error', f'Failed to export HTML: {e}')


def main():
    app = QApplication([])
    win = MainWindow()
    win.show()
    app.exec()


if __name__ == '__main__':
    main()
