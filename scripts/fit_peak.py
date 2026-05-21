#!/usr/bin/env python3
"""Simple CLI to fit a peak defined by an integral-based model.

Example:
    python scripts/fit_peak.py data.txt -c configs/default_params.json -o result.html
"""
import argparse
import json
import sys
import os
import threading
import queue
import time

from peakfit import io as io_module
from peakfit import fitting
from peakfit import plotting


def gui_collect_paths():
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox
    except Exception:
        return None
    # Create a visible temporary root so file dialogs have a taskbar entry
    root = tk.Tk()
    root.title('Peakfit — Select files')
    try:
        # Small window so it's not intrusive but still appears in taskbar
        root.geometry('320x48+200+200')
        root.update_idletasks()
        root.lift()
        root.attributes('-topmost', True)
    except Exception:
        pass

    data_path = filedialog.askopenfilename(parent=root, title="Select data file", filetypes=[("Text files", "*.txt;*.dat;*.csv"), ("All files", "*.*")])
    if not data_path:
        try:
            messagebox.showinfo("Cancelled", "No data file selected", parent=root)
        except Exception:
            pass
        try:
            root.destroy()
        except Exception:
            pass
        return None

    config_path = filedialog.askopenfilename(parent=root, title="Select parameter JSON (optional)", initialdir='configs', filetypes=[("JSON files", "*.json"), ("All files", "*.*")])
    if not config_path:
        config_path = 'configs/default_params.json'

    output_path = filedialog.asksaveasfilename(parent=root, title="Save output HTML", defaultextension='.html', initialfile='fit_result.html', filetypes=[("HTML files", "*.html")])
    if not output_path:
        output_path = 'fit_result.html'

    try:
        root.destroy()
    except Exception:
        pass

    return data_path, config_path, output_path


def main(argv=None):
    p = argparse.ArgumentParser(description='Fit integral-defined peak model to two-column data')
    p.add_argument('data', help='Two-column data file (X Y)')
    p.add_argument('-c', '--config', default='configs/default_params.json', help='Parameter config JSON')
    p.add_argument('-b', '--backend', choices=['scipy', 'lmfit'], default='scipy', help='Fitting backend')
    p.add_argument('-m', '--integrator', choices=['grid', 'quad'], default='grid', help='Integrator method')
    p.add_argument('-a', '--accelerator', choices=['auto', 'c', 'numba', 'numpy'], default='auto', help='Acceleration mode for grid integrator')
    p.add_argument('-g', '--grid-size', type=int, default=400, help='Grid size for grid integrator')
    p.add_argument('-o', '--output', default='fit_result.html', help='Output HTML plot file')
    # If no argv and no CLI args provided, pop up GUI file dialogs
    if argv is None and len(sys.argv) == 1:
        gui_res = gui_collect_paths()
        if gui_res is None:
            return
        data_path, config_path, output_path = gui_res
        argv = [data_path, '-c', config_path, '-o', output_path]

    args = p.parse_args(argv)

    x, y = io_module.load_data(args.data)
    param_config = io_module.load_param_config(args.config)

    integrator_opts = {
        'integrator': args.integrator,
        'grid_size': args.grid_size,
        'accelerator': args.accelerator,
    }

    # Prefer a small GUI progress indicator (useful for the EXE built without a console).
    use_tk = False
    try:
        import tkinter as tk
        from tkinter import ttk, messagebox
        use_tk = True
    except Exception:
        use_tk = False

    if use_tk:
        # Thread-safe queue and cancel event
        q = queue.Queue()
        cancel_evt = threading.Event()
        result_holder = {}
        exception_holder = {}

        def progress_callback(call_n, pvec=None):
            # Called from the worker thread; send updates to the GUI poller
            try:
                q.put_nowait(call_n)
            except Exception:
                pass
            return not cancel_evt.is_set()

        def worker():
            try:
                if args.backend == 'scipy':
                    res = fitting.fit_with_scipy(x, y, param_config, integrator_opts=integrator_opts, curvefit_opts=None, progress_callback=progress_callback)
                else:
                    res = fitting.fit_with_lmfit(x, y, param_config, integrator_opts=integrator_opts, minimizer_opts=None, progress_callback=progress_callback)
                result_holder['res'] = res
            except Exception as e:
                exception_holder['e'] = e

        th = threading.Thread(target=worker, daemon=True)
        th.start()

        # Build a small window with an indeterminate progress bar and Cancel button
        root = tk.Tk()
        root.title('Peakfit — Running')
        try:
            root.geometry('420x120')
        except Exception:
            pass
        label_var = tk.StringVar(value='Starting fit...')
        lbl = tk.Label(root, textvariable=label_var)
        lbl.pack(padx=12, pady=(12, 6))
        try:
            pbar = ttk.Progressbar(root, mode='indeterminate')
            pbar.pack(fill='x', padx=12, pady=6)
            pbar.start(80)
        except Exception:
            pbar = None

        def on_cancel():
            cancel_evt.set()
            label_var.set('Cancelling...')

        btn = tk.Button(root, text='Cancel', command=on_cancel)
        btn.pack(pady=(4, 12))

        def poll():
            try:
                while True:
                    val = q.get_nowait()
                    label_var.set(f'Fitting... eval {val}')
            except queue.Empty:
                pass
            if not th.is_alive():
                # Worker finished; close GUI
                try:
                    if pbar is not None:
                        pbar.stop()
                except Exception:
                    pass
                # signal mainloop to exit but keep root available for subsequent dialogs
                try:
                    root.quit()
                except Exception:
                    try:
                        root.destroy()
                    except Exception:
                        pass
                return
            root.after(120, poll)

        root.after(120, poll)
        root.mainloop()

        # After GUI closes, check result or exception
        if 'e' in exception_holder:
            e = exception_holder['e']
            if isinstance(e, RuntimeError) and 'cancel' in str(e).lower():
                print('Fit cancelled by user.')
                return
            else:
                # Re-raise unknown exceptions to surface errors
                raise e
        else:
            res = result_holder['res']
            # support both old (8-tuple) and new (9-tuple) return signatures
            if isinstance(res, (list, tuple)) and len(res) == 9:
                fitted_params, errs, popt, pcov, y_model, r2, converged, conv_message, nfev = res
            else:
                fitted_params, errs, popt, pcov, y_model, r2, converged, conv_message = res

    else:
        # No GUI available; run blocking fit
        if args.backend == 'scipy':
            res = fitting.fit_with_scipy(x, y, param_config, integrator_opts=integrator_opts)
        else:
            try:
                res = fitting.fit_with_lmfit(x, y, param_config, integrator_opts=integrator_opts)
            except Exception as e:
                print("lmfit fitting failed or lmfit not installed:", e)
                sys.exit(2)
        # unpack new or old signature
        if isinstance(res, (list, tuple)) and len(res) == 9:
            fitted_params, errs, popt, pcov, y_model, r2, converged, conv_message, nfev = res
        else:
            fitted_params, errs, popt, pcov, y_model, r2, converged, conv_message = res

    # Print fitted parameters with uncertainties
    print("Fitted parameters:")
    for k in fitted_params:
        v = fitted_params.get(k)
        se = errs.get(k) if errs is not None else None
        if se is None:
            print(f"  {k}: {v}")
        else:
            print(f"  {k}: {v} ± {se}")

    # Print R^2
    try:
        print(f"R^2: {r2:.6g}")
    except Exception:
        pass

    # Print convergence status
    try:
        if converged:
            print("Fit converged:", conv_message)
        else:
            print("Fit failed or did not converge:", conv_message)
    except NameError:
        # Older code paths may not set these variables
        pass

    # Save fitted-function data to a TXT file next to the input data file.
    try:
        residuals = (y - y_model)
        base, ext = os.path.splitext(args.data)
        default_out = base + '_fitted_func' + ext
        out_path = default_out

        # If tkinter is available and we previously created a progress root, create a
        # temporary visible Tk root for the Save As dialog so it shows in the taskbar.
        if use_tk and 'root' in locals():
            try:
                from tkinter import filedialog, Tk
                # Destroy the previous progress root (if any) to avoid a hidden parent
                try:
                    root.destroy()
                except Exception:
                    pass

                # Create a temporary visible root so the dialog gets a taskbar entry
                try:
                    dialog_root = Tk()
                    try:
                        dialog_root.lift()
                        dialog_root.attributes('-topmost', True)
                        dialog_root.update()
                    except Exception:
                        pass
                except Exception:
                    dialog_root = None

                if dialog_root is not None:
                    save_name = filedialog.asksaveasfilename(
                        parent=dialog_root,
                        title='Save fitted function data as',
                        defaultextension=ext or '.txt',
                        initialfile=os.path.basename(default_out),
                        initialdir=os.path.dirname(default_out) or os.getcwd(),
                        filetypes=[('Text files', '*.txt;*.dat;*.csv'), ('All files', '*.*')]
                    )
                    try:
                        dialog_root.destroy()
                    except Exception:
                        pass
                else:
                    # Fallback: no dialog_root, try a parentless dialog
                    save_name = filedialog.asksaveasfilename(
                        title='Save fitted function data as',
                        defaultextension=ext or '.txt',
                        initialfile=os.path.basename(default_out),
                        initialdir=os.path.dirname(default_out) or os.getcwd(),
                        filetypes=[('Text files', '*.txt;*.dat;*.csv'), ('All files', '*.*')]
                    )

                if save_name:
                    out_path = save_name
                else:
                    out_path = default_out
            except Exception:
                out_path = default_out
        else:
            # tkinter not available or not using GUI; fall back to default path
            out_path = default_out

        with open(out_path, 'w', encoding='utf-8') as fh:
            fh.write('# Peakfit fitted function data\n')
            fh.write(f'# Source: {args.data}\n')
            fh.write('# Parameters:\n')
            for k, v in (fitted_params or {}).items():
                fh.write(f'# {k}={v}\n')
            fh.write('#\n')
            fh.write('# iw\ty_model\ty\tresidual\n')
            for xi, ym, yi, r in zip(x, y_model, y, residuals):
                fh.write(f'{xi}\t{ym}\t{yi}\t{r}\n')
        print(f'Saved fitted function data to: {out_path}')
    except Exception as e:
        print('Failed to save fitted data file:', e)

    plotting.plot_fit(x, y, y_model, params=fitted_params, param_errs=errs, r2=r2,
                      output_html=args.output, show=True,
                      converged=(converged if 'converged' in locals() else None),
                      convergence_message=(conv_message if 'conv_message' in locals() else None))


if __name__ == '__main__':
    main()
