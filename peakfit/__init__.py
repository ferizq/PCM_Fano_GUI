"""Simple peak-fitting package exposing the model."""
"""Simple peak-fitting package exposing the model.

This module also performs a conservative numeric-locale initialization
so numeric parsing/formatting is consistent across user systems.
"""
try:
	import locale
	try:
		# Force C numeric locale so decimal separators and other numeric
		# behaviours are stable across user machines/locales.
		locale.setlocale(locale.LC_NUMERIC, 'C')
	except Exception:
		# Best-effort only: don't fail import if setting locale is not allowed
		pass
except Exception:
	# ignore locale setup failures
	pass

from .model import model, kernel_integrand

__all__ = ["model", "kernel_integrand"]
