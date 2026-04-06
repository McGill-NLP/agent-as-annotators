"""
Monkey-patch browsergym's observation extraction to add timeouts.

Problem: `cdp.send("Page.captureScreenshot")` in browsergym/core/observation.py
can hang for ~39 minutes when Chrome's renderer gets stuck on certain pages
(typically checkout/cart pages in VisualWebArena). When Ray force-kills the task
after its timeout, the Playwright event loop on that worker gets corrupted,
causing 30+ subsequent tasks to fail instantly ("no running event loop").

Fix: Replace the CDP-based screenshot with Playwright's page.screenshot() which
has a built-in timeout parameter that works correctly with uvloop. On timeout,
return a blank screenshot so the task can continue without losing the worker.

Note: signal.alarm does NOT work with uvloop (Playwright's event loop). The
SIGALRM fires inside uvloop._ceval_process_signals, and the raised exception
crashes the event loop instead of being caught by the try/except wrapper.

Usage: import this module before any browsergym environment is created.
    import agent_as_annotators.obs_timeout  # noqa: F401
"""

import io
import logging

import numpy as np
import PIL.Image

logger = logging.getLogger(__name__)

# Timeout for screenshot extraction (milliseconds for Playwright API).
SCREENSHOT_TIMEOUT_MS = 30_000

_patched = False


class ScreenshotTimeoutError(Exception):
    pass


def _patch_extract_screenshot():
    global _patched
    if _patched:
        return
    _patched = True

    import browsergym.core.observation as obs_module

    def _timed_extract_screenshot(page):
        """extract_screenshot using Playwright's built-in timeout instead of signal.alarm."""
        try:
            png_bytes = page.screenshot(
                timeout=SCREENSHOT_TIMEOUT_MS,
                type="png",
                full_page=False,
            )
            img = PIL.Image.open(io.BytesIO(png_bytes)).convert("RGB")
            return np.array(img)
        except Exception as e:
            error_name = type(e).__name__
            logger.warning(
                f"extract_screenshot failed ({error_name}: {e}) — "
                f"returning blank screenshot (url={page.url!r})"
            )
            viewport = page.viewport_size
            if viewport:
                w, h = viewport["width"], viewport["height"]
            else:
                w, h = 1280, 720
            return np.zeros((h, w, 3), dtype=np.uint8)

    # Patch at the module level (used by env.py's import)
    obs_module.extract_screenshot = _timed_extract_screenshot

    # Also patch in env.py since it imports the function directly
    import browsergym.core.env as env_module

    env_module.extract_screenshot = _timed_extract_screenshot

    logger.info(
        f"Observation timeout patch applied (screenshot timeout={SCREENSHOT_TIMEOUT_MS}ms, "
        f"using Playwright built-in timeout)"
    )


def _patch_run_exp():
    """Patch agentlab's run_exp so the timeout is applied inside Ray workers."""
    import agentlab.experiments.exp_utils as exp_utils

    _original_run_exp = exp_utils.run_exp

    def _timeout_run_exp(exp_arg, *dependencies, avg_step_timeout=60):
        _patch_extract_screenshot()  # apply patch in this worker process
        return _original_run_exp(exp_arg, *dependencies, avg_step_timeout=avg_step_timeout)

    exp_utils.run_exp = _timeout_run_exp

    # Re-wrap with ray.remote since graph_execution_ray does this at import time
    import agentlab.experiments.graph_execution_ray as graph_ray
    import ray

    graph_ray.run_exp = ray.remote(_timeout_run_exp)

    logger.info("run_exp patched for Ray worker observation timeout")


# Apply patches in main process
_patch_extract_screenshot()
_patch_run_exp()
