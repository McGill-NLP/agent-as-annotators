"""
Monkey-patch browsergym's BrowserEnv to use anti-detection measures.

Changes:
1. Uses Chrome's new headless mode (--headless=new) instead of Playwright's old
   headless mode. The new mode shares the same rendering pipeline as headed Chrome,
   making it much harder for sites to detect automation.
2. Adds --disable-blink-features=AutomationControlled to hide navigator.webdriver.
3. Sets a real Chrome user agent instead of the default HeadlessChrome one.
4. Applies playwright-stealth patches (navigator.webdriver, plugins, etc.) to every page.

Usage: import this module before any browsergym environment is created.
    import llm_annotators.browser_stealth  # noqa: F401
"""

import logging

from playwright_stealth import Stealth

logger = logging.getLogger(__name__)

_stealth = Stealth(navigator_platform_override="Linux x86_64")

_STEALTH_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)

# Flag to distinguish stealth headless=False from user-intended headed mode
_stealth_headless_active = False

_patched = False


def _patch_browsergym():
    global _stealth_headless_active, _patched
    if _patched:
        return
    _patched = True

    import browsergym.core.env as env_module
    import playwright.sync_api

    _original_reset = env_module.BrowserEnv.reset

    def _stealth_reset(self, seed=None, *args, **kwargs):
        global _stealth_headless_active

        # Inject user_agent into pw_context_kwargs if not already set
        if "user_agent" not in self.pw_context_kwargs:
            self.pw_context_kwargs["user_agent"] = _STEALTH_USER_AGENT

        # If headless, switch to new headless mode via flag
        orig_headless = self.headless
        if self.headless:
            self.headless = False  # tell Playwright not to use old headless
            _stealth_headless_active = True
        try:
            result = _original_reset(self, seed=seed, *args, **kwargs)
        finally:
            self.headless = orig_headless
            _stealth_headless_active = False

        return result

    env_module.BrowserEnv.reset = _stealth_reset

    # Patch browser launch to inject stealth args
    _original_launch = playwright.sync_api.BrowserType.launch

    def _stealth_launch(self, **kwargs):
        args = list(kwargs.get("args", []))

        # Always add anti-automation flag
        if "--disable-blink-features=AutomationControlled" not in args:
            args.append("--disable-blink-features=AutomationControlled")

        # Use new headless mode only when stealth redirect is active
        if _stealth_headless_active and "--headless=new" not in args:
            args.append("--headless=new")

        kwargs["args"] = args
        return _original_launch(self, **kwargs)

    playwright.sync_api.BrowserType.launch = _stealth_launch

    # Patch new_page to apply stealth scripts
    _original_new_page = playwright.sync_api.BrowserContext.new_page

    def _stealth_new_page(self, **kwargs):
        page = _original_new_page(self, **kwargs)
        try:
            _stealth.apply_stealth_sync(page)
        except Exception:
            pass  # don't break if stealth fails
        return page

    playwright.sync_api.BrowserContext.new_page = _stealth_new_page

    logger.info("Browser stealth patches applied")


def _patch_run_exp():
    """Patch agentlab's run_exp so stealth is applied inside Ray workers."""
    import agentlab.experiments.exp_utils as exp_utils

    _original_run_exp = exp_utils.run_exp

    def _stealth_run_exp(exp_arg, *dependencies, avg_step_timeout=60):
        _patch_browsergym()  # apply patches in this worker process
        return _original_run_exp(exp_arg, *dependencies, avg_step_timeout=avg_step_timeout)

    exp_utils.run_exp = _stealth_run_exp

    # Also re-wrap with ray.remote since graph_execution_ray does this at import time
    import agentlab.experiments.graph_execution_ray as graph_ray
    import ray

    graph_ray.run_exp = ray.remote(_stealth_run_exp)

    logger.info("run_exp patched for Ray worker stealth")


# Apply patches in main process
_patch_browsergym()
_patch_run_exp()
