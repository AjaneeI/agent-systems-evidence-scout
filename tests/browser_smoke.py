from pathlib import Path
from playwright.sync_api import sync_playwright

BASE_URL = "http://127.0.0.1:7860"
OUT = Path("/tmp/evidence-scout-ui")
OUT.mkdir(parents=True, exist_ok=True)

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1440, "height": 900})
    page.goto(BASE_URL, wait_until="networkidle")
    assert page.title() == "Agent Systems Evidence Scout"
    assert page.get_by_text("Evidence Scout", exact=False).first.is_visible()
    assert page.get_by_text("What would you like to research?", exact=True).is_visible()
    assert page.get_by_role("button", name="Run evidence search").is_visible()
    assert page.locator("#contract-title").is_visible()
    assert page.get_by_text("Ready to research", exact=True).first.is_visible()
    page.screenshot(path=str(OUT / "desktop-empty.png"), full_page=False)

    example = page.get_by_role("button", name="Reliability or failure modes in LLM agents")
    example.click()
    textbox = page.locator("textarea").first
    assert "reliability or failure modes" in textbox.input_value().lower()
    assert page.evaluate("document.documentElement.scrollWidth <= document.documentElement.clientWidth")

    mobile = browser.new_page(viewport={"width": 390, "height": 844})
    mobile.goto(BASE_URL, wait_until="networkidle")
    assert mobile.get_by_role("button", name="Run evidence search").is_visible()
    assert mobile.locator("textarea").first.is_visible()
    assert mobile.evaluate("document.documentElement.scrollWidth <= document.documentElement.clientWidth")
    mobile.screenshot(path=str(OUT / "mobile-empty.png"), full_page=False)
    browser.close()
