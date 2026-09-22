# FanficCode
Run pip3 install requests beautifulsoup4 feedparser and python3 -m pip install numpy pandas before using any versions of the scraper

For V4/ao3statswithplaywright.py (the browser-driven version), also run:

    pip3 install playwright
    python3 -m playwright install chromium

It drives a real Chromium window instead of sending bare HTTP requests, and keeps its browser profile in V4/ao3_browser_profile so clearance cookies carry over between runs. Set AO3_HEADLESS=1 to run without a visible window. Filters, cache, CSV output, and distributions are identical to ao3_research_sampler_v4.py.
CROSSOVER FILTER DOES NOT WORK! Correctly lists all fandoms but does the true/false does not work