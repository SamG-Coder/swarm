"""Replay a command trace in headless Chrome (WebGPU) and save the last frame.

    python3 tools/shoot.py web/trace.bin out.png [timeout_seconds]

Serves ./web on a local port, opens index.html?trace=..., waits for the replay
to finish, and fails on any shader, validation or console error.
"""
import asyncio, base64, functools, http.server, os, sys, threading
from playwright.async_api import async_playwright

root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "web")
trace, out = sys.argv[1], sys.argv[2]
timeout = float(sys.argv[3]) if len(sys.argv) > 3 else 600

handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=root)
handler.log_message = lambda *a: None
server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
threading.Thread(target=server.serve_forever, daemon=True).start()
url = f"http://127.0.0.1:{server.server_address[1]}/index.html?trace={os.path.basename(trace)}"

async def main():
    errors = []
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            executable_path=os.environ.get("CHROME", "/usr/bin/google-chrome"), headless=True,
            args=["--no-sandbox", "--enable-unsafe-webgpu", "--enable-features=Vulkan",
                  "--use-angle=swiftshader", "--use-webgpu-adapter=swiftshader", "--ignore-gpu-blocklist", "--disable-gpu-watchdog"])
        page = await browser.new_page(viewport={"width": 1280, "height": 720})
        page.on("console", lambda m: errors.append(m.text) if m.type in ("error", "warning") else None)
        page.on("pageerror", lambda e: errors.append(str(e)))
        await page.goto(url)
        try:
            await page.wait_for_function("window.__done || window.__error", timeout=timeout * 1000)
        except Exception as e:
            errors.append(f"timeout: {e}")
        print("frames replayed:", await page.evaluate("window.__frame"))
        png = await page.evaluate("window.__png")
        err = await page.evaluate("window.__error")
        if err: errors.append(err)
        hud = await page.evaluate("document.getElementById('hud').textContent")
        if png:
            open(out, "wb").write(base64.b64decode(png.split(",")[1]))
        await browser.close()
    print("hud:", hud)
    for e in errors: print("ERROR:", e)
    sys.exit(1 if errors or not png else 0)

asyncio.run(main())
