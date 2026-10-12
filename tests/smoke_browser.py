"""Teste de fumaça no navegador (opcional): python tests/smoke_browser.py <site_dir> <node_modules_dir>
Intercepta CDNs (leaflet/three) com cópias locais e tiles com PNG vazio, para rodar offline."""
import sys, os, threading, http.server, functools
from playwright.sync_api import sync_playwright

site, nm = sys.argv[1], sys.argv[2]
h = functools.partial(http.server.SimpleHTTPRequestHandler, directory=site)
srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), h); threading.Thread(target=srv.serve_forever, daemon=True).start()
PNG = bytes.fromhex("89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4890000000d49444154789c6360f8ffff3f0005fe02fea7d6d2c80000000049454e44ae426082")
erros = []

def rota(route):
    u = route.request.url
    if "leaflet.css" in u: route.fulfill(path=f"{nm}/leaflet/dist/leaflet.css", content_type="text/css")
    elif "leaflet.js" in u: route.fulfill(path=f"{nm}/leaflet/dist/leaflet.js", content_type="application/javascript")
    elif "three.min.js" in u: route.fulfill(path=f"{nm}/three/build/three.min.js", content_type="application/javascript")
    elif u.startswith("http://127.0.0.1"): route.continue_()
    else: route.fulfill(body=PNG, content_type="image/png")

with sync_playwright() as p:
    b = p.chromium.launch(executable_path="/opt/pw-browsers/chromium", args=["--use-gl=swiftshader", "--enable-unsafe-swiftshader", "--no-sandbox"])
    pg = b.new_page(viewport={"width": 1200, "height": 800})
    pg.on("pageerror", lambda e: erros.append(str(e))); pg.on("console", lambda m: erros.append(m.text) if m.type == "error" else None)
    pg.route("**/*", rota)
    pg.goto(f"http://127.0.0.1:{srv.server_port}/index.html"); pg.wait_for_timeout(1500)
    print("camadas no painel:", pg.locator("input[data-id]").count())
    pg.screenshot(path=os.path.join(site, "..", "smoke_mapa.png"))
    pg.click("#mapa", position={"x": 500, "y": 400}); pg.wait_for_timeout(300)
    print("popup:", pg.locator(".leaflet-popup-content").first.inner_text()[:120].replace("\n", " | "))
    pg.click("#bTodos"); print("marcadas após 'desmarcar todos':", pg.locator("input[data-id]:checked").count())
    pg.click("#bMed"); pg.click("#mapa", position={"x": 300, "y": 300}); pg.click("#mapa", position={"x": 600, "y": 300})
    print("régua:", pg.locator("#med").inner_text())
    pg.click("#bMed")
    if pg.locator("#b3d").count():
        pg.click("#b3d"); pg.wait_for_timeout(800); pg.screenshot(path=os.path.join(site, "..", "smoke_3d.png"))
        print("3D canvas:", pg.locator("#v3d canvas").count())
    b.close()
print("ERROS JS:", [e for e in erros if "net::" not in e and "Failed to load resource" not in e] or "nenhum")
