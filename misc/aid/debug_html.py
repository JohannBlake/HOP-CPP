from playwright.sync_api import sync_playwright
import pathlib
import http.server
import socketserver
import threading
import time
import os

# Pfad zur HTML-Datei - use current working directory
html_dir = os.getcwd()
html_filename = "vis_3d_index.html"

# Lokalen Server starten
PORT = 8000
# html_dir is already the current working directory, so no need to change

class Handler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        # Unterdrücke Server-Logs
        pass

httpd = socketserver.TCPServer(("", PORT), Handler)
server_thread = threading.Thread(target=httpd.serve_forever)
server_thread.daemon = True
server_thread.start()

print(f"Server gestartet auf http://localhost:{PORT}")
time.sleep(1)  # Kurz warten bis Server bereit ist

# URL für lokalen Server
html_url = f"http://localhost:{PORT}/{html_filename}"

try:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # Erweiterte Fehler-Handler
        def handle_console(msg):
            timestamp = time.strftime("%H:%M:%S")
            msg_type = msg.type
            msg_text = msg.text
            
            print(f"[{timestamp}] Console {msg_type.upper()}: {msg_text}")
            
            # Zusätzliche Details für Errors
            if msg_type == "error":
                try:
                    location = msg.location
                    if location:
                        print(f"    Location: {location.get('url', 'unknown')}:{location.get('lineNumber', '?')}:{location.get('columnNumber', '?')}")
                except:
                    pass

        def handle_page_error(exception):
            timestamp = time.strftime("%H:%M:%S")
            print(f"[{timestamp}] Page Exception: {exception}")

        def handle_request_failed(request):
            timestamp = time.strftime("%H:%M:%S")
            print(f"[{timestamp}] Request Failed: {request.url}")
            print(f"    Method: {request.method}")
            if request.failure:
                print(f"    Failure: {request.failure}")

        def handle_response(response):
            timestamp = time.strftime("%H:%M:%S")
            if response.status >= 400:
                print(f"[{timestamp}] HTTP {response.status}: {response.url}")
            elif response.status >= 300:
                print(f"[{timestamp}] HTTP {response.status}: {response.url}")

        # Event-Handler registrieren
        page.on("console", handle_console)
        page.on("pageerror", handle_page_error)
        page.on("requestfailed", handle_request_failed)
        page.on("response", handle_response)

        print(f"Opening: {html_url}")
        page.goto(html_url)

        # warten, um JS auszuführen - stoppt nach 5 Sekunden
        page.wait_for_timeout(5000)
        print("Timeout reached - closing browser")

        browser.close()

finally:
    # Server beenden
    print("Shutting down server...")
    httpd.shutdown()
    httpd.server_close()
    print("Server stopped")
