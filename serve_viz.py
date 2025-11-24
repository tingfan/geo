import http.server
import socketserver
import webbrowser
import os
import sys

PORT = 8000

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=os.getcwd(), **kwargs)

def main():
    # Allow reusing the address to avoid "Address already in use" errors on quick restarts
    socketserver.TCPServer.allow_reuse_address = True
    
    try:
        with socketserver.TCPServer(("", PORT), Handler) as httpd:
            print(f"Serving at http://localhost:{PORT}")
            print("Press Ctrl+C to stop.")
            
            # Open the visualization in the default browser
            url = f"http://localhost:{PORT}/terrain_viz.html"
            print(f"Opening {url}...")
            webbrowser.open(url)
            
            httpd.serve_forever()
    except OSError as e:
        if e.errno == 48: # Address already in use
            print(f"Port {PORT} is already in use. Please try again later or kill the process using it.")
        else:
            print(f"Error starting server: {e}")

if __name__ == "__main__":
    main()
