"""
Simple HTTP server for container health checks.
Run this alongside the main application in Docker to ensure container health.
"""
import http.server
import socketserver
import argparse
from threading import Thread

class HealthCheckHandler(http.server.SimpleHTTPRequestHandler):
    """Handler for health check requests"""
    
    def do_GET(self):
        """Handle GET requests"""
        if self.path == '/health':
            # Return 200 OK for health checks
            self.send_response(200)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write(b'OK')
            return
        else:
            # Return 404 for any other path
            self.send_response(404)
            self.end_headers()
    
    def log_message(self, format, *args):
        """Suppress logging to keep the console clean"""
        return

def start_health_check_server(port=8000):
    """Start a health check server on the specified port"""
    with socketserver.TCPServer(("0.0.0.0", port), HealthCheckHandler) as httpd:
        print(f"Health check server started at port {port}")
        httpd.serve_forever()

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Health check server for Docker container")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    args = parser.parse_args()
    
    try:
        start_health_check_server(args.port)
    except KeyboardInterrupt:
        print("Health check server stopped.")
    except OSError as e:
        if e.errno == 10048:  # Port already in use on Windows
            print(f"Error: Port {args.port} is already in use.")
        else:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
