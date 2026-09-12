#!/usr/bin/env python3
"""Simple HTTP server for Appian demo UI with ClickHouse proxy"""
import http.server
import socketserver
import json
import urllib.request
import urllib.error

PORT = 8090

class CORSRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()
    
    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()
    
    def do_POST(self):
        if self.path == '/query':
            # Proxy ClickHouse queries
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            try:
                data = json.loads(post_data)
                query = data.get('query', '')
                
                # Forward to ClickHouse
                req = urllib.request.Request('http://localhost:8123/', 
                                            data=query.encode(), 
                                            method='POST')
                req.add_header('Content-Type', 'text/plain')
                
                with urllib.request.urlopen(req) as response:
                    clickhouse_response = response.read()
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    self.wfile.write(clickhouse_response)
            except urllib.error.URLError as e:
                self.send_response(503)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                error_msg = json.dumps({"error": f"ClickHouse connection failed: {str(e)}"})
                self.wfile.write(error_msg.encode())
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                error_msg = json.dumps({"error": str(e)})
                self.wfile.write(error_msg.encode())
        else:
            super().do_POST()

if __name__ == '__main__':
    with socketserver.TCPServer(("", PORT), CORSRequestHandler) as httpd:
        print(f"🚀 Appian Demo UI running at http://localhost:{PORT}")
        print(f"📊 Open the 'Database Queries' tab to run ClickHouse queries")
        print(f"\n✨ Features:")
        print(f"   - Live Dashboard with real-time monitoring")
        print(f"   - Manual Predictions (Duration & SLA Risk)")
        print(f"   - What-If Simulation Sandbox")
        print(f"   - Interactive ClickHouse Query Executor")
        print(f"\n🔄 Press Ctrl+C to stop\n")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n\n👋 Server stopped")

