"""
Boston Housing Price Predictor - Startup Script
Usage:
    python app.py
       OR
    python run.py
"""

from app import app, is_port_in_use

if __name__ == "__main__":
    target_port = 5050
    if not is_port_in_use(5000):
        target_port = 5000

    print("=" * 60)
    print("  BOSTON HOUSING PRICE PREDICTOR")
    print(f"  Open in your browser: http://127.0.0.1:{target_port}")
    print("=" * 60)
    app.run(host="127.0.0.1", port=target_port, debug=False)
