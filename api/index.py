import sys
import os
import json
from http.server import BaseHTTPRequestHandler

# Allow the API file to import model_service.py
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from model_service import predict_student


class handler(BaseHTTPRequestHandler):

    def _send_json(self, status_code, data):
        response = json.dumps(data).encode("utf-8")

        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header(
            "Access-Control-Allow-Methods",
            "POST, OPTIONS"
        )
        self.send_header(
            "Access-Control-Allow-Headers",
            "Content-Type"
        )
        self.end_headers()

        self.wfile.write(response)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header(
            "Access-Control-Allow-Methods",
            "POST, OPTIONS"
        )
        self.send_header(
            "Access-Control-Allow-Headers",
            "Content-Type"
        )
        self.end_headers()

    def do_POST(self):
        try:
            # Read request body
            content_length = int(
                self.headers.get("Content-Length", 0)
            )

            body = self.rfile.read(content_length)

            data = json.loads(body.decode("utf-8"))

            # Run ML prediction
            result = predict_student(
                attendance=float(data["attendance"]),
                study_hours=float(data["study_hours"]),
                previous_marks=float(data["previous_marks"]),
                assignments=float(data["assignments"]),
                sleep_hours=float(data["sleep_hours"]),
                internet_access=data.get(
                    "internet_access",
                    "Yes"
                ),
                extra_classes=data.get(
                    "extra_classes",
                    "No"
                ),
                model_type=data.get(
                    "model_type",
                    "random_forest"
                )
            )

            self._send_json(200, result)

        except Exception as e:
            self._send_json(
                500,
                {
                    "error": str(e)
                }
            )

    def do_GET(self):
        self._send_json(
            200,
            {
                "message": "Student Performance Prediction API is running."
            }
        )