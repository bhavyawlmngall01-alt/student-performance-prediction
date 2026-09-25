import sys
import os
import json

# Allow the API file to import model_service.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_service import predict_student


def handler(request):
    """
    Vercel Python API endpoint.

    Expected JSON:
    {
        "attendance": 85,
        "study_hours": 6.5,
        "previous_marks": 75,
        "assignments": 18,
        "sleep_hours": 7.5,
        "internet_access": "Yes",
        "extra_classes": "No",
        "model_type": "random_forest"
    }
    """

    try:
        # Handle OPTIONS request for browser CORS
        if request.method == "OPTIONS":
            return {
                "statusCode": 200,
                "headers": {
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Methods": "POST, OPTIONS",
                    "Access-Control-Allow-Headers": "Content-Type",
                },
                "body": ""
            }

        if request.method != "POST":
            return {
                "statusCode": 405,
                "headers": {
                    "Content-Type": "application/json"
                },
                "body": json.dumps({
                    "error": "Only POST requests are allowed."
                })
            }

        data = request.body

        if isinstance(data, bytes):
            data = data.decode("utf-8")

        if isinstance(data, str):
            data = json.loads(data)

        result = predict_student(
            attendance=float(data["attendance"]),
            study_hours=float(data["study_hours"]),
            previous_marks=float(data["previous_marks"]),
            assignments=float(data["assignments"]),
            sleep_hours=float(data["sleep_hours"]),
            internet_access=data.get("internet_access", "Yes"),
            extra_classes=data.get("extra_classes", "No"),
            model_type=data.get("model_type", "random_forest")
        )

        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*"
            },
            "body": json.dumps(result)
        }

    except Exception as e:
        return {
            "statusCode": 500,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*"
            },
            "body": json.dumps({
                "error": str(e)
            })
        }