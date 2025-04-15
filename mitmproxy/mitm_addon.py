from mitmproxy import http, ctx
import json
import requests

FASTAPI_URL = "http://guardian-api:8000/label"
REQUEST_TIMEOUT_SEC = 5

def request(flow: http.HTTPFlow):
    body = flow.request.get_text()
    parsed = json.loads(body)
    for msg in parsed.get("messages", []):
        content = msg.get("content")
        if content:
            payload = {
                "text": content,
                "threshold": 0.6
            }
            try:
                res = requests.post(FASTAPI_URL, json=payload, timeout=REQUEST_TIMEOUT_SEC)
                if res.status_code == 200:
                    label = res.json().get("label")
                    if label is not None:
                        block_msg = {
                            "message": f" Sorry, your prompt cannot be answered as it contains: {label}"
                        }
                        flow.response = http.Response.make(
                            200,
                            json.dumps(block_msg),
                            {"Content-Type": "application/json"}
                        )
                        return 
                else:
                    ctx.log.error(f"FastAPI error: {res.status_code}")
            except Exception as e:
                ctx.log.error(f"Request failed, status code:{res.status_code}")
    
    

