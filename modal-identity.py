# get_modal_info.py
import modal
import os
import json

app = modal.App("get-workspace-info")

@app.function(image=modal.Image.debian_slim().pip_install("pyjwt"))
def get_workspace_info():
    import jwt
    token = os.environ.get("MODAL_IDENTITY_TOKEN")
    if not token:
        return {"error": "No MODAL_IDENTITY_TOKEN found"}
    
    claims = jwt.decode(token, options={"verify_signature": False})
    print(json.dumps(claims, indent=2))
    return claims

@app.local_entrypoint()
def main():
    get_workspace_info.remote()