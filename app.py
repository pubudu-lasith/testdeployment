
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse
from starlette.responses import HTMLResponse, RedirectResponse
from uvicorn import run as app_run
import numpy as np
from scipy.signal import detrend, savgol_filter
import pandas as pd
import pickle
from math import ceil
import os
from numpy import array, mean, ones, where, zeros
import numpy as np
from scipy.signal import medfilt



from typing import Optional

APP_HOST = "0.0.0.0"
APP_PORT = 8080
totlen = 1


app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["authentication"])
async def index(request: Request):
    return RedirectResponse(url="/docs")

@app.get("/ping")
async def ping():
    return {"status": "healthy"}



@app.post("/invocations")
async def predictRouteClient(request: Request):
   
    
    try:
        input_data = await request.json()
        content = input_data.get("record")
        content = np.array(content).reshape(1, -1)

        filename = 'model.pkl'
        model_dir = '/opt/ml/model/'
        model_path =  os.path.join(model_dir, filename)

        with open(model_path, 'rb') as model_file:
                loaded_model = pickle.load(model_file)

        prediction=loaded_model.predict(content)

        

        return JSONResponse(content={"status": True, "prediction": prediction.tolist()})
    except Exception as e:
        return JSONResponse(content={"status": False, "error": str(e)}, status_code=500)
    
if __name__ == "__main__":
    app_run(app, host=APP_HOST, port=APP_PORT)
