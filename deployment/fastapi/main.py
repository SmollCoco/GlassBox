from fastapi import FastAPI, File, UploadFile, Form
from GlassBox.ml import autofit

app = FastAPI()

@app.post("/autofit")
async def execute_autofit(target_col: str = Form(...), file: UploadFile = File(...)):
    content = await file.read()
    csv_string = content.decode('utf-8')
    result = autofit(csv_string, target_col)
    return result
