import argparse
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
import os
import shutil
import uvicorn
import tempfile

from sdk import process_files, test, infer

app = FastAPI()

@app.post("/load-model/")
async def load_model(file: UploadFile):
    # Carrega o modelo RKNN usando um arquivo temporário
    with tempfile.NamedTemporaryFile(delete=True) as temp:
        temp.write(await file.read())
        temp.flush()
        
        ret = test(temp)
        
        if ret != 0:
            return {"error": "Falha ao carregar o modelo."}

@app.post("/infer/")
async def infer_image(file: UploadFile = File(...)):

    contents = await file.read()
    result = infer(contents)

    return result


@app.post("/upload/")
async def upload_files(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    # Salva os arquivos em disco
    file1_path = os.path.join(UPLOAD_DIR, file1.filename)
    file2_path = os.path.join(UPLOAD_DIR, file2.filename)

    with open(file1_path, "wb") as f1:
        shutil.copyfileobj(file1.file, f1)

    with open(file2_path, "wb") as f2:
        shutil.copyfileobj(file2.file, f2)

    # Executa a função de processamento
    result_file_path = os.path.join(UPLOAD_DIR, "model.rknn")
    print(file1_path, file2_path, result_file_path)
    process_files(file1_path, file2_path, result_file_path)

    # Retorna o arquivo de resultado
    return FileResponse(result_file_path, media_type="text/plain", filename="model.rknn")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run FastAPI server with custom settings.")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the FastAPI server on")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the FastAPI server on")
    parser.add_argument("--upload_dir", type=str, default="/src/upload", help="Directory to save uploaded files")

    args = parser.parse_args()

    # Define o diretório de upload com base nos argumentos
    global UPLOAD_DIR
    UPLOAD_DIR = args.upload_dir
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    # Executa o servidor com os parâmetros fornecidos
    uvicorn.run(app, host=args.host, port=args.port)