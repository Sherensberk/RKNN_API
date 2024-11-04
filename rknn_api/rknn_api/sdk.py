
from rknn.api import RKNN
import numpy as np
import cv2

rknn = RKNN(verbose=True)

def convert_model(model_path, weight_path):
    rknn.config(mean_values=[0, 0, 0], std_values=[255, 255, 255], target_platform='rk3566')
    rknn.load_darknet(model=model, weight=weight)
    ret = rknn.build(do_quantization=True, dataset=DATASET)
    rknn.export_rknn(RKNN_MODEL_PATH)

def process_files(model_path: str, weight_path: str, rknn_model_path: str):
    # Exemplo de função que manipula os arquivos e salva o resultado em um terceiro arquivo
    rknn.config(mean_values=[0, 0, 0], std_values=[255, 255, 255], target_platform='rk3566')
    rknn.load_darknet(model=model_path, weight=weight_path)
    ret = rknn.build(do_quantization=False)
    rknn.export_rknn(rknn_model_path)

def test(temp):
    return rknn.load_rknn(temp.name)  # Carrega o modelo a partir do caminho temporário


def infer(image):
    rknn.init_runtime()

    # Carregar a imagem recebida
    nparr = np.frombuffer(image, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Pré-processamento da imagem (ajustar conforme necessário para o seu modelo)
    input_data = cv2.resize(image, (224, 224))  # Exemplo de redimensionamento
    input_data = input_data / 255.0  # Normalização, ajuste conforme seu modelo
    input_data = np.expand_dims(input_data, axis=0).astype(np.float32)

    # Fazer a inferência
    outputs = rknn.inference(inputs=[input_data])

    rknn.release()
    # Processar e retornar resultados
    # Aqui, você precisa ajustar conforme a saída do seu modelo
    return {"predictions": outputs}  # Formate a saída conforme necessário