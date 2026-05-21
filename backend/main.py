import joblib, json
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path

BASE = Path(__file__).parent
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["GET"], allow_headers=["*"])


scaler = joblib.load(BASE / 'models/scaler.pkl')
kmeans = joblib.load(BASE / 'models/kmeans.pkl')
xgb_model = joblib.load(BASE / 'models/xgboost.pkl')

with open(BASE / 'models/config.json') as f:
    cfg = json.load(f)
    
pesos = np.array(cfg['pesos'])
f_clus = cfg['features_cluster']
f_xgb = cfg['features_xgb']

def calcular_risco(linha: pd.DataFrame) -> dict:
    X_scaled = scaler.transform(linha[f_clus])

    distancias = kmeans.transform(X_scaled)
    dist_inv = 1.0 / (distancias + 1e-8)
    probs = dist_inv / dist_inv.sum(axis=1, keepdims=True)

    indice_atual = float((probs * pesos).sum(axis=1)[0] * 100)

    cluster = int(kmeans.predict(X_scaled)[0])
    mapa_status = {
        0: 'Clima de Atenção',
        1: 'Condição Favorável',
        2: 'Risco de Seca',
        3: 'Risco de Enchente',
    }

    indice_72h = float(
        np.clip(xgb_model.predict(linha[f_xgb])[0], 0, 100)
    )

    return {
        'indice_atual': round(indice_atual, 1),
        'indice_72h': round(indice_72h, 1),
        'tendencia': round(indice_72h - indice_atual, 1),
        'status': mapa_status[cluster],
        'cluster': cluster,
    }

@app.on_event("startup")
def carregar():
    global df, coords
    df = pd.read_parquet(BASE.parent / 'data/clima_preprocessado.parquet')
    coords = pd.read_csv(BASE.parent / 'data/estacoes_mapeadas.csv')
    coords = coords.rename(columns={'Código': 'ESTACAO', 'Nome': 'NOME'})

@app.get("/api/risco")
def risco(estacao: str = "A401", data: str = "2021-01-15"):
    snapshot = (df[(df['ESTACAO'] == estacao) & 
                   (df.index.date == pd.Timestamp(data).date())]
                .iloc[[-1]])
    
    if snapshot.empty:
        return {"erro": f"Sem dados para {estacao} em {data}"}
    
    risco = calcular_risco(snapshot)
    info = coords[coords.ESTACAO == estacao].iloc[0]
    nome = coords[coords.ESTACAO == estacao].iloc[0]['NOME']
    
    return {
        'estacao': estacao,
        "nome": nome,
        'data': data,
        **risco
    }
    
# if __name__ == "__main__":
#     import json
#
#     print("Carregando dados para simulação...")
#     
#     global df, coords
#     df = pd.read_parquet(BASE.parent / 'data/clima_preprocessado.parquet')
#     coords = pd.read_csv(BASE.parent / 'data/estacoes_mapeadas.csv')
#     coords = coords.rename(columns={'Código': 'ESTACAO', 'Nome': 'NOME'})
#
#     estacao_teste = "A402"
#     data_teste = "2021-01-15"
#
#     print(f"\nSimulando predição para a estação {estacao_teste} em {data_teste}:\n")
#
#     try:
#         resultado = risco(estacao=estacao_teste, data=data_teste)
#         print(json.dumps(resultado, indent=4, ensure_ascii=False))
#         
#     except Exception as e:
#         print(f"Ocorreu um erro na simulação: {e}")