import requests
import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt
import plotly.graph_objects as go

# Função para coletar dados da API
def fetch_brent_data(api_key, start_date, end_date):
    url = "https://api.eia.gov/v2/petroleum/pri/spt/data/"
    params = {
        'api_key': api_key,
        'facets[product][]': 'EPCBRENT',
        'data[]': 'value',
        'frequency': 'daily',
        'start': start_date,
        'end': end_date,
        'offset': 0
    }
    all_data = []
    while True:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            if 'response' in data and 'data' in data['response']:
                brent_data = data['response']['data']
                all_data.extend(brent_data)
                if len(brent_data) < 5000:
                    break
                params['offset'] += 5000
            else:
                raise ValueError("Erro: Nenhum dado encontrado na resposta.")
        else:
            raise ConnectionError(f"Erro na requisição: Status Code {response.status_code}")
    df = pd.DataFrame(all_data)
    df = df[['period', 'value']].rename(columns={'period': 'ds', 'value': 'y'})
    df['ds'] = pd.to_datetime(df['ds'])
    df['y'] = pd.to_numeric(df['y'], errors='coerce')
    return df.dropna()

# Função para calcular métricas
def calculate_metrics(y_true, y_pred):
    non_zero_mask = y_true != 0
    y_true = y_true[non_zero_mask]
    y_pred = y_pred[non_zero_mask]
    
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    accuracy = 100 - mape
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "MAPE": mape, "Accuracy %": accuracy}

# Função para validação cruzada
def validate_model(model, horizon="30 days"):
    df_cv = cross_validation(model, initial='730 days', period='180 days', horizon=horizon)
    df_performance = performance_metrics(df_cv)
    return df_performance

# Função para treinar e prever usando Prophet
def train_and_forecast(data, prediction_days):
    train = data.iloc[:-prediction_days]
    test = data.iloc[-prediction_days:]
    model = Prophet()
    model.fit(train)
    future = model.make_future_dataframe(periods=prediction_days)
    forecast = model.predict(future)
    forecast_filtered = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(prediction_days)
    return model, forecast, forecast_filtered, test

# Função para visualização interativa
def interactive_plot(data, forecast, prediction_days):
    # Garantir que somente as colunas necessárias são usadas
    data = data[['ds', 'y']].dropna()

    fig = go.Figure()

    # Adicionar os dados históricos
    fig.add_trace(go.Scatter(
        x=data['ds'], 
        y=data['y'], 
        mode='lines', 
        name='Histórico',
        line=dict(color='blue', dash='solid')  # Linha sólida para histórico
    ))

    # Adicionar previsão
    fig.add_trace(go.Scatter(
        x=forecast['ds'], 
        y=forecast['yhat'], 
        mode='lines', 
        name='Previsão', 
        line=dict(dash='dot', color='orange')  # Linha pontilhada para previsão
    ))

    # Adicionar limites de confiança
    fig.add_trace(go.Scatter(
        x=forecast['ds'], 
        y=forecast['yhat_upper'], 
        mode='lines', 
        name='Limite Superior', 
        line=dict(color='rgba(204, 204, 204, 0.6)'),
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=forecast['ds'], 
        y=forecast['yhat_lower'], 
        mode='lines', 
        name='Limite Inferior', 
        line=dict(color='rgba(204, 204, 204, 0.6)'),
        fill='tonexty',
        fillcolor='rgba(204, 204, 204, 0.3)',
        showlegend=True
    ))

    # Linha de início da previsão
    start_date = data['ds'].iloc[-prediction_days]
    fig.add_vline(x=start_date, line=dict(color="red", dash="dash"), name="Início da Previsão")

    fig.update_layout(
        title="Previsão do Preço do Petróleo Brent",
        xaxis_title="Data",
        yaxis_title="Preço Brent (USD)",
        legend_title="Legenda",
        template="plotly_white"
    )
    fig.show()

# Pipeline principal
def main():
    # Configuração inicial
    api_key = "kjZJakW4UuR2fvPQPEoC5c1Ngyvfj96lnYUj9rcJ"
    start_date = "1987-05-25"
    end_date = "2024-11-30"
    prediction_days = 30  # Altere para 30, 60 ou 90 conforme necessário
    
    print("Coletando dados da API...")
    raw_data = fetch_brent_data(api_key, start_date, end_date)
    
    print("Total de registros coletados:", len(raw_data))

    print("Treinando o modelo e gerando previsões...")
    model, forecast, forecast_filtered, test = train_and_forecast(raw_data, prediction_days)

    print("Calculando métricas de teste...")
    metrics = calculate_metrics(test['y'].values, forecast_filtered['yhat'].values)
    print("Métricas do Modelo:")
    for key, value in metrics.items():
        print(f"{key}: {value:.2f}")
    
    print("Realizando validação cruzada...")
    performance = validate_model(model, horizon=f"{prediction_days} days")
    print("Métricas da Validação Cruzada:")
    print(performance)

    print("Plotando os resultados...")
    interactive_plot(raw_data, forecast, prediction_days)

# Execução
if __name__ == "__main__":
    main()
