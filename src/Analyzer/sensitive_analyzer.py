import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from tqdm import tqdm
from src.InventoryCore.inventory_core import ContinuousReviewInventory

class SensitivityAnalyzer:
    """
    Analiza la sensibilidad del sistema ante cambios en:
    - lambda_ (tasa de llegada)
    - L (tiempo de entrega)
    """
    
    @staticmethod
    def analyze(base_params, param_ranges, n_replicas=5):
        """
        Propósito:
        Identificar cómo cambios en parámetros clave afectan el desempeño del sistema.
        Esto ayuda a:
        - Entender puntos críticos del sistema
        - Prepararse para variaciones estacionales
        - Diseñar estrategias de contingencia
        """
        results = []
        for param_name, values in param_ranges.items():
            for value in tqdm(values, desc=f'Analizando {param_name}'):
                modified_params = base_params.copy()
                modified_params[param_name] = value
                
                # Usar política óptima previamente determinada
                sim = ContinuousReviewInventory(s=20, S=50, **modified_params)
                sim.run()
                metrics = sim.get_metrics()
                
                results.append({
                    'parameter': param_name,
                    'value': value,
                    'profit': metrics['average_profit_per_time'],
                    'service_level': metrics['service_level']
                })
        
        return pd.DataFrame(results)

    @staticmethod
    def plot_results(df):
        """Visualización de resultados de sensibilidad"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        sns.lineplot(data=df, x='value', y='profit', hue='parameter', 
                    style='parameter', markers=True, ax=ax1)
        ax1.set_title('Sensibilidad de la Ganancia Promedio')
        ax1.set_xlabel('Valor del Parámetro')
        ax1.set_ylabel('Ganancia por unidad de tiempo')
        
        sns.lineplot(data=df, x='value', y='service_level', hue='parameter',
                    style='parameter', markers=True, ax=ax2)
        ax2.set_title('Sensibilidad del Nivel de Servicio')
        ax2.set_xlabel('Valor del Parámetro')
        ax2.set_ylabel('Nivel de Servicio (1 - Pérdidas)')


"""
Interpretación:
- Pendiente pronunciada → Alta sensibilidad
- Cambios bruscos → Puntos de inflexión críticos
- Lambda: Aumentos generan más demanda pero también más pérdidas si no hay stock
- L: Mayores tiempos de entrega requieren mayores inventarios de seguridad
"""

