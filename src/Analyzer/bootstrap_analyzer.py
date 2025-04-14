import numpy as np 
from tqdm import tqdm

class BootstrapAnalyzer:
    """
    Realiza análisis de bootstrap para estimar intervalos de confianza
    """
    
    @staticmethod
    def analyze(simulator, n_bootstrap=1000, ci=95):
        """
        Propósito:
        Cuantificar la incertidumbre en las métricas estimadas.
        Proporciona:
        - Intervalos de confianza robustos
        - Estimación no paramétrica de variabilidad
        - Validación de resultados
        """
        stats = []
        for _ in tqdm(range(n_bootstrap), desc='Bootstrap'):
            simulator.reset()
            simulator.run()
            metrics = simulator.get_metrics()
            stats.append(metrics['average_profit_per_time'])
        
        lower = np.percentile(stats, (100 - ci)/2)
        upper = np.percentile(stats, 100 - (100 - ci)/2)
        
        return {
            'mean': np.mean(stats),
            'ci_lower': lower,
            'ci_upper': upper,
            'ci_level': ci,
            'distribution': stats
        }
    
    @staticmethod
    def bootstrap_ci(data, n_bootstrap=1000, ci=95):
        """Calcula intervalos de confianza usando bootstrap"""
        bootstraps = np.zeros(n_bootstrap)
        for i in range(n_bootstrap):
            sample = np.random.choice(data, size=len(data), replace=True)
            bootstraps[i] = np.mean(sample)
        lower = np.percentile(bootstraps, (100 - ci)/2)
        upper = np.percentile(bootstraps, 100 - (100 - ci)/2)
        return lower, upper

"""
Interpretación:
- Intervalos estrechos → Alta confiabilidad en resultados
- Intervalos amplios → Necesidad de más réplicas
- Asimetría en la distribución → Posibles valores atípicos
"""