import numpy as np
import pandas as pd
from src.InventoryCore.inventory_core import ContinuousReviewInventory

class DemandDistributionAnalyzer:
    """
    Evalúa el impacto de diferentes distribuciones de demanda
    """
    
    DISTRIBUTIONS = {
        'Poisson': lambda lam: (lambda: np.random.poisson(lam)),
        'Normal': lambda mu, sigma: (lambda: max(0, int(np.random.normal(mu, sigma)))),
        'Uniform': lambda a, b: (lambda: np.random.randint(a, b+1)),
        'Geometric': lambda p: (lambda: np.random.geometric(p))
    }
    
    @classmethod
    def analyze(cls, base_params, distributions, n_replicas=10):
        """
        Versión corregida que asegura que demand_dist sea siempre una función
        """
        results = []
        for dist_name, dist_config in distributions.items():
            for _ in range(n_replicas):
                modified_params = base_params.copy()
                
                # Generar la función de demanda adecuada
                if dist_name == 'Poisson':
                    modified_params['demand_dist'] = cls.DISTRIBUTIONS[dist_name](dist_config['lambda'])
                elif dist_name == 'Normal':
                    modified_params['demand_dist'] = cls.DISTRIBUTIONS[dist_name](
                        dist_config['mu'], dist_config['sigma'])
                elif dist_name == 'Uniform':
                    modified_params['demand_dist'] = cls.DISTRIBUTIONS[dist_name](
                        dist_config['a'], dist_config['b'])
                # Agregar otros casos según necesites
                
                # Verificar que demand_dist sea callable
                if not callable(modified_params['demand_dist']):
                    raise ValueError(f"demand_dist no es callable para {dist_name}")
                
                sim = ContinuousReviewInventory(s=20, S=50, **modified_params)
                sim.run()
                metrics = sim.get_metrics()
                
                results.append({
                    'distribution': dist_name,
                    'config': str(dist_config),
                    **metrics
                })
        
        return pd.DataFrame(results)

"""
Interpretación:
- Distribuciones con colas pesadas (ej. Geometrica) → Requieren más inventario de seguridad
- Distribuciones simétricas (ej. Normal) → Más predecibles
- Variabilidad alta → Mayores costos por faltantes o excesos
"""
