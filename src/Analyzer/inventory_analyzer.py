import numpy as np
from tqdm import tqdm
from src.InventoryCore.inventory_core import ContinuousReviewInventory

class InventoryAnalyzer:
    @staticmethod
    def grid_search(s_values, S_values, params, n_replicas=10):
        """
        Realiza una búsqueda en grid sobre los valores de s y S
        """
        resultados = []
        for s in tqdm(s_values, desc='Analizando s'):
            for S in S_values:
                if S <= s:
                    continue
                metrics = []
                for _ in range(n_replicas):
                    sim = ContinuousReviewInventory(s=s, S=S, **params)
                    sim.run()
                    m = sim.get_metrics()
                    metrics.append(m['average_profit_per_time'])
                
                resultado = {
                    's': s,
                    'S': S,
                    'profit_mean': np.mean(metrics),
                    'profit_std': np.std(metrics)
                    #'service_level': np.mean([m['service_level'] for m in metrics])
                }
                resultados.append(resultado)
        return resultados