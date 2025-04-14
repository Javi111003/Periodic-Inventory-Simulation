import pandas as pd
from src.InventoryCore.inventory_core import ContinuousReviewInventory


class CostStructureAnalyzer:
    """
    Evalúa diferentes estructuras de costos de pedido
    """
    
    @staticmethod
    def analyze(base_params, cost_structures, n_replicas=10):
        """
        Propósito:
        Entender cómo la estructura de costos impacta:
        - Frecuencia óptima de pedidos
        - Tamaño económico de pedido
        - Rentabilidad global
        
        Permite negociar mejores términos con proveedores.
        """
        results = []
        for cost_name, cost_func in cost_structures.items():
            for _ in range(n_replicas):
                modified_params = base_params.copy()
                modified_params['order_cost_func'] = cost_func
                
                sim = ContinuousReviewInventory(s=20, S=50, **modified_params)
                sim.run()
                metrics = sim.get_metrics()
                
                results.append({
                    'cost_structure': cost_name,
                    **metrics
                })
        
        return pd.DataFrame(results)

"""
Interpretación:
- Costos fijos altos → Pedidos menos frecuentes y más grandes
- Costos variables dominantes → Pedidos más frecuentes
- Descuentos por volumen pueden justificar mayores inventarios
"""