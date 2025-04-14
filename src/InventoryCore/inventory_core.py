import heapq
import numpy as np

class ContinuousReviewInventory:
    def __init__(self, s, S, lambda_, lead_time, demand_dist, order_cost_func,
                 holding_cost, unit_revenue, time_horizon, seed=None):
        if s >= S:
            raise ValueError("s debe ser menor que S")
        self.s = s
        self.S = S
        self.lambda_ = lambda_  # Tasa de llegada de clientes
        self.lead_time = lead_time  # Tiempo de entrega
        self.demand_dist = demand_dist  # Función que genera la demanda
        self.order_cost_func = order_cost_func  # Función de costo de pedido
        self.holding_cost = holding_cost  # Costo de mantenimiento por unidad
        self.unit_revenue = unit_revenue  # Ingreso por unidad vendida
        self.T = time_horizon  # Horizonte de tiempo

        # Estado inicial
        self.inventory = S  
        self.on_order = 0  
        self.current_time = 0.0

        # Cola de eventos: (tiempo, tipo, datos)
        self.event_heap = []
        # Programar primera llegada
        self._schedule_next_arrival()

        # Estadísticas
        self.total_ordering_cost = 0.0
        self.total_holding_cost = 0.0
        self.total_revenue = 0.0
        self.last_update_time = 0.0  # Para cálculo de holding cost
        self.lost_clients = 0 # Clientes con demanda insatisfecha
        self.lost_units = 0 # Unidades no vendidas por falta de inventario
        self.lost_revenue = 0.0 # Ganancia no obtenida por unidades perdidas

        if seed is not None:
            np.random.seed(seed)
    
    def reset(self):
        """Reinicia el estado de la simulación manteniendo los mismos parámetros"""
        self.inventory = self.S  # Inventario inicial
        self.on_order = 0        # Pedidos pendientes
        self.current_time = 0.0  # Reloj de simulación
        
        # Reiniciar estadísticas
        self.total_ordering_cost = 0.0
        self.total_holding_cost = 0.0
        self.total_revenue = 0.0
        self.last_update_time = 0.0
        self.lost_clients = 0
        self.lost_units = 0
        self.lost_revenue = 0.0
        
        # Reiniciar eventos
        self.event_heap = []
        self._schedule_next_arrival()
    
    def _schedule_next_arrival(self):
        """ Programa la próxima llegada usando el proceso de Poisson. """
        inter_arrival = np.random.exponential(1 / self.lambda_)
        new_time = self.current_time + inter_arrival
        heapq.heappush(self.event_heap, (new_time, 'ARRIVAL', None))

    def _schedule_delivery(self, order_quantity):
        """ Programa la entrega de un pedido. """
        delivery_time = self.current_time + self.lead_time
        heapq.heappush(self.event_heap, (delivery_time, 'DELIVERY', order_quantity))

    def _update_holding_cost(self):
        """ Actualiza el costo de mantenimiento acumulado. """
        delta = self.current_time - self.last_update_time
        self.total_holding_cost += self.inventory * delta * self.holding_cost
        self.last_update_time = self.current_time

    def run(self):
        """ Ejecuta la simulación hasta el horizonte temporal T. """
        while self.event_heap and self.current_time <= self.T:
            event_time, event_type, data = heapq.heappop(self.event_heap)
            
            # Detener si el evento es posterior a T
            if event_time > self.T:
                break
            
            # Actualizar costos hasta el tiempo del evento
            self._update_holding_cost()
            self.current_time = event_time

            if event_type == 'ARRIVAL':
                self._handle_arrival()
            elif event_type == 'DELIVERY':
                self._handle_delivery(data)

        # Actualizar holding cost hasta T si es necesario
        if self.current_time < self.T:
            self._update_holding_cost()
            self.current_time = self.T

    def _handle_arrival(self):
        """ Procesa una llegada de cliente. """
        demand = self.demand_dist()
        sold = min(demand, self.inventory)
        self.inventory -= sold
        self.total_revenue += sold * self.unit_revenue
        lost = demand - sold
        if lost > 0:
            self.lost_clients += 1
            self.lost_units += lost
            self.lost_revenue += lost * self.unit_revenue

        # Verificar si se debe hacer pedido
        if self.inventory < self.s and self.on_order == 0:
            order_qty = self.S - self.inventory
            self.on_order = order_qty
            self._schedule_delivery(order_qty)

        # Programar próxima llegada
        self._schedule_next_arrival()

    def _handle_delivery(self, quantity):
        """ Procesa la entrega de un pedido. """
        self.inventory += quantity
        self.total_ordering_cost += self.order_cost_func(quantity)
        self.on_order = 0  # Resetear pedidos pendientes

    def get_metrics(self):
        """ Retorna métricas clave de la simulación. """
        profit = self.total_revenue - self.total_ordering_cost - self.total_holding_cost
        avg_profit = profit / self.T if self.T > 0 else 0
        return {
            "average_profit_per_time": avg_profit,
            "total_profit": profit,
            "holding_cost": self.total_holding_cost,
            "ordering_cost": self.total_ordering_cost,
            "revenue": self.total_revenue,
            "lost_clients": self.lost_clients,
            "lost_units": self.lost_units,
            "lost_revenue": self.lost_revenue,
            "service_level": 1 - self.lost_units / (self.lost_units + (self.total_revenue / self.unit_revenue)) if (self.lost_units + self.total_revenue) > 0 else 1.0,
            "final_inventory": self.inventory
        }