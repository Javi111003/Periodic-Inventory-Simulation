import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
import ipywidgets as widgets
from IPython.display import display


class InventoryPlotter:
    
    @staticmethod
    def plot_comparison(df, group_col, metric_col):
        """Comparación entre grupos para una métrica específica"""
        plotter = InventoryPlotter() 
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=df, x=group_col, y=metric_col, ax=ax)
        ax.set_title(f'Comparación de {metric_col} por {group_col}')
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        
    @staticmethod
    def plot_bootstrap(results):
        """Visualización de resultados de bootstrap"""
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(results['distribution'], kde=True, ax=ax)
        ax.axvline(results['mean'], color='r', linestyle='--', label='Media')
        ax.axvline(results['ci_lower'], color='g', linestyle=':', label='Límite CI')
        ax.axvline(results['ci_upper'], color='g', linestyle=':')
        ax.set_title(f'Distribución Bootstrap (CI {results["ci_level"]}%)')
        ax.legend()
        plt.tight_layout()

    @staticmethod
    def plot_heatmap(resultados, metric='profit_mean', title='Análisis de Políticas (s,S)'):
        """
        Genera un heatmap de las políticas (s,S) con la métrica especificada
        
        Args:
            resultados (list/dict/DataFrame): Resultados del grid search
            metric (str): Métrica a visualizar ('profit_mean', 'service_level', etc.)
            title (str): Título del gráfico
            
        Returns:
            fig, ax: Figura y ejes matplotlib
        """
        df = pd.DataFrame(resultados)
        pivot = df.pivot(index='s', columns='S', values=metric)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(pivot, annot=True, fmt=".1f", cmap='viridis', ax=ax)
        ax.set_title(title)
        ax.set_xlabel('Nivel objetivo (S)')
        ax.set_ylabel('Punto de reorden (s)')
        plt.tight_layout()
    
    @staticmethod
    def plot_policy_3d(resultados, metric='profit_mean', elev=30, azim=45, interactive=False):
        """
        Visualización 3D mejorada con control de ángulo y rotación interactiva
        
        Args:
            resultados: Datos de entrada
            metric: Métrica a visualizar
            elev: Ángulo de elevación inicial (30 por defecto)
            azim: Ángulo azimutal inicial (45 por defecto)
            interactive: Si True, muestra controles para rotación
        """
        df = pd.DataFrame(resultados)
        df = df[df['S'] > df['s']]  # Solo combinaciones válidas
        
        # Crear malla
        s_vals = df['s'].unique()
        S_vals = df['S'].unique()
        s_grid, S_grid = np.meshgrid(s_vals, S_vals)
        
        # Preparar matriz Z
        z = np.full_like(s_grid, np.nan, dtype=float)
        for _, row in df.iterrows():
            i = np.where(s_vals == row['s'])[0][0]
            j = np.where(S_vals == row['S'])[0][0]
            z[j,i] = row[metric]

        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot de superficie
        surf = ax.plot_surface(
            s_grid, S_grid, z,
            cmap='viridis',
            edgecolor='k',
            alpha=0.8,
            linewidth=0.5,
            antialiased=True
        )
        
        # Puntos óptimos
        max_idx = df[metric].idxmax()
        ax.scatter(
            df.loc[max_idx, 's'], 
            df.loc[max_idx, 'S'], 
            df.loc[max_idx, metric],
            color='red',
            s=100,
            label=f'Óptimo: {df.loc[max_idx, metric]:.1f}'
        )
        
        # Configuración
        ax.set_xlabel('\ns', fontsize=12)
        ax.set_ylabel('\nS', fontsize=12)
        ax.set_zlabel(f'\n{metric}', fontsize=12)
        ax.set_title(f'Distribución 3D de Ganancia promedio por Política (s,S)', fontsize=14)
        ax.legend()
        
        # Barra de color
        fig.colorbar(surf, shrink=0.5, aspect=10, label=metric)
        
        # Configurar ángulo de vista
        ax.view_init(elev=elev, azim=azim)
        
        plt.tight_layout()
        
        if interactive:
            # Activar modo interactivo en Jupyter
            # O para ventana emergente interactiva:
            # %matplotlib qt
            
            # Widgets para controlar la vista
            def update_view(elevation=30, azimuth=45):
                ax.view_init(elev=elevation, azim=azimuth)
                fig.canvas.draw()
                
            widgets.interact(
                update_view,
                elevation=widgets.IntSlider(min=0, max=90, step=5, value=elev),
                azimuth=widgets.IntSlider(min=0, max=360, step=5, value=azim)
            )
        
        return fig, ax

    @staticmethod
    def plot_sensitivity(resultados, param_name, metric='profit_mean'):
        plt.figure(figsize=(10, 6))
        sns.lineplot(x=param_name, y=metric, data=resultados, marker='o')
        plt.title(f'Sensibilidad de Ganancia promiedo respecto a {param_name}')
        plt.grid(True)
        plt.show()

    @staticmethod
    def plot_inventory_trace(sim, window=100):
        # Este método necesitaría registrar el historial de inventario
        plt.figure(figsize=(12, 6))
        plt.plot(sim.inventory_history)
        plt.axhline(y=sim.s, color='r', linestyle='--', label='Punto de reorden (s)')
        plt.axhline(y=sim.S, color='g', linestyle='--', label='Nivel objetivo (S)')
        plt.title('Comportamiento del Inventario en el Tiempo')
        plt.xlabel('Eventos')
        plt.ylabel('Nivel de Inventario')
        plt.legend()
        plt.show()
        