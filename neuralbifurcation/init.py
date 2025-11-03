"""
Neural Bifurcation Framework
=============================

Framework multi-objetivo de early stopping para Deep Learning.

Uso básico:
    >>> from neuralbifurcation import MultiObjectiveDetector
    >>> detector = MultiObjectiveDetector(objective='balanced')
    >>> model.fit(X, y, callbacks=[detector])

Objetivos disponíveis:
    - 'accuracy': Maximizar performance (tradicional)
    - 'robustness': Maximizar generalização
    - 'cost': Minimizar custo (ROI)
    - 'balanced': Equilíbrio entre tudo

Autor: Marcelo Galdino de Souza
Licença: MIT
"""

__version__ = "0.1.0-beta"
__author__ = "Marcelo Galdino de Souza"
__license__ = "MIT"

# Imports principais
from .detector import MultiObjectiveRegimeDetector
from .objectives import ObjectiveConfig, get_objective_config
from .states import SystemState, StateClassifier

# Alias para facilitar uso
MultiObjectiveDetector = MultiObjectiveRegimeDetector

# Exports públicos
__all__ = [
    "MultiObjectiveDetector",
    "MultiObjectiveRegimeDetector",
    "ObjectiveConfig",
    "get_objective_config",
    "SystemState",
    "StateClassifier",
]

# Mensagem de boas-vindas
def _show_welcome():
    """Mostra mensagem de boas-vindas (primeira vez)"""
    import os
    welcome_file = os.path.expanduser("~/.neuralbifurcation_welcome")
    
    if not os.path.exists(welcome_file):
        print("="*60)
        print("🎯 Neural Bifurcation Framework v" + __version__)
        print("="*60)
        print("Obrigado por testar! Esta é uma versão BETA.")
        print()
        print("📚 Guia rápido:")
        print("   - Exemplos: /exemplos/")
        print("   - Docs: LEIAME.md")
        print("   - Dúvidas: DUVIDAS_FREQUENTES.md")
        print()
        print("🐛 Bugs? Sugestões?")
        print("   Email: seu-email@exemplo.com")  # ← TROCAR
        print("="*60)
        print()
        
        # Criar arquivo para não mostrar novamente
        try:
            with open(welcome_file, 'w') as f:
                f.write(__version__)
        except:
            pass

# Mostrar boas-vindas (apenas primeira importação)
_show_welcome()
