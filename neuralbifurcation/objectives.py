"""
objectives.py
====================

Configurações para diferentes objetivos de treinamento.

Este módulo define objetivos pré-configurados para o Neural Bifurcation Framework.
Cada objetivo representa uma estratégia diferente de otimização durante o treinamento,
balanceando trade-offs entre accuracy, robustez, custo e descoberta de padrões.

Objetivos disponíveis:
    - accuracy: Maximiza performance em validação (tradicional)
    - robustness: Maximiza generalização e estabilidade
    - cost: Maximiza ROI (retorno por custo)
    - discovery: Busca padrões gerais (At > 1.0)
    - balanced: Balanço entre todos os objetivos

Example:
    >>> from neuralbifurcation.objectives import ObjectiveLibrary
    >>> config = ObjectiveLibrary.get_config('balanced')
    >>> print(config.name)
    'Balanced Multi-Objective'
"""

from dataclasses import dataclass
from typing import Callable, Dict, Optional


@dataclass
class ObjectiveConfig:
    """
    Configuração de um objetivo de treinamento.
    
    Esta classe encapsula todos os parâmetros necessários para definir um objetivo
    de otimização, incluindo métricas, tolerâncias e função de decisão.
    
    Attributes:
        name (str): Nome descritivo do objetivo.
        description (str): Descrição curta do objetivo.
        primary_metric (str): Métrica principal ('val_acc', 'At', 'roi', 'composite').
        threshold (Optional[float]): Valor mínimo aceitável para a métrica primária.
        tolerance (float): Quanta perda de val_acc é aceitável (0.0 a 1.0).
        decision_function (Callable): Função que calcula score a partir de métricas.
        recommendation (str): Texto com recomendações de uso.
    
    Example:
        >>> config = ObjectiveConfig(
        ...     name='Custom Objective',
        ...     description='My custom objective',
        ...     primary_metric='val_acc',
        ...     threshold=0.95,
        ...     tolerance=0.02,
        ...     decision_function=lambda m: m['val_acc'],
        ...     recommendation='Use when...'
        ... )
    """
    name: str
    description: str
    primary_metric: str
    threshold: Optional[float]
    tolerance: float
    decision_function: Callable
    recommendation: str


class ObjectiveLibrary:
    """
    Biblioteca de objetivos pré-configurados.
    
    Esta classe fornece acesso aos objetivos padrão do framework através de
    métodos estáticos. Não deve ser instanciada diretamente.
    """
    
    @staticmethod
    def get_config(objective: str) -> ObjectiveConfig:
        """
        Retorna configuração para um objetivo específico.
        
        Args:
            objective (str): Nome do objetivo. Opções: 'accuracy', 'robustness', 
                'cost', 'discovery', 'balanced'.
        
        Returns:
            ObjectiveConfig: Configuração completa do objetivo solicitado.
        
        Raises:
            ValueError: Se o objetivo não for reconhecido.
        
        Example:
            >>> config = ObjectiveLibrary.get_config('balanced')
            >>> print(config.primary_metric)
            'composite'
        """
        
        # Objetivo: Máxima accuracy em validação (abordagem tradicional)
        # Quando usar: Competições, benchmarks, prova de conceito
        # Trade-offs: Pode overfit, ser frágil ou caro
        configs = {
            'accuracy': ObjectiveConfig(
                name='Maximum Accuracy',
                description='Maximizar performance em validação (tradicional)',
                primary_metric='val_acc',
                threshold=None,
                tolerance=0.0,
                decision_function=lambda m: m['val_acc'],
                recommendation='''
                Use quando:
                  ✅ Competições (Kaggle, benchmarks)
                  ✅ Prova de conceito
                  ✅ Custo não é problema
                  
                Cuidado:
                  ⚠️ Pode overfit
                  ⚠️ Pode ser frágil
                  ⚠️ Pode ser caro
                '''
            ),
            
            # Objetivo: Máxima robustez e generalização
            # Quando usar: Aplicações críticas, produção, datasets pequenos
            # Trade-offs: Pode ter -5% accuracy mas ganha muito em estabilidade
            'robustness': ObjectiveConfig(
                name='Maximum Robustness',
                description='Maximizar generalização e estabilidade',
                primary_metric='At',
                threshold=0.90,
                tolerance=0.05,
                decision_function=lambda m: (
                    0.5 * m['At'] + 
                    0.3 * m['Rt'] + 
                    0.2 * m['val_acc']
                ),
                recommendation='''
                Use quando:
                  ✅ Aplicações críticas (medicina, finanças)
                  ✅ Produção (real-world)
                  ✅ Dataset pequeno/enviesado
                  ✅ Prioridade: confiabilidade
                  
                Trade-off:
                  ⚠️ Pode ter -5% accuracy
                  ✅ Muito mais estável
                  ✅ Generaliza melhor
                '''
            ),
            
            # Objetivo: Maximizar ROI (retorno por custo)
            # Quando usar: Orçamento limitado, produção em larga escala
            # Trade-offs: Para cedo, economiza 40-70% custo, modelo "bom o suficiente"
            'cost': ObjectiveConfig(
                name='Cost-Benefit Optimization',
                description='Maximizar ROI (retorno por custo)',
                primary_metric='roi',
                threshold=0.5,
                tolerance=0.10,
                decision_function=lambda m: (
                    0.6 * m['roi'] + 
                    0.4 * m['val_acc']
                ),
                recommendation='''
                Use quando:
                  ✅ Orçamento limitado
                  ✅ Produção em larga escala
                  ✅ Deadline apertado
                  ✅ Iteração rápida importante
                  
                Benefício:
                  ✅ Para treino cedo
                  ✅ Economiza 40-70% custo
                  ✅ Modelo "bom o suficiente"
                '''
            ),
            
            # Objetivo: Descobrir padrões gerais (At > 1.0 é DESEJADO)
            # Quando usar: Pesquisa, transfer learning, features universais
            # Trade-offs: Aceita -15% accuracy para aprender padrões gerais
            'discovery': ObjectiveConfig(
                name='Pattern Discovery',
                description='Descobrir padrões gerais (At > 1.0 desejável)',
                primary_metric='At',
                threshold=1.0,
                tolerance=0.15,
                decision_function=lambda m: (
                    0.7 * m['At'] + 
                    0.3 * max(m['val_acc'], 0.5)
                ),
                recommendation='''
                Use quando:
                  ✅ Pesquisa exploratória
                  ✅ Transfer learning futuro
                  ✅ Busca por features universais
                  ✅ Dataset é proxy do problema real
                  
                Objetivo:
                  ✅ At > 1.0 é DESEJADO!
                  ✅ Modelo aprende padrões GERAIS
                  ✅ Generaliza ALÉM do treino
                '''
            ),
            
            # Objetivo: Balanço entre accuracy, robustez e custo
            # Quando usar: Propósito geral, não sabe qual priorizar
            # Trade-offs: Compromisso equilibrado, evita extremos
            'balanced': ObjectiveConfig(
                name='Balanced Multi-Objective',
                description='Balanço entre accuracy, robustez e custo',
                primary_metric='composite',
                threshold=None,
                tolerance=0.03,
                decision_function=lambda m: (
                    0.40 * m['val_acc'] + 
                    0.25 * m['At'] + 
                    0.20 * m['Rt'] + 
                    0.15 * min(m['roi'] / 2.0, 1.0)
                ),
                recommendation='''
                Use quando:
                  ✅ Propósito geral
                  ✅ Não sabe qual priorizar
                  ✅ Exploração inicial
                  
                Comportamento:
                  ✅ Compromisso entre objetivos
                  ✅ Evita extremos
                  ✅ Seguro como padrão
                '''
            )
        }
        
        if objective not in configs:
            raise ValueError(f"Objetivo '{objective}' não reconhecido. "
                           f"Opções: {list(configs.keys())}")
        
        return configs[objective]


class ObjectiveComparator:
    """
    Compara resultados entre diferentes objetivos.
    
    Esta classe fornece métodos para analisar o histórico de treinamento e
    determinar qual teria sido o melhor epoch para cada objetivo disponível.
    """
    
    @staticmethod
    def compare_all(history: Dict, epoch_range: range = None) -> Dict:
        """
        Encontra o melhor epoch para cada objetivo disponível.
        
        Analisa o histórico de treinamento e determina, retroativamente, qual teria
        sido o melhor epoch se cada objetivo tivesse sido usado.
        
        Args:
            history (Dict): Histórico de treinamento contendo métricas por epoch.
                Deve conter as chaves: 'val_acc', 'At', 'Rt', 'roi' (opcional).
            epoch_range (range, optional): Range de epochs a considerar. Se None,
                considera todos os epochs disponíveis. Default: None.
        
        Returns:
            Dict: Dicionário mapeando nome do objetivo para seus resultados:
                - best_epoch (int): Melhor epoch (0-indexed)
                - best_score (float): Score nesse epoch
                - metrics_at_best (Dict): Todas as métricas nesse epoch
        
        Example:
            >>> history = {'val_acc': [...], 'At': [...], 'Rt': [...]}
            >>> results = ObjectiveComparator.compare_all(history)
            >>> print(results['balanced']['best_epoch'])
            15
        """
        if epoch_range is None:
            epoch_range = range(len(history['val_acc']))
        
        objectives = ['accuracy', 'robustness', 'cost', 'discovery', 'balanced']
        results = {}
        
        for obj in objectives:
            config = ObjectiveLibrary.get_config(obj)
            best_epoch, best_score = ObjectiveComparator._find_best_epoch(
                history, config, epoch_range
            )
            
            results[obj] = {
                'best_epoch': best_epoch,
                'best_score': best_score,
                'metrics_at_best': {
                    'val_acc': history['val_acc'][best_epoch],
                    'At': history['At'][best_epoch],
                    'Rt': history['Rt'][best_epoch],
                    'roi': history.get('roi', [0] * len(history['val_acc']))[best_epoch]
                }
            }
        
        return results
    
    @staticmethod
    def _find_best_epoch(history: Dict, config: ObjectiveConfig, epoch_range: range) -> tuple:
        """
        Encontra o melhor epoch baseado na função de decisão do objetivo.
        
        Args:
            history (Dict): Histórico de treinamento com métricas.
            config (ObjectiveConfig): Configuração do objetivo a avaliar.
            epoch_range (range): Range de epochs a considerar.
        
        Returns:
            tuple: (best_epoch, best_score) onde:
                - best_epoch (int): Epoch com melhor score
                - best_score (float): Melhor score encontrado
        """
        best_epoch = 0
        best_score = -float('inf')
        
        for epoch in epoch_range:
            metrics = {
                'val_acc': history['val_acc'][epoch],
                'At': history['At'][epoch],
                'Rt': history['Rt'][epoch],
                'roi': history.get('roi', [0] * len(history['val_acc']))[epoch]
            }
            
            score = config.decision_function(metrics)
            
            if score > best_score:
                best_score = score
                best_epoch = epoch
        
        return best_epoch, best_score
