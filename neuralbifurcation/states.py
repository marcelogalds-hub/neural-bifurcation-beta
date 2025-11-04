"""
states.py
===================

Classificador de estados de aprendizado
"""

import numpy as np
from enum import Enum

class SystemState(Enum):
    """Estados possíveis do treinamento"""
    INITIALIZATION = "inicializacao"
    LEARNING_FAST = "aprendizado_rapido"
    LEARNING_HEALTHY = "aprendizado_saudavel"
    OVERFITTING_EARLY = "overfitting_inicial"
    OVERFITTING_SEVERE = "overfitting_severo"
    PLATEAU = "plateau"
    UNDERFITTING = "underfitting"
    INSTABILITY = "instabilidade"
    COLLAPSE_IMMINENT = "colapso_iminente"

class StateClassifier:
    """
    Classifica estado atual do treinamento
    """
    
    def __init__(self, window_size: int = 5):
        self.window_size = window_size
    
    def classify(self, history: dict, epoch: int) -> SystemState:
        """
        Classifica estado no epoch atual
        
        Args:
            history: Histórico completo de métricas
            epoch: Epoch atual (0-indexed)
        
        Returns:
            SystemState
        """
        # Pegar janela recente
        start = max(0, epoch - self.window_size + 1)
        end = epoch + 1
        
        train_acc = history['train_acc'][start:end]
        val_acc = history['val_acc'][start:end]
        At = history['At'][start:end]
        Rt = history['Rt'][start:end]
        
        # Métricas agregadas
        train_mean = np.mean(train_acc)
        val_mean = np.mean(val_acc)
        gap = train_mean - val_mean
        
        # Tendências
        if len(val_acc) >= 3:
            val_trend = np.polyfit(range(len(val_acc)), val_acc, 1)[0]
            At_trend = np.polyfit(range(len(At)), At, 1)[0]
        else:
            val_trend = 0
            At_trend = 0
        
        # Taxa de melhora (% por epoch)
        if len(val_acc) >= 2:
            improvement_rate = (val_acc[-1] - val_acc[0]) / len(val_acc)
        else:
            improvement_rate = 0
        
        # Variância (para detectar instabilidade)
        At_std = np.std(At) if len(At) > 1 else 0
        At_mean = np.mean(At)
        At_cv = At_std / At_mean if At_mean > 0 else 0  # Coeficiente de variação
        
        # ====================================
        # CLASSIFICAÇÃO (ordem de prioridade)
        # ====================================
        
        # 1. Inicialização
        if epoch < 4:
            return SystemState.INITIALIZATION
        
        # 2. Colapso iminente
        if At_mean < 0.5 and gap > 0.5 and val_trend < -0.02:
            return SystemState.COLLAPSE_IMMINENT
        
        # 3. Overfitting severo
        if gap > 0.35 and val_trend < 0 and At_mean < 0.70:
            return SystemState.OVERFITTING_SEVERE
        
        # 4. Instabilidade
        if At_cv > 0.15:  # Variação > 15%
            return SystemState.INSTABILITY
        
        # 5. Underfitting
        if train_mean < 0.6 and val_mean < 0.6 and At_mean > 1.15:
            return SystemState.UNDERFITTING
        
        # 6. Overfitting inicial
        if gap > 0.20 and At_trend < -0.01 and At_mean < 0.85:
            return SystemState.OVERFITTING_EARLY
        
        # 7. Plateau
        if abs(improvement_rate) < 0.002 and epoch > 10:
            return SystemState.PLATEAU
        
        # 8. Aprendizado rápido
        if improvement_rate > 0.01 and 0.85 <= At_mean <= 1.15:
            return SystemState.LEARNING_FAST
        
        # 9. Aprendizado saudável
        if 0.003 <= improvement_rate <= 0.01 and 0.85 <= At_mean <= 1.15:
            return SystemState.LEARNING_HEALTHY
        
        # 10. Saudável (padrão)
        return SystemState.LEARNING_HEALTHY
    
    def get_state_info(self, state: SystemState) -> dict:
        """
        Retorna informações sobre um estado
        """
        info = {
            SystemState.INITIALIZATION: {
                'emoji': '🔧',
                'severity': 'info',
                'description': 'Calibrando baseline, aguarde',
                'action': 'continue',
                'urgency': 'none'
            },
            SystemState.LEARNING_FAST: {
                'emoji': '🚀',
                'severity': 'good',
                'description': 'Progresso rápido, ROI alto',
                'action': 'continue',
                'urgency': 'none',
                'note': 'Zona de maior eficiência!'
            },
            SystemState.LEARNING_HEALTHY: {
                'emoji': '✅',
                'severity': 'good',
                'description': 'Aprendizado saudável e estável',
                'action': 'continue',
                'urgency': 'none'
            },
            SystemState.OVERFITTING_EARLY: {
                'emoji': '⚠️',
                'severity': 'warning',
                'description': 'Overfitting detectado, agir preventivamente',
                'action': 'adjust',
                'urgency': 'medium',
                'suggestions': [
                    'Aumentar dropout (+0.1)',
                    'Aumentar weight_decay (×2)',
                    'Adicionar data augmentation'
                ]
            },
            SystemState.OVERFITTING_SEVERE: {
                'emoji': '🔴',
                'severity': 'critical',
                'description': 'Overfitting severo, parar recomendado',
                'action': 'stop',
                'urgency': 'high',
                'note': 'Continuar apenas prejudica o modelo'
            },
            SystemState.PLATEAU: {
                'emoji': '😴',
                'severity': 'warning',
                'description': 'Estagnação detectada, avaliar ROI',
                'action': 'decide',
                'urgency': 'medium',
                'suggestions': [
                    'Aumentar learning rate (×1.5)',
                    'OU parar se ROI baixo'
                ]
            },
            SystemState.UNDERFITTING: {
                'emoji': '📉',
                'severity': 'critical',
                'description': 'Modelo muito simples, precisa mais capacidade',
                'action': 'stop',
                'urgency': 'high',
                'suggestions': [
                    'Adicionar camadas',
                    'Aumentar unidades (×1.5-2)',
                    'Trocar arquitetura'
                ]
            },
            SystemState.INSTABILITY: {
                'emoji': '🌊',
                'severity': 'warning',
                'description': 'Treino instável, métricas oscilando',
                'action': 'adjust',
                'urgency': 'high',
                'suggestions': [
                    'Reduzir learning rate (÷2)',
                    'Aumentar batch size (×2)',
                    'Adicionar gradient clipping'
                ]
            },
            SystemState.COLLAPSE_IMMINENT: {
                'emoji': '💀',
                'severity': 'critical',
                'description': 'COLAPSO IMINENTE - parar imediatamente',
                'action': 'stop',
                'urgency': 'critical',
                'note': 'Modelo entrando em colapso catastrófico'
            }
        }
        
        return info.get(state, {
            'emoji': '❓',
            'severity': 'unknown',
            'description': 'Estado não classificado',
            'action': 'monitor',
            'urgency': 'low'
        })
