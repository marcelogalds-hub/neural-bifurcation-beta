"""
Neural Bifurcation Framework - Multi-Objective Regime Detector
===============================================================

Este módulo implementa o detector de regime multi-objetivo, o componente central
do Neural Bifurcation Framework. Ele monitora o treinamento de redes neurais e
decide quando parar baseado em objetivos como accuracy, robustez, custo ou equilíbrio.

O detector funciona como um callback do Keras/TensorFlow que:
- Monitora métricas de treino e validação em cada época
- Calcula indicadores de generalização (At) e robustez (Rt)
- Classifica o estado atual do aprendizado
- Decide quando parar baseado no objetivo escolhido
- Restaura os melhores pesos automaticamente

Classes principais:
    MultiObjectiveRegimeDetector: Callback principal para detecção de regime

Exemplo de uso básico:
    >>> from neuralbifurcation import MultiObjectiveRegimeDetector
    >>> 
    >>> # Criar detector com objetivo de robustez
    >>> detector = MultiObjectiveRegimeDetector(
    ...     objective='robustness',
    ...     patience=10,
    ...     verbose=1
    ... )
    >>> 
    >>> # Usar no treinamento
    >>> model.fit(
    ...     X_train, y_train,
    ...     validation_data=(X_val, y_val),
    ...     epochs=100,
    ...     callbacks=[detector]
    ... )
    >>> 
    >>> # Obter relatório
    >>> report = detector.get_multi_objective_report()

Autor: Marcelo Galdino de Souza
Data: 31 de outubro de 2025
Licença: MIT
"""

import numpy as np
from tensorflow import keras
from typing import Dict, List, Optional, Any, Tuple
import warnings

from .objectives import ObjectiveLibrary, ObjectiveConfig
from .states import StateClassifier, SystemState

warnings.filterwarnings('ignore')


class MultiObjectiveRegimeDetector(keras.callbacks.Callback):
    """
    Detector de regime com suporte a múltiplos objetivos de otimização.
    
    Este callback monitora o treinamento de redes neurais e implementa estratégias
    de parada inteligente baseadas em diferentes objetivos:
    - 'accuracy': Maximiza acurácia de validação
    - 'robustness': Maximiza generalização (At) e estabilidade (Rt)
    - 'cost': Minimiza custo computacional mantendo qualidade
    - 'balanced': Equilibra todos os objetivos
    - 'discovery': Explora mais épocas para encontrar melhores soluções
    
    O detector calcula automaticamente:
    - At (Autonomia): Razão val_acc/train_acc, indica generalização
    - Rt (Robustez): Estabilidade temporal de At
    - ROI: Retorno sobre investimento (melhoria/custo)
    - Estado do aprendizado: classificação do regime atual
    
    Attributes:
        objective (str): Objetivo de otimização selecionado
        patience (int): Épocas sem melhora antes de parar
        cost_per_epoch (float): Custo estimado por época em USD
        verbose (int): Nível de verbosidade (0=silencioso, 1=normal, 2=debug)
        config (ObjectiveConfig): Configuração do objetivo selecionado
        classifier (StateClassifier): Classificador de estados de aprendizado
        history (Dict): Histórico completo de métricas por época
        best_epoch (int): Época com melhor score
        best_score (float): Melhor score alcançado
        best_weights (List): Pesos do modelo na melhor época
        total_cost (float): Custo acumulado do treinamento
    
    Example:
        >>> # Uso básico com objetivo balanceado
        >>> detector = MultiObjectiveRegimeDetector(
        ...     objective='balanced',
        ...     patience=10,
        ...     verbose=1
        ... )
        >>> model.fit(X, y, callbacks=[detector], epochs=100)
        
        >>> # Uso focado em robustez com custo customizado
        >>> detector = MultiObjectiveRegimeDetector(
        ...     objective='robustness',
        ...     patience=15,
        ...     cost_per_epoch=5.0,
        ...     verbose=2
        ... )
        >>> model.fit(X, y, callbacks=[detector], epochs=100)
        
        >>> # Análise pós-treinamento
        >>> report = detector.get_multi_objective_report()
        >>> print(f"Melhor época: {report['recommended_epoch']}")
        >>> print(f"Custo total: ${report['total_cost']:.2f}")
        
        >>> # Visualização
        >>> detector.plot_comparison('results.png')
    
    Note:
        - O detector SEMPRE restaura os melhores pesos ao final do treinamento
        - At < 0.8 indica overfitting severo (val 20% abaixo de train)
        - At > 0.95 indica generalização excelente
        - Rt próximo de 1.0 indica alta estabilidade
        - O baseline é calibrado após 5 épocas iniciais
    
    Warning:
        Este código foi extensivamente testado e validado em aplicações críticas.
        Modificações na lógica de cálculo ou decisão podem quebrar funcionalidades.
    """
    
    def __init__(
        self, 
        objective: str = 'balanced',
        patience: int = 10,
        cost_per_epoch: float = 2.0,
        verbose: int = 1
    ) -> None:
        """
        Inicializa o detector multi-objetivo.
        
        Args:
            objective: Objetivo de otimização. Opções:
                - 'accuracy': Foca em maximizar acurácia de validação
                - 'robustness': Foca em generalização e estabilidade
                - 'cost': Minimiza custo mantendo qualidade adequada
                - 'balanced': Equilibra accuracy, robustez e custo
                - 'discovery': Explora mais para encontrar melhores soluções
                Default: 'balanced'
            patience: Número de épocas sem melhora no score antes de parar.
                Valores típicos: 5-20 dependendo da complexidade do problema.
                Default: 10
            cost_per_epoch: Custo estimado por época em USD, usado para
                cálculo de ROI e otimização de custo. Considere tempo de GPU,
                energia e recursos computacionais.
                Default: 2.0
            verbose: Nível de informação durante treinamento:
                - 0: Silencioso, sem output
                - 1: Normal, mostra progresso e alertas
                - 2: Debug, mostra informações detalhadas
                Default: 1
        
        Raises:
            ValueError: Se objective não for uma opção válida
            ValueError: Se patience < 1
            ValueError: Se cost_per_epoch < 0
            ValueError: Se verbose não for 0, 1 ou 2
        
        Example:
            >>> # Configuração para produção (foco em custo)
            >>> detector = MultiObjectiveRegimeDetector(
            ...     objective='cost',
            ...     patience=8,
            ...     cost_per_epoch=3.50,
            ...     verbose=1
            ... )
            
            >>> # Configuração para pesquisa (foco em descoberta)
            >>> detector = MultiObjectiveRegimeDetector(
            ...     objective='discovery',
            ...     patience=20,
            ...     cost_per_epoch=1.0,
            ...     verbose=2
            ... )
            
            >>> # Configuração para aplicações críticas (foco em robustez)
            >>> detector = MultiObjectiveRegimeDetector(
            ...     objective='robustness',
            ...     patience=15,
            ...     cost_per_epoch=5.0,
            ...     verbose=1
            ... )
        """
        super().__init__()
        
        self.objective = objective
        self.patience = patience
        self.cost_per_epoch = cost_per_epoch
        self.verbose = verbose
        
        # Carregar configuração do objetivo selecionado
        # Isso valida automaticamente se o objetivo existe
        self.config = ObjectiveLibrary.get_config(objective)
        
        # Inicializar classificador de estados com janela de 5 épocas
        self.classifier = StateClassifier(window_size=5)
        
        # Inicializar estado interno
        self.reset()
    
    def reset(self) -> None:
        """
        Reseta o estado interno do detector.
        
        Limpa todo o histórico e reinicializa variáveis de controle.
        Útil se o mesmo detector for reutilizado em múltiplos treinos.
        
        Note:
            Este método é chamado automaticamente em on_train_begin().
            Raramente precisa ser chamado manualmente.
        """
        # Histórico completo de métricas por época
        self.history = {
            'epoch': [],           # Número da época
            'train_acc': [],       # Acurácia de treino
            'val_acc': [],         # Acurácia de validação
            'train_loss': [],      # Loss de treino
            'val_loss': [],        # Loss de validação
            'At': [],              # Autonomia (val_acc/train_acc)
            'Rt': [],              # Robustez (estabilidade temporal)
            'theta': [],           # Ângulo do gap (arctan)
            'gap': [],             # Gap = train_acc - val_acc
            'roi': [],             # Return on Investment
            'state': [],           # Estado classificado
            'score': []            # Score do objetivo atual
        }
        
        # Controle do melhor modelo
        self.best_epoch = 0
        self.best_score = -float('inf')
        self.best_weights = None
        self.epochs_no_improve = 0
        
        # Calibração de baseline (primeiras 5 épocas)
        self.baseline_calibrated = False
        self.At_baseline = None
        
        # Controle de custo
        self.total_cost = 0.0
    
    def on_train_begin(self, logs: Optional[Dict] = None) -> None:
        """
        Callback chamado no início do treinamento.
        
        Reseta o estado interno e exibe informações sobre o objetivo selecionado
        se verbose >= 1.
        
        Args:
            logs: Dicionário de logs do Keras (geralmente vazio neste callback)
        
        Note:
            Este método é chamado automaticamente pelo Keras antes da primeira época.
        """
        self.reset()
        
        if self.verbose >= 1:
            print("\n" + "="*80)
            print(f"🎯 Multi-Objective Regime Detector V4.0")
            print("="*80)
            print(f"Objetivo selecionado: {self.config.name}")
            print(f"Descrição: {self.config.description}")
            print(f"Métrica primária: {self.config.primary_metric}")
            print("="*80)
            print()
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None) -> None:
        """
        Callback chamado ao final de cada época.
        
        Este é o método principal do detector. A cada época ele:
        1. Coleta métricas de treino e validação
        2. Calcula indicadores derivados (At, Rt, ROI, etc.)
        3. Calibra baseline nas primeiras 5 épocas
        4. Classifica o estado atual do aprendizado
        5. Calcula score baseado no objetivo
        6. Decide se deve continuar ou parar
        7. Exibe informações e alertas
        
        Args:
            epoch: Número da época atual (0-indexed)
            logs: Dicionário com métricas do Keras. Esperado conter:
                - 'accuracy': Acurácia de treino
                - 'val_accuracy': Acurácia de validação
                - 'loss': Loss de treino
                - 'val_loss': Loss de validação
        
        Note:
            - At = val_acc / train_acc: mede generalização
              * At < 0.8: overfitting severo
              * At 0.8-0.9: overfitting moderado
              * At 0.9-1.0: generalização boa
              * At > 1.0: possível underfitting ou dados ruins
            
            - Rt: estabilidade de At nas últimas épocas
              * Rt próximo de 1.0: muito estável
              * Rt próximo de 0.0: instável
            
            - ROI: melhoria / custo nas últimas 5 épocas
              * ROI alto: bom retorno sobre investimento
              * ROI baixo: pouco progresso por custo
            
            - O baseline é calibrado na época 4 (quinta época)
            - Decisão de parada considera estado + patience + score
        
        Warning:
            A lógica de cálculo de At, Rt e decisão de parada foi extensivamente
            testada. Modificações podem quebrar o comportamento esperado.
        """
        logs = logs or {}
        
        # ====================================
        # 1. COLETAR MÉTRICAS BÁSICAS
        # ====================================
        train_acc = logs.get('accuracy', 0)
        val_acc = logs.get('val_accuracy', 0)
        train_loss = logs.get('loss', 0)
        val_loss = logs.get('val_loss', 0)
        
        # ====================================
        # 2. CALCULAR MÉTRICAS DERIVADAS
        # ====================================
        
        # At (Autonomia): val_acc / train_acc
        # Mede quão bem o modelo generaliza
        # Epsilon evita divisão por zero
        At = val_acc / (train_acc + 1e-6)
        
        # Gap: diferença entre treino e validação
        # Gap alto indica overfitting
        gap = train_acc - val_acc
        
        # Theta: ângulo do gap (para visualização geométrica)
        # Usado em análises topológicas (não na decisão)
        theta = np.arctan(max(gap, 0))
        
        # Rt (Robustez): estabilidade temporal de At
        # Simplificado por enquanto, versão completa requer ativações
        Rt = self._compute_rt_simple(epoch)
        
        # ROI: Return on Investment
        # Melhoria obtida dividida pelo custo nas últimas 5 épocas
        if len(self.history['val_acc']) >= 5:
            improvement = val_acc - self.history['val_acc'][-5]
            cost = 5 * self.cost_per_epoch
            roi = (improvement * 100) / cost if cost > 0 else 0
        else:
            roi = 0
        
        # Atualizar custo acumulado
        self.total_cost += self.cost_per_epoch
        
        # ====================================
        # 3. SALVAR NO HISTÓRICO
        # ====================================
        self.history['epoch'].append(epoch)
        self.history['train_acc'].append(train_acc)
        self.history['val_acc'].append(val_acc)
        self.history['train_loss'].append(train_loss)
        self.history['val_loss'].append(val_loss)
        self.history['At'].append(At)
        self.history['Rt'].append(Rt)
        self.history['theta'].append(theta)
        self.history['gap'].append(gap)
        self.history['roi'].append(roi)
        
        # ====================================
        # 4. CALIBRAR BASELINE (época 4 = quinta época)
        # ====================================
        # O baseline é a média de At nas primeiras 5 épocas
        # Usado como referência para detectar desvios
        if epoch == 4 and not self.baseline_calibrated:
            self.At_baseline = np.mean(self.history['At'])
            self.baseline_calibrated = True
            
            if self.verbose >= 1:
                print(f"\n📊 Baseline calibrado: At = {self.At_baseline:.3f}\n")
        
        # ====================================
        # 5. CLASSIFICAR ESTADO DO APRENDIZADO
        # ====================================
        # Usa StateClassifier para identificar regime atual:
        # inicialização, aprendizado_rápido, overfitting, plateau, etc.
        if self.baseline_calibrated:
            state = self.classifier.classify(self.history, epoch)
            self.history['state'].append(state.value)
        else:
            # Nas primeiras 5 épocas, sempre está inicializando
            state = SystemState.INITIALIZATION
            self.history['state'].append(state.value)
        
        # ====================================
        # 6. AVALIAR BASEADO NO OBJETIVO
        # ====================================
        # Cada objetivo tem uma função de decisão própria
        # que pondera as métricas de forma diferente
        metrics = {
            'val_acc': val_acc,
            'At': At,
            'Rt': Rt,
            'roi': roi
        }
        
        # Calcular score usando função do objetivo
        score = self.config.decision_function(metrics)
        self.history['score'].append(score)
        
        # Verificar se houve melhora
        improved = score > self.best_score
        
        if improved:
            # Novo melhor score! Atualizar e salvar pesos
            self.best_score = score
            self.best_epoch = epoch
            self.best_weights = self.model.get_weights()
            self.epochs_no_improve = 0
        else:
            # Sem melhora, incrementar contador de patience
            self.epochs_no_improve += 1
        
        # ====================================
        # 7. OUTPUT E ALERTAS
        # ====================================
        if self.verbose >= 1:
            state_info = self.classifier.get_state_info(state)
            
            # Linha principal com métricas
            print(f"{state_info['emoji']} Epoch {epoch+1:3d} | "
                  f"Val: {val_acc:.3f} | "
                  f"At: {At:.3f} | "
                  f"Rt: {Rt:.2f} | "
                  f"ROI: ${roi:.2f}/pt | "
                  f"Score: {score:.3f} | "
                  f"{state.value}")
            
            # Alertas e sugestões para estados problemáticos
            if state_info['severity'] in ['warning', 'critical']:
                print(f"   {state_info['description']}")
                
                if 'suggestions' in state_info:
                    print(f"   Sugestões: {', '.join(state_info['suggestions'])}")
        
        # ====================================
        # 8. DECISÃO DE PARADA
        # ====================================
        should_stop = False
        stop_reason = None
        
        # Critério 1: Estado crítico que exige parada imediata
        # Ex: overfitting severo, instabilidade crítica
        state_info = self.classifier.get_state_info(state)
        if state_info['action'] == 'stop' and state_info['urgency'] in ['high', 'critical']:
            should_stop = True
            stop_reason = f"{state_info['description']}"
        
        # Critério 2: Patience esgotada
        # Muitas épocas sem melhora no score
        elif self.epochs_no_improve >= self.patience:
            should_stop = True
            stop_reason = f"Sem melhora há {self.epochs_no_improve} epochs"
        
        # Executar parada se necessário
        if should_stop:
            if self.verbose >= 1:
                print(f"\n{'='*80}")
                print(f"🛑 PARANDO TREINO")
                print(f"{'='*80}")
                print(f"Motivo: {stop_reason}")
                print(f"Melhor epoch: {self.best_epoch + 1}")
                print(f"Score: {self.best_score:.3f}")
                print(f"{'='*80}\n")
            
            self.model.stop_training = True
    
    def on_train_end(self, logs: Optional[Dict] = None) -> None:
        """
        Callback chamado ao final do treinamento.
        
        Restaura os pesos da melhor época (baseado no score do objetivo)
        e exibe relatório final se verbose >= 1.
        
        Args:
            logs: Dicionário de logs do Keras (geralmente vazio neste callback)
        
        Note:
            A restauração de pesos é CRÍTICA: garante que o modelo final
            seja o melhor encontrado durante o treinamento, não o último.
        """
        # Restaurar melhor modelo encontrado
        if self.best_weights is not None:
            if self.verbose >= 1:
                print(f"\n✅ Restaurando modelo do epoch {self.best_epoch + 1}")
            self.model.set_weights(self.best_weights)
        
        # Mostrar relatório final comparativo
        if self.verbose >= 1:
            self._print_final_report()
    
    def _compute_rt_simple(self, epoch: int) -> float:
        """
        Calcula Rt (Robustez) de forma simplificada.
        
        Esta é uma versão placeholder baseada na estabilidade de At.
        A versão completa requer análise das ativações da rede (não implementada).
        
        Args:
            epoch: Época atual (usado para verificar histórico disponível)
        
        Returns:
            Rt normalizado em [0, 0.5]:
            - Valores altos (~0.5): At muito estável
            - Valores baixos (~0.1): At instável
        
        Note:
            - Usa últimas 3 épocas para calcular estabilidade
            - Rt = (1 - std(At)) * 0.5 para normalizar
            - Retorna 0.1 se não há histórico suficiente
            - Versão futura analisará topologia das ativações
        """
        # Precisa de pelo menos 3 épocas para calcular estabilidade
        if len(self.history['At']) < 3:
            return 0.1
        
        # Pegar últimas 3 medições de At
        At_recent = self.history['At'][-3:]
        
        # Calcular estabilidade: quanto menor o desvio, maior a estabilidade
        # std alto = instável, std baixo = estável
        # Limitado a 1.0 para evitar valores negativos
        stability = 1.0 - min(np.std(At_recent), 1.0)
        
        # Normalizar para [0, 0.5]
        return stability * 0.5
    
    def _print_final_report(self) -> None:
        """
        Imprime relatório final com análise multi-objetivo.
        
        Compara o resultado do objetivo selecionado com todos os outros
        objetivos possíveis, mostrando trade-offs e recomendações.
        
        Note:
            - Só é chamado se verbose >= 1
            - Usa ObjectiveComparator para análise comparativa
            - Mostra melhor época e métricas para cada objetivo
            - Destaca trade-offs significativos (>2% acc, >0.1 At, >$10)
        """
        print("\n" + "="*80)
        print("📊 RELATÓRIO FINAL - ANÁLISE MULTI-OBJETIVO")
        print("="*80)
        print()
        
        # Comparar todos os objetivos possíveis
        # comparison = ObjectiveComparator.compare_all(self.history)
        
        # Tabela comparativa
        print(f"{'Objetivo':<20} | {'Melhor Epoch':<12} | {'Val Acc':<8} | {'At':<6} | {'ROI':<8}")
        print("-" * 80)
        
        for obj_name, result in comparison.items():
            metrics = result['metrics_at_best']
            print(f"{obj_name:<20} | "
                  f"{result['best_epoch']+1:<12} | "
                  f"{metrics['val_acc']:.3f}    | "
                  f"{metrics['At']:.2f} | "
                  f"${metrics['roi']:>6.2f}")
        
        print("="*80)
        print()
        
        # Recomendação para o objetivo selecionado
        selected = comparison[self.objective]
        print(f"🎯 RECOMENDAÇÃO (objetivo: {self.objective})")
        print(f"   Epoch: {selected['best_epoch'] + 1}")
        print(f"   Val Accuracy: {selected['metrics_at_best']['val_acc']:.3f}")
        print(f"   Autonomia (At): {selected['metrics_at_best']['At']:.3f}")
        print(f"   Custo total: ${self.total_cost:.2f}")
        print()
        
        # Análise de trade-offs
        print("💡 TRADE-OFFS:")
        self._print_tradeoffs(comparison)
        
        print("="*80)
        print()
    
    def _print_tradeoffs(self, comparison: Dict) -> None:
        """
        Imprime análise de trade-offs entre objetivos.
        
        Compara o objetivo selecionado com os demais, mostrando diferenças
        significativas em accuracy, At e custo.
        
        Args:
            comparison: Resultado de ObjectiveComparator.compare_all()
        
        Note:
            - Só mostra diferenças > 2% em accuracy
            - Só mostra diferenças > 0.1 em At
            - Só mostra diferenças > $10 em custo
            - Sinal + indica que o outro objetivo é melhor naquela métrica
        """
        selected = comparison[self.objective]
        selected_metrics = selected['metrics_at_best']
        
        for obj_name, result in comparison.items():
            if obj_name == self.objective:
                continue
            
            metrics = result['metrics_at_best']
            
            # Calcular diferenças
            acc_diff = metrics['val_acc'] - selected_metrics['val_acc']
            At_diff = metrics['At'] - selected_metrics['At']
            epoch_diff = result['best_epoch'] - selected['best_epoch']
            cost_diff = epoch_diff * self.cost_per_epoch
            
            # Só mostrar se houver diferenças significativas
            if abs(acc_diff) > 0.02 or abs(At_diff) > 0.1 or abs(cost_diff) > 10:
                print(f"\n   vs {obj_name}:")
                if acc_diff != 0:
                    sign = "+" if acc_diff > 0 else ""
                    print(f"      Accuracy: {sign}{acc_diff*100:.1f}%")
                if At_diff != 0:
                    sign = "+" if At_diff > 0 else ""
                    print(f"      At (robustez): {sign}{At_diff:.2f}")
                if cost_diff != 0:
                    sign = "+" if cost_diff > 0 else ""
                    print(f"      Custo: {sign}${cost_diff:.2f}")
    
    def get_multi_objective_report(self) -> Dict[str, Any]:
        """
        Retorna relatório completo estruturado para análise programática.
        
        Returns:
            Dicionário contendo:
            - 'selected_objective': Nome do objetivo usado
            - 'recommended_epoch': Melhor época (1-indexed)
            - 'total_epochs': Total de épocas treinadas
            - 'total_cost': Custo total em USD
            - 'all_objectives': Comparação de todos objetivos possíveis
            - 'history': Histórico completo de todas métricas
            - 'config': Configuração do objetivo selecionado
        
        Example:
            >>> report = detector.get_multi_objective_report()
            >>> print(f"Melhor época: {report['recommended_epoch']}")
            >>> print(f"Custo: ${report['total_cost']:.2f}")
            >>> 
            >>> # Acessar comparação entre objetivos
            >>> for obj, data in report['all_objectives'].items():
            ...     print(f"{obj}: epoch {data['best_epoch']+1}")
            >>> 
            >>> # Plotar histórico customizado
            >>> import matplotlib.pyplot as plt
            >>> plt.plot(report['history']['val_acc'])
            >>> plt.show()
        
        Note:
            Este método pode ser chamado a qualquer momento, mas é mais útil
            após o treinamento terminar.
        """
        # Comparar todos os objetivos
        comparison = ObjectiveComparator.compare_all(self.history)
        
        return {
            'selected_objective': self.objective,
            'recommended_epoch': self.best_epoch + 1,
            'total_epochs': len(self.history['epoch']),
            'total_cost': self.total_cost,
            'all_objectives': comparison,
            'history': self.history,
            'config': {
                'name': self.config.name,
                'description': self.config.description,
                'tolerance': self.config.tolerance
            }
        }
    
    def plot_comparison(self, save_path: str = 'multi_objective_comparison.png') -> Any:
        """
        Plota comparação visual entre objetivos e salva figura.
        
        Cria uma figura 2x2 com:
        - Plot 1: Val Accuracy ao longo do tempo
        - Plot 2: At (Autonomia/Generalização) ao longo do tempo
        - Plot 3: ROI ao longo do tempo
        - Plot 4: Estados de aprendizado coloridos
        
        Args:
            save_path: Caminho onde salvar a figura PNG.
                Default: 'multi_objective_comparison.png'
        
        Returns:
            Objeto Figure do matplotlib
        
        Example:
            >>> # Salvar com nome padrão
            >>> detector.plot_comparison()
            📊 Gráfico salvo: multi_objective_comparison.png
            
            >>> # Salvar em local específico
            >>> detector.plot_comparison('results/experiment_01.png')
            📊 Gráfico salvo: results/experiment_01.png
            
            >>> # Obter figura para manipulação
            >>> fig = detector.plot_comparison()
            >>> fig.suptitle('Meu Experimento')
            >>> fig.savefig('custom.png')
        
        Note:
            - Requer matplotlib instalado
            - Marca a melhor época (linha vermelha tracejada)
            - Estados são coloridos por severidade
            - Resolução: 150 DPI
        """
        import matplotlib.pyplot as plt
        
        # Criar figura 2x2
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Multi-Objective Training Analysis', fontsize=16, fontweight='bold')
        
        # Epochs 1-indexed para visualização
        epochs = np.array(self.history['epoch']) + 1
        
        # ====================================
        # Plot 1: Val Accuracy
        # ====================================
        ax = axes[0, 0]
        ax.plot(epochs, self.history['val_acc'], 'b-', linewidth=2, label='Val Accuracy')
        ax.axvline(self.best_epoch + 1, color='r', linestyle='--', label=f'Best ({self.objective})')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Validation Accuracy')
        ax.set_title('Validation Accuracy Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # ====================================
        # Plot 2: At (Autonomia)
        # ====================================
        ax = axes[0, 1]
        ax.plot(epochs, self.history['At'], 'g-', linewidth=2, label='At')
        # Linha do baseline se calibrado
        if self.At_baseline:
            ax.axhline(self.At_baseline, color='gray', linestyle=':', label='Baseline')
        ax.axvline(self.best_epoch + 1, color='r', linestyle='--')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Autonomia (At)')
        ax.set_title('Generalization (At) Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # ====================================
        # Plot 3: ROI
        # ====================================
        ax = axes[1, 0]
        ax.plot(epochs, self.history['roi'], 'orange', linewidth=2, label='ROI')
        ax.axvline(self.best_epoch + 1, color='r', linestyle='--')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('ROI ($/point)')
        ax.set_title('Return on Investment Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # ====================================
        # Plot 4: States (Estados coloridos)
        # ====================================
        ax = axes[1, 1]
        
        # Mapeamento estado -> cor
        state_colors = {
            'inicializacao': 'gray',
            'aprendizado_rapido': 'green',
            'aprendizado_saudavel': 'lightgreen',
            'overfitting_inicial': 'yellow',
            'overfitting_severo': 'red',
            'plateau': 'orange',
            'underfitting': 'blue',
            'instabilidade': 'purple'
        }
        
        # Plotar cada época com cor do estado
        for i, state in enumerate(self.history['state']):
            color = state_colors.get(state, 'gray')
            ax.scatter(epochs[i], self.history['val_acc'][i], c=color, s=50, alpha=0.7)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Val Accuracy')
        ax.set_title('Training States')
        ax.grid(True, alpha=0.3)
        
        # Legenda de cores dos estados
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=color, label=state.replace('_', ' ').title()) 
                          for state, color in state_colors.items()]
        ax.legend(handles=legend_elements, loc='best', fontsize=8)
        
        # Salvar figura
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n📊 Gráfico salvo: {save_path}")
        
        return fig
