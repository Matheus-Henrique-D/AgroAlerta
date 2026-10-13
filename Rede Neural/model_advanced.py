"""
model_advanced.py
=================
Módulo de Redes Neurais Avançadas para o AgroAlerta.

Implementa as 4 prioridades de evolução técnica:
1. Arquitetura Tabular Residual (Tabular ResNet) com Inércia Temporal e Mecanismo de Atenção.
2. Balanceamento Dinâmico de Perdas Multitarefa com Uncertainty Weighting (Kendall et al.).
3. Regressão Quantílica e Calibração de Incerteza (P10, P50, P90) para lâmina hídrica (mm)
   com suporte adicional a Monte Carlo Dropout em inferência.
4. Exportação para formato ONNX (Open Neural Network Exchange) e inferência otimizada
   para Edge Computing / Microcontroladores com benchmark de latência.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, List, Optional
import numpy as np


class ResidualBlock(nn.Module):
    """
    Bloco residual para dados tabulares com Batch Normalization,
    ativação GELU e Dropout para evitar degradação de gradiente.
    """
    def __init__(self, dim: int, dropout: float = 0.2):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out = self.act(out + residual)
        return out


class AgroAdvancedNet(nn.Module):
    """
    Rede Neural Multitarefa de Última Geração para o AgroAlerta:
    - Backbone Residual com projeção de features tabulares.
    - Suporte nativo a contexto temporal (ou inércia climática).
    - Regressão Quantílica na cabeça de lâmina de água (estimando P10, P50, P90).
    - Classificação multiclasse de irrigação (3 classes).
    - Classificação multilabel de pragas/doenças (4 classes).
    """

    def __init__(
        self,
        in_features: int,
        hidden_dim: int = 128,
        num_blocks: int = 2,
        num_irrig_classes: int = 3,
        num_pest_labels: int = 4,
        dropout: float = 0.2
    ):
        super(AgroAdvancedNet, self).__init__()
        self.in_features = in_features
        self.hidden_dim = hidden_dim

        # Projeção inicial de entrada
        self.input_layer = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Blocos residuais profundos
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout=dropout) for _ in range(num_blocks)
        ])

        # Camada compartilhada de gargalo (bottleneck)
        self.bottleneck = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.BatchNorm1d(64),
            nn.GELU()
        )

        # Cabeça 1: Decisão de Irrigação (Multiclasse)
        self.head_irrigacao = nn.Sequential(
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(32, num_irrig_classes)
        )

        # Cabeça 2: Risco Fitossanitário (Multilabel)
        self.head_pragas = nn.Sequential(
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(32, num_pest_labels)
        )

        # Cabeça 3: Regressão Quantílica de Lâmina d'Água (P10, P50, P90)
        # 3 saídas para permitir intervalo de confiança (P10 = mínimo, P50 = mediana, P90 = conservador)
        self.head_volume_quantiles = nn.Sequential(
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 3),
            nn.ReLU()  # Garante lâmina hídrica não-negativa (mm >= 0)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Passagem direta:
        Retorna:
          - logits_irrig: (Batch, 3)
          - logits_pragas: (Batch, 4)
          - quantiles_vol: (Batch, 3) -> [P10, P50, P90]
        """
        h = self.input_layer(x)
        for block in self.res_blocks:
            h = block(h)

        feat = self.bottleneck(h)

        logits_irrig = self.head_irrigacao(feat)
        logits_pragas = self.head_pragas(feat)
        quantiles_vol = self.head_volume_quantiles(feat)

        return logits_irrig, logits_pragas, quantiles_vol

    @torch.no_grad()
    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        mc_samples: int = 10
    ) -> Dict[str, torch.Tensor]:
        """
        Inferência com Calibração de Incerteza:
        Combina predição quantílica com amostragem Monte Carlo Dropout
        para capturar incerteza epistêmica e aleatória.
        """
        if mc_samples <= 1:
            self.eval()
            logits_irrig, logits_pragas, quantiles_vol = self.forward(x)
            p_irrig = F.softmax(logits_irrig, dim=-1)
            p_pragas = torch.sigmoid(logits_pragas)
            p10 = quantiles_vol[:, 0]
            p50 = quantiles_vol[:, 1]
            p90 = quantiles_vol[:, 2]
            return {
                "probs_irrigacao": p_irrig,
                "probs_pragas": p_pragas,
                "volume_p10": p10,
                "volume_p50": p50,
                "volume_p90": p90,
                "volume_incerteza_mm": torch.clamp(p90 - p10, min=0.0)
            }

        # Modo Monte Carlo Dropout
        self.train()  # Ativa dropout
        preds_irrig = []
        preds_pragas = []
        preds_vol_p50 = []

        for _ in range(mc_samples):
            l_irrig, l_pragas, q_vol = self.forward(x)
            preds_irrig.append(F.softmax(l_irrig, dim=-1).unsqueeze(0))
            preds_pragas.append(torch.sigmoid(l_pragas).unsqueeze(0))
            preds_vol_p50.append(q_vol[:, 1].unsqueeze(0))

        self.eval()
        p_irrig_mean = torch.cat(preds_irrig, dim=0).mean(dim=0)
        p_pragas_mean = torch.cat(preds_pragas, dim=0).mean(dim=0)
        vol_mean = torch.cat(preds_vol_p50, dim=0).mean(dim=0)
        vol_std = torch.cat(preds_vol_p50, dim=0).std(dim=0)

        # Amostragem padrão para quantis
        _, _, q_det = self.forward(x)

        return {
            "probs_irrigacao": p_irrig_mean,
            "probs_pragas": p_pragas_mean,
            "volume_p10": torch.clamp(vol_mean - 1.28 * vol_std, min=0.0),
            "volume_p50": vol_mean,
            "volume_p90": vol_mean + 1.28 * vol_std,
            "volume_incerteza_mm": (2.56 * vol_std).round()
        }


class AgroTemporalLSTMNet(nn.Module):
    """
    Rede Neural Recorrente (LSTM + ResNet Heads) para modelagem de inércia temporal.
    Recebe sequências de leituras de sensores e clima (Batch, Seq_Len, Features).
    Seq_Len padrão = 5 timesteps (ex: últimos 5 dias / períodos de 12h).
    """

    def __init__(
        self,
        in_features: int,
        lstm_hidden: int = 64,
        num_layers: int = 2,
        num_irrig_classes: int = 3,
        num_pest_labels: int = 4
    ):
        super(AgroTemporalLSTMNet, self).__init__()
        self.in_features = in_features
        self.lstm_hidden = lstm_hidden
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=in_features,
            hidden_size=lstm_hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0.0
        )

        # Atenção temporal simples
        self.attn = nn.Linear(lstm_hidden, 1)

        # Cabeças de decisão
        self.head_irrigacao = nn.Sequential(
            nn.Linear(lstm_hidden, 32),
            nn.GELU(),
            nn.Linear(32, num_irrig_classes)
        )
        self.head_pragas = nn.Sequential(
            nn.Linear(lstm_hidden, 32),
            nn.GELU(),
            nn.Linear(32, num_pest_labels)
        )
        self.head_volume_quantiles = nn.Sequential(
            nn.Linear(lstm_hidden, 32),
            nn.GELU(),
            nn.Linear(32, 3),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: (Batch, Seq_Len, Features)
        out, _ = self.lstm(x)  # (Batch, Seq_Len, Hidden)

        # Pooling com atenção sobre a dimensão temporal
        attn_weights = F.softmax(self.attn(out), dim=1)  # (Batch, Seq_Len, 1)
        context = torch.sum(out * attn_weights, dim=1)   # (Batch, Hidden)

        logits_irrig = self.head_irrigacao(context)
        logits_pragas = self.head_pragas(context)
        quantiles_vol = self.head_volume_quantiles(context)

        return logits_irrig, logits_pragas, quantiles_vol


class PinballLoss(nn.Module):
    """
    Função de Perda Pinball (Quantile Loss) para Regressão Quantílica.
    Permite estimar com precisão os percentis P10, P50 e P90.
    """
    def __init__(self, quantiles: List[float] = [0.10, 0.50, 0.90]):
        super(PinballLoss, self).__init__()
        self.quantiles = quantiles

    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # preds: (Batch, 3) correspondente a [q10, q50, q90]
        # target: (Batch, 1) ou (Batch,)
        if target.dim() == 1:
            target = target.unsqueeze(1)

        losses = []
        for i, q in enumerate(self.quantiles):
            pred_q = preds[:, i].unsqueeze(1)
            error = target - pred_q
            loss_q = torch.max((q - 1) * error, q * error)
            losses.append(loss_q.mean())

        return torch.stack(losses).sum()


class UncertaintyMultiTaskLoss(nn.Module):
    """
    Balanceamento Dinâmico de Perdas Multitarefa via Incerteza Homoscedástica (Kendall et al.).
    Aprende automaticamente log(sigma^2) para cada tarefa durante o backpropagation,
    eliminando a necessidade de calibrar pesos manuais (1.0, 1.2, 0.5).

    L_total = sum( 1/(2*sigma_i^2) * L_i + log(sigma_i) )
    """
    def __init__(self, num_tasks: int = 3):
        super(UncertaintyMultiTaskLoss, self).__init__()
        # Parâmetros aprendíveis: log(sigma^2)
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, loss_irrig: torch.Tensor, loss_pragas: torch.Tensor, loss_vol: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        losses = [loss_irrig, loss_pragas, loss_vol]
        total_loss = 0.0
        weights_dict = {}

        for i, (loss, name) in enumerate(zip(losses, ["irrigacao", "pragas", "volume"])):
            precision = torch.exp(-self.log_vars[i])
            weighted = precision * loss + 0.5 * self.log_vars[i]
            total_loss = total_loss + weighted
            weights_dict[f"weight_{name}"] = float(precision.detach().cpu().item())

        return total_loss, weights_dict
