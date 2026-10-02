"""
retrain_500k.py
===============
Script de re-treinamento (fine-tuning) da AgroMultitaskNet.
- Simula 500.000 novas amostras sobre o CSV base existente
- Carrega os pesos ja treinados em best_agro_multitask.pth
- Executa 10 epocas de refinamento com AdamW + ReduceLROnPlateau
- Gera plotagens completas salvas em PNG
- Imprime tabela de metricas finais com comparativo vs baseline
"""

import os, sys, time
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, mean_absolute_error,
    confusion_matrix, classification_report
)
try:
    from sklearn.metrics import root_mean_squared_error
    def rmse_fn(a, b): return root_mean_squared_error(a, b)
except ImportError:
    def rmse_fn(a, b): return float(np.sqrt(np.mean((np.array(a)-np.array(b))**2)))

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path: sys.path.insert(0, BASE_DIR)

from data_engine import AgroDataEngine, AgroMultiTaskDataset
from model_multitask import AgroMultitaskNet

CSV_PATH    = os.path.join(BASE_DIR, "dados_solo_2casas.csv")
MODEL_PATH  = os.path.join(BASE_DIR, "best_agro_multitask.pth")
SCALER_PATH = os.path.join(BASE_DIR, "agro_scaler_meta.pkl")
OUT_DIR     = BASE_DIR

N_SAMPLES    = 500_000
EPOCHS       = 10
BATCH_SIZE   = 512
LR           = 3e-4
WEIGHT_DECAY = 1e-4
RANDOM_SEED  = 99

PRAGAS_COLS   = ["target_fungos","target_insetos","target_acaros","target_estresse"]
PRAGAS_LABELS = ["Fungos/Doencas","Insetos/Lagartas","Acaros/Tripes","Estresse Hidrico"]
IRRIG_LABELS  = ["0-Nao Irrigar","1-Leve/Moderada","2-Abundante/Turno"]

BASELINE = dict(acc_irrig=0.9722, f1_irrig=0.9727, subset_acc_pragas=0.9236,
                f1_pragas_micro=0.9692, mae_vol=0.228, rmse_vol=0.488)

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}\n  Re-treinamento: 500k amostras / 10 epocas  |  {device}\n{'='*60}\n")

    engine = AgroDataEngine(scaler_path=SCALER_PATH)
    df = engine.load_and_augment_dataset(csv_path=CSV_PATH, n_samples=N_SAMPLES, random_seed=RANDOM_SEED)

    train_df, temp_df = train_test_split(df, test_size=0.20, random_state=42, stratify=df["target_irrigacao"])
    val_df,   test_df = train_test_split(temp_df, test_size=0.50, random_state=42, stratify=temp_df["target_irrigacao"])
    print(f"Treino: {len(train_df):,} | Val: {len(val_df):,} | Teste: {len(test_df):,}\n")

    X_train = engine.prepare_feature_matrix(train_df, is_train=True)
    X_val   = engine.prepare_feature_matrix(val_df,   is_train=False)
    X_test  = engine.prepare_feature_matrix(test_df,  is_train=False)

    mk = lambda df2: AgroMultiTaskDataset(
        engine.prepare_feature_matrix(df2, is_train=False) if df2 is not train_df else X_train,
        df2["target_irrigacao"].values, df2[PRAGAS_COLS].values, df2["target_volume_mm"].values)

    ds_train = AgroMultiTaskDataset(X_train, train_df["target_irrigacao"].values, train_df[PRAGAS_COLS].values, train_df["target_volume_mm"].values)
    ds_val   = AgroMultiTaskDataset(X_val,   val_df["target_irrigacao"].values,   val_df[PRAGAS_COLS].values,   val_df["target_volume_mm"].values)
    ds_test  = AgroMultiTaskDataset(X_test,  test_df["target_irrigacao"].values,  test_df[PRAGAS_COLS].values,  test_df["target_volume_mm"].values)

    kw = dict(num_workers=0)
    loader_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True,  drop_last=True, **kw)
    loader_val   = DataLoader(ds_val,   batch_size=BATCH_SIZE, shuffle=False, **kw)
    loader_test  = DataLoader(ds_test,  batch_size=BATCH_SIZE, shuffle=False, **kw)

    in_features = X_train.shape[1]
    model = AgroMultitaskNet(in_features=in_features, num_irrig_classes=3, num_pest_labels=4).to(device)
    if os.path.exists(MODEL_PATH):
        try:
            ckpt = torch.load(MODEL_PATH, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            print(f"Pesos carregados (epoca base: {ckpt.get('epoch','?')})\n")
        except RuntimeError as e:
            print(f"AVISO – nao foi possivel carregar pesos: {e}\nTreinando do zero.\n")

    c_ir = nn.CrossEntropyLoss(); c_pg = nn.BCEWithLogitsLoss(); c_vo = nn.SmoothL1Loss()
    opt  = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sch  = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=2)

    hist = {k:[] for k in ["train_total","val_total","train_irrig","val_irrig",
                            "train_pragas","val_pragas","train_vol","val_vol","lr"]}
    best_val = float("inf")
    best_path = os.path.join(OUT_DIR, "best_agro_multitask_500k.pth")
    t0 = time.time()

    for ep in range(1, EPOCHS+1):
        model.train()
        tl=ti=tp=tv_=nb = 0.0
        for bx,bi,bp,bv in loader_train:
            bx,bi,bp,bv = bx.to(device),bi.to(device),bp.to(device),bv.to(device)
            opt.zero_grad()
            pi,pp,pv = model(bx)
            li=c_ir(pi,bi); lp=c_pg(pp,bp); lv=c_vo(pv,bv)
            loss = li + 1.2*lp + 0.5*lv
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            tl+=loss.item(); ti+=li.item(); tp+=lp.item(); tv_+=lv.item(); nb+=1

        model.eval()
        vl=vi=vp=vv=nvb = 0.0
        with torch.no_grad():
            for bx,bi,bp,bv in loader_val:
                bx,bi,bp,bv = bx.to(device),bi.to(device),bp.to(device),bv.to(device)
                pi,pp,pv = model(bx)
                li=c_ir(pi,bi); lp=c_pg(pp,bp); lv=c_vo(pv,bv)
                v=li+1.2*lp+0.5*lv
                vl+=v.item(); vi+=li.item(); vp+=lp.item(); vv+=lv.item(); nvb+=1

        tr=tl/nb; vr=vl/nvb; cur_lr=opt.param_groups[0]["lr"]
        for k,v2 in zip(["train_total","val_total","train_irrig","val_irrig",
                         "train_pragas","val_pragas","train_vol","val_vol","lr"],
                        [tr,vr,ti/nb,vi/nvb,tp/nb,vp/nvb,tv_/nb,vv/nvb,cur_lr]):
            hist[k].append(v2)
        sch.step(vr)
        salvo=""
        if vr < best_val:
            best_val = vr
            torch.save({"model_state_dict":model.state_dict(),"in_features":in_features,
                        "epoch":ep,"best_val_loss":best_val}, best_path)
            salvo=" [* SALVO]"
        print(f"Ep[{ep:02d}/{EPOCHS}] Train:{tr:.4f}(Ir:{ti/nb:.3f} Pg:{tp/nb:.3f} Vo:{tv_/nb:.3f}) "
              f"Val:{vr:.4f}(Ir:{vi/nvb:.3f} Pg:{vp/nvb:.3f} Vo:{vv/nvb:.3f}) LR:{cur_lr:.2e}{salvo}", flush=True)

    print(f"\nTreinamento: {time.time()-t0:.1f}s\n")

    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"]); model.eval()

    pir=[]; tir=[]; ppg=[]; tpg=[]; pvo=[]; tvo=[]
    with torch.no_grad():
        for bx,bi,bp,bv in loader_test:
            bx=bx.to(device)
            oi,op,ov = model(bx)
            pir.extend(torch.argmax(oi,dim=-1).cpu().numpy()); tir.extend(bi.numpy())
            ppg.extend((torch.sigmoid(op)>=0.5).cpu().numpy().astype(int)); tpg.extend(bp.numpy())
            pvo.extend(ov.cpu().numpy().flatten()); tvo.extend(bv.numpy().flatten())

    pir=np.array(pir); tir=np.array(tir)
    ppg=np.array(ppg); tpg=np.array(tpg)
    pvo=np.array(pvo); tvo=np.array(tvo)

    acc_irrig   = accuracy_score(tir, pir)
    f1_irrig    = f1_score(tir, pir, average="macro")
    f1pg_micro  = f1_score(tpg, ppg, average="micro",  zero_division=0)
    f1pg_macro  = f1_score(tpg, ppg, average="macro",  zero_division=0)
    subset_acc  = accuracy_score(tpg, ppg)
    mae_vol     = mean_absolute_error(tvo, pvo)
    rmse_vol    = rmse_fn(tvo, pvo)
    pla  = [accuracy_score(tpg[:,i], ppg[:,i]) for i in range(4)]
    plf  = [f1_score(tpg[:,i], ppg[:,i], zero_division=0) for i in range(4)]

    _plot_loss_curves(hist, OUT_DIR)
    _plot_confusion_matrix(tir, pir, OUT_DIR)
    _plot_pragas_bars(pla, plf, OUT_DIR)
    _plot_volume_scatter(tvo, pvo, mae_vol, rmse_vol, OUT_DIR)
    _plot_metrics_comparison(acc_irrig, f1_irrig, subset_acc, f1pg_micro, mae_vol, rmse_vol, OUT_DIR)
    print("Plotagens salvas em:", OUT_DIR, "\n")

    print(f"{'='*60}\n  METRICAS FINAIS – TESTE CEGO ({len(tir):,} amostras)\n{'='*60}")
    print(f"\n[TAREFA 1 – IRRIGACAO]")
    print(f"  Acuracia  : {acc_irrig*100:.2f}%  (baseline {BASELINE['acc_irrig']*100:.2f}%  D={( acc_irrig-BASELINE['acc_irrig'])*100:+.2f}pp)")
    print(f"  Macro F1  : {f1_irrig:.4f}  (baseline {BASELINE['f1_irrig']:.4f}  D={f1_irrig-BASELINE['f1_irrig']:+.4f})")
    print(classification_report(tir, pir, target_names=IRRIG_LABELS, digits=3))
    print(f"[TAREFA 2 – FITOSSANIDADE]")
    print(f"  Subset Acc: {subset_acc*100:.2f}%  (baseline {BASELINE['subset_acc_pragas']*100:.2f}%  D={(subset_acc-BASELINE['subset_acc_pragas'])*100:+.2f}pp)")
    print(f"  Micro F1  : {f1pg_micro:.4f}  (baseline {BASELINE['f1_pragas_micro']:.4f}  D={f1pg_micro-BASELINE['f1_pragas_micro']:+.4f})")
    print(f"  Macro F1  : {f1pg_macro:.4f}")
    for i,lb in enumerate(PRAGAS_LABELS):
        print(f"   - {lb:<25}: Acc={pla[i]*100:.2f}%  F1={plf[i]:.4f}")
    print(f"\n[TAREFA 3 – LAMINA AGUA mm]")
    print(f"  MAE  : {mae_vol:.3f} mm  (baseline {BASELINE['mae_vol']:.3f}  D={mae_vol-BASELINE['mae_vol']:+.3f})")
    print(f"  RMSE : {rmse_vol:.3f} mm  (baseline {BASELINE['rmse_vol']:.3f}  D={rmse_vol-BASELINE['rmse_vol']:+.3f})")
    print(f"{'='*60}")

    _print_melhorias()


def _plot_loss_curves(h, od):
    eps = range(1, len(h["train_total"])+1)
    fig,axes = plt.subplots(2,2,figsize=(14,9))
    fig.suptitle("Curvas de Perda – Re-treinamento 500k/10ep", fontsize=13, fontweight="bold")
    pairs = [("train_total","val_total","Loss Total"),("train_irrig","val_irrig","Loss Irrigacao"),
             ("train_pragas","val_pragas","Loss Pragas"),("train_vol","val_vol","Loss Volume mm")]
    ct=["#1f77b4","#2ca02c","#ff7f0e","#9467bd"]; cv=["#aec7e8","#98df8a","#ffbb78","#c5b0d5"]
    for ax,(tk,vk,ttl),c1,c2 in zip(axes.flat,pairs,ct,cv):
        ax.plot(eps,h[tk],color=c1,marker="o",label="Treino",linewidth=2)
        ax.plot(eps,h[vk],color=c2,marker="s",ls="--",label="Validacao",linewidth=2)
        ax.set_title(ttl,fontsize=10); ax.set_xlabel("Epoca"); ax.set_ylabel("Loss")
        ax.legend(fontsize=8); ax.grid(True,alpha=0.35); ax.set_xticks(list(eps))
    plt.tight_layout()
    p=os.path.join(od,"plot_01_loss_curves.png"); plt.savefig(p,dpi=150,bbox_inches="tight"); plt.close()
    print(f"  Salvo: {p}")

def _plot_confusion_matrix(ti,pi,od):
    cm=confusion_matrix(ti,pi); cmp=cm.astype(float)/cm.sum(axis=1,keepdims=True)*100
    fig,axes=plt.subplots(1,2,figsize=(14,5))
    fig.suptitle("Matriz de Confusao – Irrigacao",fontsize=13,fontweight="bold")
    lbs=["Nao Irrigar","Leve/Moderada","Abundante"]
    for ax,data,ttl,fs in zip(axes,[cm,cmp],["Contagem Absoluta","Percentual por Classe"],
                               ["{:,.0f}","{:.1f}%"]):
        im=ax.imshow(data,cmap="Blues")
        ax.set_xticks(range(3)); ax.set_yticks(range(3))
        ax.set_xticklabels(lbs,rotation=15,ha="right",fontsize=9)
        ax.set_yticklabels(lbs,fontsize=9)
        ax.set_xlabel("Predito"); ax.set_ylabel("Real"); ax.set_title(ttl,fontsize=10)
        plt.colorbar(im,ax=ax)
        th=data.max()/1.8
        for i in range(3):
            for j in range(3):
                v=data[i,j]
                ax.text(j,i,fs.format(v),ha="center",va="center",
                        color="white" if v>th else "black",fontsize=9,fontweight="bold")
    plt.tight_layout()
    p=os.path.join(od,"plot_02_confusion_matrix.png"); plt.savefig(p,dpi=150,bbox_inches="tight"); plt.close()
    print(f"  Salvo: {p}")

def _plot_pragas_bars(pla,plf,od):
    lbs=["Fungos/\nDoencas","Insetos/\nLagartas","Acaros/\nTripes","Estresse\nHidrico"]
    x=np.arange(4); w=0.35
    fig,ax=plt.subplots(figsize=(10,6))
    ba=ax.bar(x-w/2,[v*100 for v in pla],w,label="Acuracia (%)",color="#2ca02c",alpha=0.85,edgecolor="white")
    bf=ax.bar(x+w/2,[v*100 for v in plf],w,label="F1-Score (x100)",color="#1f77b4",alpha=0.85,edgecolor="white")
    ax.set_ylim(70,103); ax.set_ylabel("Percentual (%)"); ax.set_xticks(x); ax.set_xticklabels(lbs,fontsize=10)
    ax.set_title("Desempenho por Label – Fitossanidade",fontsize=13,fontweight="bold")
    ax.legend(fontsize=10); ax.grid(axis="y",alpha=0.35)
    for b in list(ba)+list(bf):
        ax.text(b.get_x()+b.get_width()/2,b.get_height()+0.3,f"{b.get_height():.1f}",
                ha="center",va="bottom",fontsize=8,fontweight="bold")
    plt.tight_layout()
    p=os.path.join(od,"plot_03_fitossanidade_labels.png"); plt.savefig(p,dpi=150,bbox_inches="tight"); plt.close()
    print(f"  Salvo: {p}")

def _plot_volume_scatter(tv,pv,mae,rmse,od):
    idx=np.random.default_rng(0).choice(len(tv),size=min(8000,len(tv)),replace=False)
    ts,ps=tv[idx],pv[idx]
    fig,axes=plt.subplots(1,2,figsize=(14,5))
    fig.suptitle("Lamina de Agua Predita vs Real (mm)",fontsize=13,fontweight="bold")
    ax=axes[0]
    ax.scatter(ts,ps,alpha=0.25,s=8,color="#ff7f0e",rasterized=True)
    lims=[min(ts.min(),ps.min())-.5,max(ts.max(),ps.max())+.5]
    ax.plot(lims,lims,"k--",lw=1.5,label="Pred. Perfeita")
    ax.set_xlabel("Real (mm)"); ax.set_ylabel("Predito (mm)")
    ax.set_title(f"Scatter | MAE={mae:.3f} mm  RMSE={rmse:.3f} mm")
    ax.legend(fontsize=9); ax.grid(True,alpha=0.3)
    ax2=axes[1]; res=ps-ts
    ax2.hist(res,bins=80,color="#9467bd",alpha=0.8,edgecolor="white")
    ax2.axvline(0,color="black",lw=1.5,ls="--")
    ax2.axvline(res.mean(),color="red",lw=1.2,ls=":",label=f"Media={res.mean():.3f}")
    ax2.set_xlabel("Residuo (mm)"); ax2.set_ylabel("Frequencia")
    ax2.set_title("Distribuicao dos Residuos"); ax2.legend(fontsize=9); ax2.grid(True,alpha=0.3)
    plt.tight_layout()
    p=os.path.join(od,"plot_04_volume_scatter.png"); plt.savefig(p,dpi=150,bbox_inches="tight"); plt.close()
    print(f"  Salvo: {p}")

def _plot_metrics_comparison(acc,f1,sa,f1m,mae,rmse,od):
    cats=["Acuracia\nIrrigacao","F1 Irrigacao","Subset Acc\nPragas","Micro F1\nPragas"]
    bv=[BASELINE["acc_irrig"]*100,BASELINE["f1_irrig"]*100,
        BASELINE["subset_acc_pragas"]*100,BASELINE["f1_pragas_micro"]*100]
    nv=[acc*100,f1*100,sa*100,f1m*100]
    x=np.arange(4); w=0.35
    fig,axes=plt.subplots(1,2,figsize=(16,6))
    fig.suptitle("Comparativo: Baseline vs Re-treinamento 500k",fontsize=13,fontweight="bold")
    ax=axes[0]
    b1=ax.bar(x-w/2,bv,w,label="Baseline",color="#aec7e8",edgecolor="navy",lw=0.8)
    b2=ax.bar(x+w/2,nv,w,label="500k Re-treinamento",color="#1f77b4",edgecolor="navy",lw=0.8)
    ax.set_ylim(85,102); ax.set_ylabel("Valor (%)"); ax.set_title("Metricas de Classificacao")
    ax.set_xticks(x); ax.set_xticklabels(cats,fontsize=9); ax.legend(fontsize=9); ax.grid(axis="y",alpha=0.35)
    for b in list(b1)+list(b2):
        ax.text(b.get_x()+b.get_width()/2,b.get_height()+0.1,f"{b.get_height():.2f}",
                ha="center",va="bottom",fontsize=7.5,fontweight="bold")
    ax2=axes[1]
    rc=["MAE (mm)","RMSE (mm)"]; rb=[BASELINE["mae_vol"],BASELINE["rmse_vol"]]; rn=[mae,rmse]
    x2=np.arange(2)
    b3=ax2.bar(x2-w/2,rb,w,label="Baseline",color="#ffbb78",edgecolor="#d62728",lw=0.8)
    b4=ax2.bar(x2+w/2,rn,w,label="500k",color="#ff7f0e",edgecolor="#d62728",lw=0.8)
    ax2.set_ylabel("Erro (mm)"); ax2.set_title("Erro de Regressao – Lamina de Agua")
    ax2.set_xticks(x2); ax2.set_xticklabels(rc,fontsize=10); ax2.legend(fontsize=9); ax2.grid(axis="y",alpha=0.35)
    for b in list(b3)+list(b4):
        ax2.text(b.get_x()+b.get_width()/2,b.get_height()+0.005,f"{b.get_height():.3f}",
                 ha="center",va="bottom",fontsize=9,fontweight="bold")
    plt.tight_layout()
    p=os.path.join(od,"plot_05_metrics_comparison.png"); plt.savefig(p,dpi=150,bbox_inches="tight"); plt.close()
    print(f"  Salvo: {p}")

def _print_melhorias():
    print("""
+------------------------------------------------------------------+
|           POSSIVEIS MELHORIAS PARA A REDE NEURAL                 |
+------------------------------------------------------------------+
|                                                                  |
|  [ARQUITETURA]                                                   |
|  1. Residual Connections (Skip Connections)                      |
|     Atalhos soma(x, h) entre blocos do backbone aceleram         |
|     convergencia e evitam vanishing gradient.                    |
|                                                                  |
|  2. Attention/Feature Gating (Squeeze-and-Excitation)            |
|     O modelo aprende a pesar quais das 32 features sao mais      |
|     relevantes por amostra (ex: ET0 em clima quente).            |
|                                                                  |
|  3. Output de Volume com Incerteza (Gaussian NLL Loss)           |
|     Prever media + desvio-padrao da lamina permite comunicar     |
|     incerteza ao agricultor (ex: "7.2 +/- 1.3 mm").             |
|                                                                  |
|  [DADOS]                                                         |
|  4. Features de NDVI / Imagem de Satelite                        |
|     Integrar indice de vegetacao Sentinel-2 detecta stress       |
|     hidrico antes do sensor fisico registrar queda.              |
|                                                                  |
|  5. Balanceamento de Classes (WeightedRandomSampler)             |
|     Se classe 2 (irrigacao abundante) for rara, oversampling     |
|     ou loss ponderada evita vies para classe 0.                  |
|                                                                  |
|  6. Mixup Data Augmentation                                      |
|     Interpolar pares de amostras durante treino reduz            |
|     overfitting em dados sinteticos.                             |
|                                                                  |
|  [OTIMIZACAO E TREINAMENTO]                                      |
|  7. Cosine Annealing Warm Restarts (SGDR)                        |
|     Permite escapes de minimos locais em runs longas.            |
|                                                                  |
|  8. Gradient Accumulation                                        |
|     Simula batches maiores sem aumentar memoria de GPU.          |
|                                                                  |
|  [INTERPRETABILIDADE]                                            |
|  9. SHAP / Permutation Importance                                |
|     Explicar ao agricultor qual feature determinou a decisao     |
|     (ex: "ET0 alta foi o principal fator hoje").                 |
|                                                                  |
|  10. Calibracao de Probabilidades (Temperature Scaling)          |
|      Garante que P=70% corresponda a 70% de acerto real.         |
|                                                                  |
|  [OPERACIONAL]                                                   |
|  11. Online Learning com Dados Reais do ESP32/Arduino            |
|      Fine-tuning diario com leituras reais corrige drift         |
|      de sensor e adapta ao microclima do talhao.                 |
|                                                                  |
+------------------------------------------------------------------+
""")

if __name__ == "__main__":
    train()
