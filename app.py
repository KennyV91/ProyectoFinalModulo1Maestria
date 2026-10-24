import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResults

st.set_page_config(page_title="Pronóstico de demanda", layout="wide")

# ----------------------------------------------------------------------------- archivos
CARPETA = os.path.dirname(os.path.abspath(__file__))
F_SERIE = os.path.join(CARPETA, "serie_modelo.csv")        # obligatorio
F_MODELO = os.path.join(CARPETA, "modelo_sarima.pkl")      # se crea solo si no existe
F_CONFIG = os.path.join(CARPETA, "config.json")            # obligatorio
F_SEGUIMIENTO = os.path.join(CARPETA, "seguimiento.csv")   # se crea al registrar meses
F_VENTAS = os.path.join(CARPETA, "ventas_limpio.csv")      # opcional (pestaña Datos)

for f in (F_SERIE, F_CONFIG):
    if not os.path.exists(f):
        st.error(f"Falta el archivo {os.path.basename(f)} en la carpeta de la app.")
        st.stop()

with open(F_CONFIG) as f:
    CFG = json.load(f)
CFG_TXT = json.dumps(CFG, sort_keys=True)
KG = CFG["kg_por_bolsa"]
MESES = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]


# ----------------------------------------------------------------------------- funciones
def leer_serie():
    s = pd.read_csv(F_SERIE, index_col=0, parse_dates=True).iloc[:, 0]
    s.name = "toneladas"
    return s.asfreq("MS")


def ajustar(serie):
    return SARIMAX(
        serie,
        order=tuple(CFG["order"]),
        seasonal_order=tuple(CFG["seasonal_order"]),
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False)


@st.cache_resource
def cargar_modelo(firma_serie):
    """Usa el modelo guardado si está al día con la serie; si no, lo reentrena y lo guarda."""
    if os.path.exists(F_MODELO) and os.path.getmtime(F_MODELO) >= firma_serie:
        try:
            return SARIMAXResults.load(F_MODELO)
        except Exception:
            pass
    res = ajustar(leer_serie())
    res.save(F_MODELO)
    return res


def pronosticar(res, h, nivel):
    fc = res.get_forecast(h)
    ic = fc.conf_int(alpha=1 - nivel / 100)
    return fc.predicted_mean, np.maximum(ic.iloc[:, 0], 0), ic.iloc[:, 1]


def f_prom12(tr, h):
    return np.repeat(tr.iloc[-12:].mean(), h)


def f_prom_est(tr, h, anios=6):
    ref = tr.iloc[-12 * anios:]
    bloque = np.arange(len(ref)) // 12
    desv = ref - ref.groupby(bloque).transform("mean")
    efecto = desv.groupby(desv.index.month).mean()
    idx_f = pd.date_range(tr.index[-1] + pd.offsets.MonthBegin(1), periods=h, freq="MS")
    return tr.iloc[-12:].mean() + efecto.reindex(idx_f.month).values


def metricas(real, pred, train=None, lo=None, hi=None):
    real = pd.Series(np.asarray(real))
    pred = pd.Series(np.asarray(pred))
    err = real - pred
    wape = 100 * err.abs().sum() / real.sum()
    out = {
        "MAE (t)": err.abs().mean(),
        "RMSE (t)": np.sqrt((err ** 2).mean()),
        "WAPE (%)": wape,
        "Precisión (%)": 100 - wape,
        "Sesgo (%)": 100 * err.sum() / real.sum(),
    }
    if train is not None and len(train) > 12:
        out["MASE"] = err.abs().mean() / train.diff(12).abs().mean()
    if lo is not None:
        out["Cobertura IC (%)"] = 100 * ((real >= np.asarray(lo)) & (real <= np.asarray(hi))).mean()
    return out


@st.cache_data(show_spinner="Evaluando el modelo con los últimos meses...")
def evaluar_holdout(firma, cfg_txt, h):
    serie = leer_serie()
    tr, te = serie.iloc[:-h], serie.iloc[-h:]
    res = ajustar(tr)
    fc = res.get_forecast(h)
    ic80, ic95 = fc.conf_int(alpha=0.20), fc.conf_int(alpha=0.05)
    return {
        "tr": tr, "te": te,
        "SARIMA": fc.predicted_mean.values,
        "Promedio 12 meses": f_prom12(tr, h),
        "Promedio 12m + estacional": f_prom_est(tr, h),
        "lo80": np.maximum(ic80.iloc[:, 0].values, 0), "hi80": ic80.iloc[:, 1].values,
        "lo95": np.maximum(ic95.iloc[:, 0].values, 0), "hi95": ic95.iloc[:, 1].values,
    }


@st.cache_data(show_spinner="Validación móvil en curso (puede tardar un minuto)...")
def validacion_movil(firma, cfg_txt, H=12):
    serie = leer_serie()
    fines = pd.date_range("2018-12-01", serie.index[-1] - pd.DateOffset(months=H), freq="6MS")
    filas = []
    for fin in fines:
        tr = serie.loc[:fin]
        te = serie.loc[fin + pd.offsets.MonthBegin(1):].iloc[:H]
        if len(te) < H or len(tr) < 48:
            continue
        preds = {
            "SARIMA": ajustar(tr).forecast(H).values,
            "Promedio 12 meses": f_prom12(tr, H),
            "Promedio 12m + estacional": f_prom_est(tr, H),
        }
        for nombre, p in preds.items():
            filas.append({"corte": fin, "modelo": nombre, **metricas(te.values, p)})
    return pd.DataFrame(filas)


def mostrar(fig):
    st.pyplot(fig)
    plt.close(fig)


def etiqueta(d):
    return f"{MESES[d.month - 1]} {d.year}"


def eje_fechas(ax, fechas, cada=3, rot=45):
    """Eje X con etiquetas como 'Oct 2026'. cada = separación en meses entre etiquetas."""
    fechas = pd.DatetimeIndex(fechas)
    marcas = fechas[(fechas.month - 1) % cada == 0]
    ax.set_xticks(marcas)
    ax.set_xticklabels([etiqueta(d) for d in marcas], rotation=rot, ha="right")


# ----------------------------------------------------------------------------- datos y modelo
serie = leer_serie()
firma = os.path.getmtime(F_SERIE)
modelo = cargar_modelo(firma)

st.sidebar.header("Parámetros")
horizonte = st.sidebar.slider("Meses a pronosticar", 3, 24, 12)
nivel = st.sidebar.radio("Intervalo de confianza", [80, 95], horizontal=True)
unidad = st.sidebar.radio("Unidad", ["Toneladas", "Bolsas (50 kg)"])
dec = 0 if unidad.startswith("Bolsas") else 1
u_corta = "bolsas" if unidad.startswith("Bolsas") else "t"


def u(x):
    return x * 1000 / KG if unidad.startswith("Bolsas") else x


st.sidebar.markdown("---")
st.sidebar.caption(
    f"Modelo SARIMA{tuple(CFG['order'])}{tuple(CFG['seasonal_order'])}\n\n"
    f"Datos: {serie.index[0].strftime('%m/%Y')} a {serie.index[-1].strftime('%m/%Y')} "
    f"({len(serie)} meses)\n\n1 bolsa = {KG} kg"
)

st.title("Pronóstico de demanda")

t_pron, t_hist, t_est, t_des, t_seg, t_dat = st.tabs(
    ["Pronóstico", "Histórico", "Estacionalidad", "Desempeño del modelo", "Seguimiento", "Datos"]
)

# ============================================================================= PRONÓSTICO
with t_pron:
    media, lo, hi = pronosticar(modelo, horizonte, nivel)
    idx = media.index
    ref_simple = f_prom_est(serie, horizonte)
    anio_ant = serie.reindex(idx - pd.DateOffset(years=1))

    tabla = pd.DataFrame(
        {
            "Pronóstico": u(media.values),
            f"Mínimo {nivel}%": u(lo.values),
            f"Máximo {nivel}%": u(hi.values),
            "Referencia simple": u(ref_simple),
            "Mismo mes año anterior": u(anio_ant.values),
        },
        index=idx,
    )
    tabla["Variación vs año anterior (%)"] = 100 * (tabla["Pronóstico"] / tabla["Mismo mes año anterior"] - 1)

    total = tabla["Pronóstico"].sum()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"Total {horizonte} meses ({u_corta})", f"{total:,.0f}")
    c2.metric(f"Promedio mensual ({u_corta})", f"{tabla['Pronóstico'].mean():,.0f}")
    if anio_ant.notna().all():
        var = 100 * (total / tabla["Mismo mes año anterior"].sum() - 1)
        c3.metric("Vs mismos meses del año anterior", f"{var:+.1f}%")
    else:
        c3.metric("Vs año anterior", "n/d")
    c4.metric("Mes más alto / más bajo",
              f"{MESES[tabla['Pronóstico'].idxmax().month - 1]} / {MESES[tabla['Pronóstico'].idxmin().month - 1]}")

    fig, ax = plt.subplots(figsize=(12, 4.5))
    hist = serie.iloc[-36:]
    ax.plot(hist.index, u(hist.values), color="black", lw=2, label="Real")
    ax.plot(idx, tabla["Pronóstico"], color="tab:green", lw=2.5, marker="o", label="Pronóstico")
    ax.plot(idx, tabla["Referencia simple"], color="tab:orange", lw=1.5, ls="--", label="Referencia simple")
    ax.fill_between(idx, tabla.iloc[:, 1], tabla.iloc[:, 2], color="tab:green", alpha=0.2, label=f"Intervalo {nivel}%")
    ax.axvline(idx[0], color="gray", ls=":")
    ax.set_ylabel(unidad)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", ncol=4)
    ax.set_title("Últimos 36 meses reales y pronóstico")
    eje_fechas(ax, hist.index.append(idx), cada=3)
    mostrar(fig)

    col_a, col_b = st.columns([3, 2])
    with col_a:
        st.subheader("Detalle por mes")
        t = tabla.copy()
        t.index = [etiqueta(d) for d in t.index]
        t = t.round(dec)
        t["Variación vs año anterior (%)"] = tabla["Variación vs año anterior (%)"].round(1).values
        st.dataframe(t, height=(len(t) + 1) * 35 + 3)
        st.download_button("Descargar pronóstico (CSV)", t.to_csv().encode("utf-8"),
                           file_name="pronostico.csv", mime="text/csv")
    with col_b:
        st.subheader("Comparación con el año anterior")
        comp = tabla.dropna(subset=["Mismo mes año anterior"]).iloc[:12]
        if len(comp):
            fig, ax = plt.subplots(figsize=(6, 4))
            x = np.arange(len(comp))
            ax.bar(x - 0.2, comp["Mismo mes año anterior"], 0.4, color="lightgray", label="Año anterior")
            ax.bar(x + 0.2, comp["Pronóstico"], 0.4, color="tab:green", label="Pronóstico")
            ax.set_xticks(x)
            ax.set_xticklabels([MESES[d.month - 1] for d in comp.index])
            ax.set_ylabel(unidad)
            ax.legend()
            ax.grid(alpha=0.3, axis="y")
            mostrar(fig)
        else:
            st.write("No hay datos del año anterior para esos meses.")

    st.info(
        "**Cómo usarlo:** el valor central sirve de referencia. Para evitar faltantes planifica con el "
        "máximo del intervalo; el mínimo es lo que casi seguro se vende. El total del período es más "
        "estable que cada mes por separado. La *referencia simple* es el nivel de los últimos 12 meses "
        "más el efecto estacional: si coincide con el modelo, el pronóstico es robusto."
    )

# ============================================================================= HISTÓRICO
with t_hist:
    st.caption("Serie con abril y mayo de 2020 (COVID) suavizados, tal como se usó para entrenar el modelo.")
    d = serie.to_frame("t")
    d["anio"] = d.index.year
    anual = d.groupby("anio")["t"].agg(total="sum", meses="count", promedio="mean")
    completo = anual["meses"] == 12
    anual["var"] = np.where(completo & completo.shift(1, fill_value=False), anual["total"].pct_change() * 100, np.nan)

    col_a, col_b = st.columns([3, 2])
    with col_a:
        fig, ax = plt.subplots(figsize=(8, 4))
        colores = ["tab:blue" if c else "lightsteelblue" for c in completo]
        ax.bar(anual.index.astype(str), u(anual["total"]), color=colores)
        ax.set_ylabel(f"Total anual ({unidad})")
        ax.set_title("Demanda total por año (claro = año incompleto)")
        ax.tick_params(axis="x", rotation=60)
        ax.grid(alpha=0.3, axis="y")
        mostrar(fig)
    with col_b:
        ta = pd.DataFrame({
            "Meses": anual["meses"],
            f"Total ({u_corta})": u(anual["total"]).round(0),
            f"Promedio mensual ({u_corta})": u(anual["promedio"]).round(dec),
            "Var. vs año anterior (%)": anual["var"].round(1),
        })
        ta.index = ta.index.astype(str)
        st.dataframe(ta, height=380)

    st.subheader("Mapa de calor: año × mes")
    piv = d.assign(mes=d.index.month).pivot(index="anio", columns="mes", values="t")
    fig, ax = plt.subplots(figsize=(11, 5))
    im = ax.imshow(u(piv.values), aspect="auto", cmap="YlGnBu")
    ax.set_xticks(range(12))
    ax.set_xticklabels(MESES)
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels(piv.index)
    vals = u(piv.values)
    vmax = np.nanmax(vals)
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            if not np.isnan(vals[i, j]):
                ax.text(j, i, f"{vals[i, j]:,.0f}", ha="center", va="center", fontsize=7,
                        color="white" if vals[i, j] > 0.6 * vmax else "black")
    fig.colorbar(im, ax=ax, label=unidad)
    mostrar(fig)

    st.subheader("Curva mensual por año")
    anios = list(piv.index)
    elegidos = st.multiselect("Años a resaltar (vacío = todos)", anios, default=anios[-2:])
    cmap = plt.get_cmap("tab20", len(anios))
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for i, a in enumerate(anios):
        if not elegidos:
            ax.plot(range(1, 13), u(piv.loc[a].values), color=cmap(i), lw=1.5, marker="o", ms=3, label=str(a))
        elif a in elegidos:
            ax.plot(range(1, 13), u(piv.loc[a].values), color=cmap(i), lw=3, marker="o", ms=5, label=str(a), zorder=3)
        else:
            ax.plot(range(1, 13), u(piv.loc[a].values), color="lightgray", lw=1, zorder=1)
    ax.plot(range(1, 13), u(piv.mean().values), color="red", lw=3.5, ls="--", marker="s", label="Promedio", zorder=10)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(MESES)
    ax.set_ylabel(unidad)
    ax.grid(alpha=0.3)
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", ncol=2, fontsize=8)
    mostrar(fig)

# ============================================================================= ESTACIONALIDAD
with t_est:
    if len(serie) >= 36:
        stl = STL(serie, period=12, robust=True).fit()
        fuerza = max(0.0, 1 - stl.resid.var() / (stl.seasonal + stl.resid).var())
        efecto = stl.seasonal.groupby(stl.seasonal.index.month).mean()
        c1, c2, c3 = st.columns(3)
        c1.metric("Fuerza estacional (0 a 1)", f"{fuerza:.2f}")
        c2.metric("Mes más fuerte", MESES[int(efecto.idxmax()) - 1])
        c3.metric("Mes más débil", MESES[int(efecto.idxmin()) - 1])

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(MESES, u(efecto.values), color=["tab:green" if v >= 0 else "tab:red" for v in efecto.values])
        ax.axhline(0, color="black", lw=0.8)
        ax.set_ylabel(f"Efecto sobre el nivel ({unidad})")
        ax.set_title("Efecto estacional promedio por mes")
        ax.grid(alpha=0.3, axis="y")
        mostrar(fig)

        fig, axs = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
        axs[0].plot(stl.trend.index, u(stl.trend), color="tab:blue")
        axs[0].set_title("Tendencia (nivel de fondo)")
        axs[1].plot(stl.seasonal.index, u(stl.seasonal), color="tab:green")
        axs[1].set_title("Componente estacional")
        axs[2].plot(stl.resid.index, u(stl.resid), color="tab:gray")
        axs[2].set_title("Residuo (lo que no explican ni nivel ni estación)")
        for a in axs:
            a.grid(alpha=0.3)
        plt.tight_layout()
        mostrar(fig)
        nivel_est = "floja" if fuerza < 0.3 else ("moderada" if fuerza < 0.5 else "marcada")
        st.info(
            f"La estacionalidad es {nivel_est} (fuerza {fuerza:.2f}). Los meses extremos se alejan hasta "
            "un 10 % del nivel, pero el mes explica una parte pequeña de la variación: pesa más el nivel "
            "de cada año. Fuerza < 0.3 = floja; entre 0.3 y 0.5 = moderada; > 0.5 = marcada. "
            "Los gráficos de descomposición reflejan también la mayor volatilidad reciente."
        )
    else:
        st.write("Se necesitan al menos 36 meses de datos.")

# ============================================================================= DESEMPEÑO
with t_des:
    st.subheader("Prueba con los últimos meses")
    h = st.slider("Meses reservados para la prueba (el modelo se reentrena sin ellos)", 6, 24, 12)
    ev = evaluar_holdout(firma, CFG_TXT, h)
    te, tr = ev["te"], ev["tr"]
    filas = {n: metricas(te.values, ev[n], tr) for n in ["SARIMA", "Promedio 12 meses", "Promedio 12m + estacional"]}
    filas["SARIMA"]["Cobertura IC (%)"] = metricas(te.values, ev["SARIMA"], tr, ev["lo80"], ev["hi80"])["Cobertura IC (%)"]
    mt = pd.DataFrame(filas).T
    ref = mt.loc["Promedio 12 meses", "MAE (t)"]
    mt["Mejora vs promedio 12m (%)"] = 100 * (1 - mt["MAE (t)"] / ref)
    st.dataframe(mt.round(2))
    st.caption("Cobertura IC: % de meses reales dentro del intervalo de 80 % (debería rondar 80). "
               "Sesgo negativo = el modelo pronostica de más. MASE < 1 = supera repetir el año anterior.")

    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(te.index, u(te.values), color="black", lw=2.5, label="Real")
    ax.plot(te.index, u(ev["SARIMA"]), color="tab:green", lw=2, marker="o", label="SARIMA")
    ax.plot(te.index, u(ev["Promedio 12 meses"]), color="tab:orange", ls="--", label="Promedio 12 meses")
    ax.fill_between(te.index, u(ev["lo80"]), u(ev["hi80"]), color="tab:green", alpha=0.2, label="Intervalo 80%")
    ax.set_ylabel(unidad)
    ax.grid(alpha=0.3)
    ax.legend(ncol=4)
    ax.set_title("Pronóstico vs. real en los meses de prueba")
    eje_fechas(ax, te.index, cada=1 if len(te) <= 12 else 2)
    mostrar(fig)

    det = pd.DataFrame({"Real": u(te.values), "Pronóstico": u(ev["SARIMA"])}, index=te.index)
    det["Error"] = det["Real"] - det["Pronóstico"]
    det["Error (%)"] = 100 * det["Error"] / det["Real"]
    det.index = [etiqueta(d) for d in det.index]
    st.dataframe(det.round(1), height=(len(det) + 1) * 35 + 3)

    st.subheader("Validación móvil (varios cortes en el tiempo)")
    st.write("Reentrena el modelo en distintas fechas y pronostica 12 meses cada vez. "
             "Es la medida más representativa del error típico.")
    if st.button("Ejecutar validación móvil"):
        st.session_state["vm"] = True
    if st.session_state.get("vm"):
        vm = validacion_movil(firma, CFG_TXT)
        if len(vm):
            prom = vm.groupby("modelo")[["MAE (t)", "RMSE (t)", "WAPE (%)", "Precisión (%)", "Sesgo (%)"]].mean()
            st.dataframe(prom.sort_values("MAE (t)").round(2))
            pv = vm.pivot(index="corte", columns="modelo", values="MAE (t)")
            fig, ax = plt.subplots(figsize=(11, 4))
            for c in pv.columns:
                ax.plot(pv.index, pv[c], marker="o", label=c)
            ax.set_ylabel("MAE (t)")
            ax.set_title("Error por fecha de corte")
            eje_fechas(ax, pv.index, cada=1)
            ax.grid(alpha=0.3)
            ax.legend()
            mostrar(fig)
        else:
            st.write("No hay suficientes datos para la validación móvil.")

    st.subheader("Diagnóstico del modelo ajustado")
    col_a, col_b = st.columns(2)
    with col_a:
        pa = pd.DataFrame({"Coeficiente": modelo.params, "p-valor": modelo.pvalues}).round(3)
        st.write("Parámetros")
        st.dataframe(pa)
        lb = acorr_ljungbox(modelo.resid.iloc[13:], lags=[12, 24])
        lb.index = ["12 rezagos", "24 rezagos"]
        st.write("Ljung-Box de residuos (p > 0.05 = no queda estructura sin explicar)")
        st.dataframe(lb.round(3))
    with col_b:
        fig, axs = plt.subplots(2, 1, figsize=(6, 5))
        axs[0].plot(modelo.resid.iloc[13:], color="tab:gray")
        axs[0].axhline(0, color="black", lw=0.8)
        axs[0].set_title("Residuos del modelo")
        plot_acf(modelo.resid.iloc[13:], lags=24, ax=axs[1], title="ACF de residuos")
        plt.tight_layout()
        mostrar(fig)

# ============================================================================= SEGUIMIENTO
with t_seg:
    st.write("Cada vez que cierra un mes, registra la demanda real: la app guarda lo que el modelo había "
             "pronosticado para ese mes, lo compara y reentrena con el dato nuevo.")
    siguiente = serie.index[-1] + pd.offsets.MonthBegin(1)
    st.write(f"Próximo mes a registrar: **{etiqueta(siguiente)}**")
    c1, c2 = st.columns([1, 2])
    with c1:
        real = st.number_input("Demanda real del mes (toneladas)", min_value=0.0, value=0.0, step=1.0)
        st.caption(f"= {real * 1000 / KG:,.0f} bolsas")
        if st.button("Guardar y reentrenar"):
            if real <= 0:
                st.error("Ingresa las toneladas reales (mayor que 0).")
            else:
                m1, l1, h1 = pronosticar(modelo, 1, 80)
                nueva_fila = pd.DataFrame([{
                    "mes": siguiente.strftime("%Y-%m-%d"), "pronostico": float(m1.iloc[0]),
                    "bajo80": float(l1.iloc[0]), "alto80": float(h1.iloc[0]), "real": real}])
                if os.path.exists(F_SEGUIMIENTO):
                    nueva_fila = pd.concat([pd.read_csv(F_SEGUIMIENTO), nueva_fila])
                nueva_fila.to_csv(F_SEGUIMIENTO, index=False)
                nueva = pd.concat([serie, pd.Series([real], index=[siguiente])])
                nueva.name = "toneladas"
                nueva.to_csv(F_SERIE, header=True, index_label="fecha")
                st.cache_resource.clear()
                st.cache_data.clear()
                st.rerun()
    with c2:
        if os.path.exists(F_SEGUIMIENTO):
            seg = pd.read_csv(F_SEGUIMIENTO, parse_dates=["mes"])
            seg["error"] = seg["real"] - seg["pronostico"]
            seg["error_pct"] = 100 * seg["error"] / seg["real"]
            seg["dentro80"] = (seg["real"] >= seg["bajo80"]) & (seg["real"] <= seg["alto80"])
            sesgo = 100 * seg["error"].sum() / seg["real"].sum()
            ultimos = seg.tail(3)
            if len(ultimos) == 3 and not ultimos["dentro80"].any():
                st.error("Los últimos 3 meses cayeron fuera del intervalo de 80 %: el nivel de la demanda "
                         "cambió. Conviene reentrenar con una ventana más reciente desde el notebook.")
            elif not ultimos["dentro80"].all():
                st.warning("Hay meses recientes fuera del intervalo de 80 %. Vigila el próximo cierre.")
            else:
                st.success("Los últimos meses registrados están dentro del intervalo esperado.")
            m1_, m2_, m3_ = st.columns(3)
            m1_.metric("Meses registrados", len(seg))
            m2_.metric("Sesgo acumulado", f"{sesgo:+.1f}%")
            m3_.metric("Dentro del intervalo 80%", f"{100 * seg['dentro80'].mean():.0f}%")
            ver = seg.copy()
            ver["mes"] = ver["mes"].map(etiqueta)
            ver = ver.rename(columns={"pronostico": "Pronóstico (t)", "bajo80": "Mín 80% (t)",
                                      "alto80": "Máx 80% (t)", "real": "Real (t)", "error": "Error (t)",
                                      "error_pct": "Error (%)", "dentro80": "Dentro 80%"})
            st.dataframe(ver.round(1).set_index("mes"))
        else:
            st.write("Aún no hay meses registrados.")

# ============================================================================= DATOS
with t_dat:
    st.subheader("Serie mensual usada por el modelo")
    sm = serie.to_frame("Toneladas")
    sm["Bolsas (50 kg)"] = sm["Toneladas"] * 1000 / KG
    sm.index = [etiqueta(d) for d in sm.index]
    st.dataframe(sm.round(1), height=300)
    st.download_button("Descargar serie (CSV)", sm.round(1).to_csv().encode("utf-8"),
                       file_name="serie_mensual.csv", mime="text/csv")

    if os.path.exists(F_VENTAS):
        st.subheader("Clientes (últimos 12 meses de ventas)")
        v = pd.read_csv(F_VENTAS, parse_dates=["fecha"])
        fin = v["fecha"].max()
        rec = v[v["fecha"] > fin - pd.DateOffset(months=12)]
        top = rec.groupby("cliente")["peso_kg"].sum().sort_values(ascending=False)
        total_kg = top.sum()
        tt = pd.DataFrame({
            f"Volumen ({u_corta})": u(top.head(10) / 1000).round(0),
            "Participación (%)": (100 * top.head(10) / total_kg).round(1),
        })
        c1, c2, c3 = st.columns(3)
        c1.metric("Clientes activos", f"{top.size}")
        c2.metric("Peso de los 5 mayores", f"{100 * top.head(5).sum() / total_kg:.0f}%")
        c3.metric("Período", f"{(fin - pd.DateOffset(months=12)).strftime('%m/%Y')} a {fin.strftime('%m/%Y')}")
        st.dataframe(tt)
        if "precio_kg" in v.columns:
            pk = v.groupby(v["fecha"].dt.year)["precio_kg"].median()
            fig, ax = plt.subplots(figsize=(9, 3.5))
            ax.plot(pk.index, pk.values, marker="o", color="tab:purple")
            ax.set_ylabel("Bs por kg (mediana)")
            ax.set_title("Precio por kg por año")
            ax.grid(alpha=0.3)
            mostrar(fig)
    else:
        st.caption("Para ver clientes y precios, copia `ventas_limpio.csv` en la carpeta de la app.")
