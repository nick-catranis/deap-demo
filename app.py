"""
DEAP Genetic Algorithm Demo — Flask backend
Exposes two GA problems:
  1. OneMax  — maximize the number of 1s in a binary string
  2. FuncOpt — minimize a continuous function (Rastrigin)
"""

import random
import math
import json
import numpy as np
from flask import Flask, request, jsonify, send_from_directory

from deap import base, creator, tools, algorithms

app = Flask(__name__, static_folder=".")


# ── helpers ────────────────────────────────────────────────────────────────

def _fresh_toolbox():
    """Return a clean Toolbox (avoids DEAP's global creator conflicts)."""
    return base.Toolbox()


# ── OneMax ─────────────────────────────────────────────────────────────────

def run_onemax(pop_size=100, n_gen=40, cx_prob=0.7, mut_prob=0.2, ind_size=50, seed=42):
    random.seed(seed)

    # Re-create types each call to avoid 'already defined' errors
    if "FitnessMax" not in creator.__dict__:
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if "IndividualOM" not in creator.__dict__:
        creator.create("IndividualOM", list, fitness=creator.FitnessMax)

    tb = _fresh_toolbox()
    tb.register("attr_bool", random.randint, 0, 1)
    tb.register("individual", tools.initRepeat, creator.IndividualOM, tb.attr_bool, ind_size)
    tb.register("population", tools.initRepeat, list, tb.individual)

    def eval_onemax(ind):
        return (sum(ind),)

    tb.register("evaluate", eval_onemax)
    tb.register("mate", tools.cxTwoPoint)
    tb.register("mutate", tools.mutFlipBit, indpb=0.05)
    tb.register("select", tools.selTournament, tournsize=3)

    pop = tb.population(n=pop_size)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", lambda vals: round(sum(v[0] for v in vals) / len(vals), 3))
    stats.register("max", lambda vals: max(v[0] for v in vals))

    history = []

    for gen in range(n_gen):
        # Evaluate
        invalid = [ind for ind in pop if not ind.fitness.valid]
        fitnesses = list(map(tb.evaluate, invalid))
        for ind, fit in zip(invalid, fitnesses):
            ind.fitness.values = fit

        hof.update(pop)
        record = stats.compile(pop)
        history.append({"gen": gen, "avg": record["avg"], "max": record["max"]})

        # Select + reproduce
        offspring = tb.select(pop, len(pop))
        offspring = list(map(tb.clone, offspring))

        for c1, c2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cx_prob:
                tb.mate(c1, c2)
                del c1.fitness.values, c2.fitness.values

        for mut in offspring:
            if random.random() < mut_prob:
                tb.mutate(mut)
                del mut.fitness.values

        pop[:] = offspring

    best = list(hof[0])
    return {
        "history": history,
        "best_fitness": hof[0].fitness.values[0],
        "best_individual": best,
        "ind_size": ind_size,
    }


# ── Rastrigin function optimisation ───────────────────────────────────────

def rastrigin(x, y):
    A = 10
    return A * 2 + (x**2 - A * math.cos(2 * math.pi * x)) + (y**2 - A * math.cos(2 * math.pi * y))


def run_funcopt(pop_size=100, n_gen=60, cx_prob=0.6, mut_prob=0.3,
                sigma=0.3, low=-5.12, high=5.12, seed=42):
    random.seed(seed)

    if "FitnessMin" not in creator.__dict__:
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    if "IndividualFO" not in creator.__dict__:
        creator.create("IndividualFO", list, fitness=creator.FitnessMin)

    tb = _fresh_toolbox()
    tb.register("attr_float", random.uniform, low, high)
    tb.register("individual", tools.initRepeat, creator.IndividualFO, tb.attr_float, 2)
    tb.register("population", tools.initRepeat, list, tb.individual)

    def eval_rastrigin(ind):
        x, y = ind[0], ind[1]
        # clamp
        x = max(low, min(high, x))
        y = max(low, min(high, y))
        return (rastrigin(x, y),)

    tb.register("evaluate", eval_rastrigin)
    tb.register("mate", tools.cxBlend, alpha=0.5)
    tb.register("mutate", tools.mutGaussian, mu=0, sigma=sigma, indpb=0.5)
    tb.register("select", tools.selTournament, tournsize=3)

    pop = tb.population(n=pop_size)
    hof = tools.HallOfFame(1)

    history = []
    snapshot_gens = set(range(0, n_gen, max(1, n_gen // 8)))

    for gen in range(n_gen):
        invalid = [ind for ind in pop if not ind.fitness.valid]
        fitnesses = list(map(tb.evaluate, invalid))
        for ind, fit in zip(invalid, fitnesses):
            ind.fitness.values = fit

        hof.update(pop)

        fits = [ind.fitness.values[0] for ind in pop]
        avg_fit = round(sum(fits) / len(fits), 4)
        best_fit = round(min(fits), 4)
        record = {"gen": gen, "avg": avg_fit, "min": best_fit}

        if gen in snapshot_gens:
            record["population"] = [[round(ind[0], 3), round(ind[1], 3)] for ind in pop]

        history.append(record)

        offspring = tb.select(pop, len(pop))
        offspring = list(map(tb.clone, offspring))

        for c1, c2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cx_prob:
                tb.mate(c1, c2)
                del c1.fitness.values, c2.fitness.values

        for mut in offspring:
            if random.random() < mut_prob:
                tb.mutate(mut)
                del mut.fitness.values

        pop[:] = offspring

    bx, by = hof[0][0], hof[0][1]
    return {
        "history": history,
        "best_x": round(bx, 5),
        "best_y": round(by, 5),
        "best_fitness": round(hof[0].fitness.values[0], 6),
        "bounds": [low, high],
    }


# ── Trading Strategy Optimiser ─────────────────────────────────────────────

# Gene bounds: [ma_fast, ma_slow, rsi_period, rsi_ob, stop_loss, take_profit]
# Strategy: buy when fast MA crosses above slow MA
#           sell when RSI > rsi_ob OR stop-loss OR take-profit hits
TR_BOUNDS_LOW  = [2,  10,  5, 60, 0.010, 0.020]
TR_BOUNDS_HIGH = [15, 80, 30, 85, 0.060, 0.120]
N_TR_GENES = 6


def _decode(encoded):
    """Map [0,1] genes → actual parameter values."""
    return [lo + e * (hi - lo)
            for e, lo, hi in zip(encoded, TR_BOUNDS_LOW, TR_BOUNDS_HIGH)]


def _generate_prices(n=1200, mu=0.0003, sigma=0.013, seed=42):
    rng = np.random.default_rng(seed)
    returns = rng.normal(mu, sigma, n)
    return 100.0 * np.cumprod(1 + returns)


def _compute_rsi(prices, period):
    period = max(2, int(period))
    deltas = np.diff(prices)
    gains  = np.where(deltas > 0, deltas,  0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    ag = np.zeros(len(prices))
    al = np.zeros(len(prices))
    ag[period] = gains[:period].mean()
    al[period] = losses[:period].mean()
    for i in range(period + 1, len(prices)):
        ag[i] = (ag[i-1] * (period - 1) + gains[i-1])  / period
        al[i] = (al[i-1] * (period - 1) + losses[i-1]) / period
    with np.errstate(divide='ignore', invalid='ignore'):
        rs  = np.where(al == 0, np.inf, ag / al)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    rsi[:period] = 50.0
    return rsi


def _compute_ma(prices, period):
    period = max(1, int(period))
    kernel = np.ones(period) / period
    ma = np.convolve(prices, kernel, mode='full')[:len(prices)]
    ma[:period - 1] = prices[:period - 1]
    return ma


def _backtest(prices, encoded, commission=0.001):
    p = _decode(encoded)
    ma_fast     = max(2, int(round(p[0])))
    ma_slow     = max(ma_fast + 2, int(round(p[1])))
    rsi_period  = max(3, int(round(p[2])))
    rsi_ob      = float(np.clip(p[3], 55, 90))
    stop_loss   = float(np.clip(p[4], 0.005, 0.10))
    take_profit = float(np.clip(p[5], stop_loss * 1.5, 0.20))

    rsi     = _compute_rsi(prices, rsi_period)
    fast_ma = _compute_ma(prices, ma_fast)
    slow_ma = _compute_ma(prices, ma_slow)

    cash      = 1.0
    position  = 0.0
    entry_px  = 0.0
    equity    = [1.0]
    n_trades  = 0
    warmup    = max(rsi_period, ma_slow) + 1

    for i in range(warmup, len(prices)):
        px = float(prices[i])
        if position > 0:
            pnl = (px - entry_px) / entry_px
            # Exit: MA reversal, RSI overbought, stop-loss, or take-profit
            if fast_ma[i] < slow_ma[i] or pnl <= -stop_loss or pnl >= take_profit or rsi[i] > rsi_ob:
                cash = position * px * (1 - commission)
                position = 0.0
                n_trades += 1
        else:
            # Enter long: fast MA above slow MA (uptrend)
            if fast_ma[i] > slow_ma[i]:
                position = cash / (px * (1 + commission))
                entry_px = px
                cash = 0.0
        equity.append(cash + position * px)

    if position > 0:
        equity[-1] = position * float(prices[-1]) * (1 - commission)

    eq  = np.array(equity, dtype=float)
    ret = np.diff(eq) / np.where(eq[:-1] == 0, 1e-9, eq[:-1])

    if len(ret) < 5 or ret.std() < 1e-10:
        return -10.0, 1.0, equity, 0

    sharpe = float((ret.mean() / ret.std()) * np.sqrt(252))
    peak   = np.maximum.accumulate(eq)
    max_dd = float(((peak - eq) / np.where(peak == 0, 1e-9, peak)).max())
    return sharpe, max_dd, equity, n_trades


def run_trading(pop_size=50, n_gen=25, cx_prob=0.65, mut_prob=0.25,
                n_wf_windows=3, commission=0.001, seed=42):
    random.seed(seed)
    np.random.seed(seed)

    prices     = _generate_prices(n=1500, seed=seed)
    train_size = 300
    test_size  = 150

    windows = []
    for w in range(int(n_wf_windows)):
        s  = w * test_size
        te = s + train_size
        oe = te + test_size
        if oe <= len(prices):
            windows.append((prices[s:te], prices[te:oe]))

    # DEAP multi-objective setup
    if "FitnessMO" not in creator.__dict__:
        creator.create("FitnessMO", base.Fitness, weights=(1.0, -1.0))
    if "IndividualTR" not in creator.__dict__:
        creator.create("IndividualTR", list, fitness=creator.FitnessMO)

    def _make_ind():
        return creator.IndividualTR([random.random() for _ in range(N_TR_GENES)])

    def _eval(ind, train_p):
        sh, dd, _, nt = _backtest(train_p, ind, commission)
        if nt < 2:
            return (-1.0, 0.0)   # penalise inactivity gently
        return (sh, dd)

    wf_results   = []
    oos_segments = []
    pareto_front = []

    for win_idx, (train_p, test_p) in enumerate(windows):
        tb = _fresh_toolbox()
        tb.register("individual", _make_ind)
        tb.register("population", tools.initRepeat, list, tb.individual)
        tb.register("evaluate", _eval, train_p=train_p)
        tb.register("mate",   tools.cxBlend, alpha=0.5)
        tb.register("mutate", tools.mutGaussian, mu=0, sigma=0.12, indpb=0.35)
        tb.register("select", tools.selNSGA2)

        pop = tb.population(n=pop_size)
        hof = tools.ParetoFront()

        for ind in pop:
            ind.fitness.values = tb.evaluate(ind)
        pop[:] = tb.select(pop, pop_size)
        hof.update(pop)

        for _ in range(n_gen):
            offspring = list(map(tb.clone, tools.selTournament(pop, pop_size, tournsize=3)))
            for c1, c2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < cx_prob:
                    tb.mate(c1, c2)
                    for ind in (c1, c2):
                        for j in range(N_TR_GENES):
                            ind[j] = max(0.0, min(1.0, ind[j]))
                    del c1.fitness.values, c2.fitness.values
            for mut in offspring:
                if random.random() < mut_prob:
                    tb.mutate(mut)
                    for j in range(N_TR_GENES):
                        mut[j] = max(0.0, min(1.0, mut[j]))
                    del mut.fitness.values
            invalid = [ind for ind in offspring if not ind.fitness.valid]
            for ind in invalid:
                ind.fitness.values = tb.evaluate(ind)
            pop[:] = tb.select(pop + offspring, pop_size)
            hof.update(pop)

        # Best = highest Sharpe on Pareto front
        best = max(hof, key=lambda ind: ind.fitness.values[0])
        oos_sh, oos_dd, oos_eq, oos_nt = _backtest(test_p, best, commission)
        dec = _decode(best)

        wf_results.append({
            "window":       win_idx + 1,
            "is_sharpe":    round(float(best.fitness.values[0]), 3),
            "is_drawdown":  round(float(best.fitness.values[1]), 3),
            "oos_sharpe":   round(float(oos_sh), 3),
            "oos_drawdown": round(float(oos_dd), 3),
            "oos_trades":   int(oos_nt),
            "best_params": {
                "ma_fast":         int(round(dec[0])),
                "ma_slow":         int(round(dec[1])),
                "rsi_period":      int(round(dec[2])),
                "rsi_ob":          round(dec[3], 1),
                "stop_loss_pct":   round(dec[4], 3),
                "take_profit_pct": round(dec[5], 3),
            },
        })
        oos_segments.append(np.array(oos_eq))

        if win_idx == len(windows) - 1:
            pareto_front = [
                {"sharpe": round(float(ind.fitness.values[0]), 3),
                 "drawdown": round(float(ind.fitness.values[1]), 3)}
                for ind in hof
            ]

    # Stitch OOS equity segments into one continuous curve
    combined = []
    scale = 1.0
    for seg in oos_segments:
        norm = seg * (scale / seg[0])
        if combined:
            combined.extend(norm[1:].tolist())
        else:
            combined.extend(norm.tolist())
        scale = combined[-1]

    n_show = train_size + len(windows) * test_size
    return {
        "wf_results":     wf_results,
        "combined_equity": [round(float(v), 4) for v in combined],
        "pareto_front":    pareto_front,
        "price_series":   [round(float(v), 2) for v in prices[:n_show]],
        "train_size":     train_size,
        "test_size":      test_size,
    }


# ── routes ─────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/api/onemax", methods=["POST"])
def api_onemax():
    p = request.json or {}
    result = run_onemax(
        pop_size=int(p.get("pop_size", 100)),
        n_gen=int(p.get("n_gen", 40)),
        cx_prob=float(p.get("cx_prob", 0.7)),
        mut_prob=float(p.get("mut_prob", 0.2)),
        ind_size=int(p.get("ind_size", 50)),
        seed=int(p.get("seed", 42)),
    )
    return jsonify(result)


@app.route("/api/funcopt", methods=["POST"])
def api_funcopt():
    p = request.json or {}
    result = run_funcopt(
        pop_size=int(p.get("pop_size", 100)),
        n_gen=int(p.get("n_gen", 60)),
        cx_prob=float(p.get("cx_prob", 0.6)),
        mut_prob=float(p.get("mut_prob", 0.3)),
        sigma=float(p.get("sigma", 0.3)),
        seed=int(p.get("seed", 42)),
    )
    return jsonify(result)


@app.route("/api/trading", methods=["POST"])
def api_trading():
    p = request.json or {}
    result = run_trading(
        pop_size=int(p.get("pop_size", 50)),
        n_gen=int(p.get("n_gen", 25)),
        cx_prob=float(p.get("cx_prob", 0.65)),
        mut_prob=float(p.get("mut_prob", 0.25)),
        n_wf_windows=int(p.get("n_wf_windows", 3)),
        commission=float(p.get("commission", 0.001)),
        seed=int(p.get("seed", 42)),
    )
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, port=5050)
