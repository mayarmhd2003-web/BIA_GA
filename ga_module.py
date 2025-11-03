import random
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import gc

def individual(n_features, p_select=0.2):
    return [1 if random.random() < p_select else 0 for _ in range(n_features)]

def fitness(chromosome, X, y):
    mask = np.array(chromosome, dtype=bool)
    if mask.sum() == 0:
        return 0.0
    
    X_sub = X.iloc[:, mask]
    
    model = RandomForestClassifier(
        n_estimators=15,
        max_depth=4,
        random_state=42,
        n_jobs=-1
    )
    try:
        score = float(np.mean(cross_val_score(model, X_sub, y, cv=2, scoring='accuracy')))
    except Exception:
        return 0.0
    
    penalty = 0.005 * (mask.sum() / X.shape[1])
    result = score - penalty
    
    del model, X_sub
    return result

def tournament_selection(pop, fitnesses, k=2):
    selected_idx = random.sample(range(len(pop)), min(k, len(pop)))
    return pop[max(selected_idx, key=lambda i: fitnesses[i])]

def single_point_crossover(a, b):
    n = len(a)
    if n < 2:
        return a[:], b[:]
    pt = random.randint(1, n-1)
    return a[:pt] + b[pt:], b[:pt] + a[pt:]

def mutate(chrom, rate=0.01):
    return [1 - g if random.random() < rate else g for g in chrom]

def _chromosome_to_string(chrom):
    return ''.join(map(str, chrom))

def run_genetic_algorithm(df, target_name, pop_size=15, generations=10, crossover_rate=0.8, mutation_rate=0.02, verbose=True):
    gc.collect()
    
    if target_name not in df.columns:
        target_name = df.columns[-1]

    X = df.drop(columns=[target_name])
    y = df[target_name]
    
    if X.shape[1] > 20:
        X = X.iloc[:, :20]
    
    try:
        y = y.astype(int)
    except Exception:
        y = pd.to_numeric(y, errors='coerce').fillna(0).astype(int)

    n_features = X.shape[1]
    
    pop = [individual(n_features, p_select=0.3) for _ in range(pop_size)]
    best_solution = None
    best_fitness = -1.0
    history = []

    if verbose:
        print(f"[GA] بدء التشغيل: المجتمع={pop_size}, الأجيال={generations}, الميزات={n_features}")

    for gen in range(generations):
        if gen % 3 == 0:
            gc.collect()
            
        fitnesses = [fitness(ind, X, y) for ind in pop]

        gen_best_idx = np.argmax(fitnesses)
        gen_best_f = float(fitnesses[gen_best_idx])
        gen_best = pop[gen_best_idx]
        gen_selected_count = int(sum(gen_best))

        history.append({
            'generation': gen, 
            'best_fitness': gen_best_f, 
            'selected_count': gen_selected_count
        })

        if gen_best_f > best_fitness:
            best_fitness = gen_best_f
            best_solution = gen_best.copy()

        if verbose and gen % 2 == 0:
            print(f"[GA] الجيل {gen:02d} | اللياقة={gen_best_f:.4f} | المختارة={gen_selected_count}")

        new_pop = []
        while len(new_pop) < pop_size:
            p1 = tournament_selection(pop, fitnesses)
            p2 = tournament_selection(pop, fitnesses)
            
            if random.random() < crossover_rate:
                c1, c2 = single_point_crossover(p1, p2)
            else:
                c1, c2 = p1[:], p2[:]
            
            c1 = mutate(c1, mutation_rate)
            c2 = mutate(c2, mutation_rate)
            
            new_pop.extend([c1, c2])
        
        pop = new_pop[:pop_size]
        
        del new_pop

    if best_solution is None:
        best_solution = pop[0]

    final_score = float(fitness(best_solution, X, y))
    best_selected_count = int(sum(best_solution))
    best_chromosome_str = _chromosome_to_string(best_solution)
    selected_features = [X.columns[i] for i, bit in enumerate(best_solution) if bit]

    if verbose:
        print("--------------------------------------------------")
        print(f"[GA] الانتهاء. أفضل لياقة نهائية = {final_score:.6f}")
        print(f"[GA] عدد الميزات المختارة = {best_selected_count}")
        print("--------------------------------------------------")

    del X, y, pop, fitnesses
    gc.collect()

    return {
        'method': 'genetic',
        'selected_features': selected_features,
        'final_score': final_score,
        'history': history,
        'best_chromosome': best_solution,
        'best_chromosome_str': best_chromosome_str,
        'best_selected_count': best_selected_count
    }
