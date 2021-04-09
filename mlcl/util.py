import os
import re

import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from mlcl.implementations import PXMRF


complexity_classes = [
    'linear',
    'quadratic',
    'cubic',
    'exponential'
]


def init_implementation(impl_name):
    implemented_classes = [PXMRF]

    for imp_cls in implemented_classes:
        if imp_cls.__name__ == impl_name:
            return imp_cls()

    avail = ' OR '.join([f'"{imp.__name__}"' for imp in implemented_classes])
    raise RuntimeError(
        f'Could not instantiate implementation "{impl_name}", please give one of the following:\n  {avail}')


def prepare_dataset_list(data_dir, benchmark_data):
    if os.path.exists(data_dir) and not os.path.isdir(data_dir):
        # Listify single object.
        print(f'Scaling data path "{data_dir}" is a file, so using this and not running further scaling benchmark experiments.')
        scaling_data = [data_dir]

    elif not os.path.exists(data_dir) or len(os.listdir(data_dir)) == 0:
        print(f'Scaling data path "{data_dir} not found or empty, not running scaling benchmark experiments.')
        scaling_data = []
    else:
        scaling_data = sorted([os.path.join(data_dir, file) for file in os.listdir(data_dir) if file.endswith('.pkl')])
    
    if not os.path.exists(benchmark_data) or os.path.isdir(benchmark_data):
        for ds in scaling_data:
            if benchmark_data == os.path.basename(ds):
                bench_data = ds
                break
        else:
            if len(scaling_data) > 0:
                bench_data = scaling_data[-1]
                print(f'Benchmark dataset "{benchmark_data}" not found as path, and also not found in scaling data "{data_dir}", so now using "{bench_data}"')
            else:
                raise RuntimeError(f'Benchmark dataset "{benchmark_data}" not found as path and no scaling data found in "{data_dir}", please give proper paths!')
    else:
        bench_data = benchmark_data
    return scaling_data, bench_data


def aggregate_runtime(measurements, rep_agg=np.mean):
    time = np.array(measurements["APPLY_RUNTIME_WCT"]) / measurements['predict_samples']
    if rep_agg is None:
        return time
    return rep_agg(time)


def aggregate_memory(measurements, gpu=False, rep_agg=np.mean, mem_agg=max):
    if gpu:
        for key in ['APPLY_GPU_memory.used', 'APPLY_GPU_gpu_memory.used']:
            if key in measurements:
                mem = np.array([mem_agg(enc) for enc in measurements[key]])
                break
        else:
            return 0
    else:
        mem = np.array([mem_agg(decode(enc)) for enc in measurements['APPLY_MEMORY_ENC']])
    if rep_agg is None:
        return mem
    return rep_agg(mem)


def aggregate_energy(measurements, gpu=False, rep_agg=np.mean, ene_agg=np.mean):
    if gpu:
        for key in ['APPLY_GPU_gpu_power', 'APPLY_GPU_power', 'APPLY_GPU_power.draw']:
            if key in measurements:
                r_en = np.array([t * ene_agg(l) for t, l in zip(measurements["APPLY_RUNTIME_WCT"], measurements[key])])
                break
        else:
            return 0
    else:
        r_en = np.array([t * l for t, l in zip(measurements["APPLY_RUNTIME_WCT"], measurements['APPLY_ENERGY_package-0'])])
    if rep_agg is None:
        return r_en
    return rep_agg(r_en)


def reformat_value(value, unit, base=1000):
    prefix_map = [
        (1e-3, '\u03BC', 2),
        (1e0, 'm', 1),
        (1e3, '', 0),
        (1e6, 'k', -1),
        (1e9, 'M', -2),
        (1e12, 'G', -3),
    ]
    for cpx, prefix, exp in prefix_map:
        if value < cpx:
            break
    else:
        cpx, prefix, exp = prefix_map[-1]
    final_val = value * (base**(exp))
    if final_val > 100:
        return f'{final_val:>3.0f} {prefix}{unit}'
    return f'{final_val:>4.1f} {prefix}{unit}'


def check_scale(value, bins):
    if sorted(bins) == bins: # ascending order
        if value <= bins[0]:
            return 3
        elif value <= bins[1]:
            return 2
        elif value <= bins[2]:
            return 1
        return 0
    elif sorted(bins, reverse=True) == bins: # descending order
        if value > bins[0]:
            return 3
        elif value > bins[1]:
            return 2
        elif value > bins[2]:
            return 1
        return 0
    else:
        raise RuntimeError(f'Bin list {bins} is not sorted!')


def extract_benchmark_results(benchmark_measurements, runtime_scale, memory_scale, energy_scale):
    if 'cl_name' in benchmark_measurements:
        name = benchmark_measurements['name']
    else:
        # for older logs
        name = ''.join(re.match('(.*)_nodes(\d*).*', benchmark_measurements['name']).groups()).capitalize()
    runtime = aggregate_runtime(benchmark_measurements)
    memory = aggregate_memory(benchmark_measurements)
    energy = aggregate_energy(benchmark_measurements)
    gpu_memory = aggregate_memory(benchmark_measurements, gpu=True)
    gpu_energy = aggregate_energy(benchmark_measurements, gpu=True)
    # remove benchmark measurements
    results = {
        'name': name,
        'runtime': runtime, # seconds
        'memory': memory, # byte
        'energy': energy, # Watt seconds
        'gpu_memory': gpu_memory, # byte
        'gpu_energy': gpu_energy, # Watt seconds
        
        'runtime_rating': check_scale(runtime, runtime_scale),
        'memory_rating': check_scale(memory + gpu_memory, memory_scale),
        'energy_rating': check_scale(energy + gpu_energy, energy_scale),
    }
    return results


def check_complexity(measurements, varied, log_key):
    values = {}
    for vary in varied: # assess all varied fields from logs
        values[vary] = sorted(np.unique([meas[vary] for meas in measurements]))

    onot_complexity = {}
    for vary in varied: # each varied input size
        not_varied = [not_vary for not_vary in varied if not_vary != vary]
        x = []
        y = []
        for meas in measurements: # assess measurements
            if all([meas[not_vary] == values[not_vary][-1] for not_vary in not_varied]):
                x.append(meas[vary])
                y.append(meas[log_key])
        order = np.argsort(x)
        x = np.array(x)[order]
        if len(x) < 5:
            onot_complexity[vary] = np.nan # TODO not enough variation, should maybe receive worst complexity class?
        else:
            complexity = []
            y = np.array(y)[order].T
            for repeat_y in y:
                complexity.append(test_onot_r2(x, repeat_y))
            onot_complexity[vary] = np.argmax(np.bincount(complexity)) # allows for some outliers
    return onot_complexity


# Returns whether the model fitted best on
# 0: Linear
# 1: Quadratic
# 2: Exponential
# TODO add cubic
def test_onot_r2(x, y, n_splits=5):
    n_splits = min(max(n_splits // 6, 2), n_splits) # have at least 6 measurements per split!

    kf = KFold(n_splits=n_splits, shuffle=True)

    r2_linear = []
    r2_quadratic = []
    r2_exp = []

    nr_reruns = 20

    for train_index, test_index in kf.split(x):
        X_train, X_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        if len(X_train.shape) < 2:
            X_train = X_train.reshape(-1, 1)
        if len(X_test.shape) < 2:
            X_test = X_test.reshape(-1, 1)

        for _ in range(nr_reruns):
            lr = LinearRegression()
            lr.fit(X_train, y_train)

            scr = r2_score(y_test, lr.predict(X_test))
            if scr > 0:
                r2_linear.append(scr)

        for _ in range(nr_reruns):
            lr = LinearRegression()
            lr.fit(X_train, np.sqrt(y_train.flatten()))
            scr = r2_score(np.sqrt(y_test), lr.predict(X_test))
            if scr > 0:
                r2_quadratic.append(scr)

        # https://stackoverflow.com/questions/3433486/how-to-do-exponential-and-logarithmic-curve-fitting-in-python-i-found-only-poly
        for _ in range(nr_reruns):
            lr = LinearRegression()
            lr.fit(X_train, np.log(y_train))

            scr = r2_score(np.log(y_test), lr.predict(X_test))
            if scr > 0:
                r2_exp.append(scr)

    return np.argmax([np.mean(r2_linear), np.mean(r2_quadratic), np.mean(r2_exp)])


def encode(value_list):
    out = []

    to_skip = 0
    for idx in range(len(value_list)):

        # Check if we have to skip repeats
        if to_skip > 0:
            to_skip = to_skip - 1
            continue

        val = value_list[idx]
        reps = 1
        # Count repeats of val
        for j in range(idx + 1, len(value_list)):
            if value_list[j] == val:
                reps += 1
            else:
                break

        to_skip = reps - 1

        out.append(val)
        out.append(reps)

    return out


def decode(value_list):
    out = []
    for i in range(0, len(value_list), 2):
        val = value_list[i]
        reps = value_list[i + 1]
        out.extend([val] * reps)
    return out


def KL(p, q):
    res = 0
    for i in range(min(len(p), len(q))):
        res = res + p[i] * (np.log(p[i]) - np.log(q[i]))

    return res
