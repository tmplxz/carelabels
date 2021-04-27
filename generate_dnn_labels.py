from mlcl.carelabels import ModelCareLabel
from mlcl.carelabels.knowledge_database import default_scales
from mlcl.util import check_scale

import pandas as pd

models = ['alexnet', 'vgg11', 'mobilenet_v3_large', 'resnet18']
cl_names = ['AlexNet', 'VGG11', 'MobileNetV3_Large', 'ResNet-18']
descriptions = [
    'Conv neural network, winner of 2012 ImageNet challenge',
    'Very deep convoulational neural network',
    'Convolutional neural network for mobile phones',
    'Convolutional neural network with 18 layers'
]

df = pd.read_csv('results/dnn_results.csv')

for model, name, descr in zip(models, cl_names, descriptions):
    df_row = df[df['model.name'] == model]

    benchmark_cpu = {
        'name': 'ImageNet',
        'runtime': df_row['cpu_latency_i7-10610'].item(),
        'runtime_rating': check_scale(df_row['cpu_latency_i7-10610'].item(), [0.1, 0.07, 0.04]),
        'accuracy': df_row['acc@top1'].item(),
        'accuracy_rating': check_scale(df_row['acc@top1'].item(), default_scales['accuracy']),
        'energy': df_row['cpu_watt_seconds_i7-10610'].item(),
        'energy_rating': check_scale(df_row['cpu_watt_seconds_i7-10610'].item(), [2, 1, 0.5]),
        'memory': df_row['host_memory_bytes_max_mean_i7-10610'].item(),
        'memory_rating': check_scale(df_row['host_memory_bytes_max_mean_i7-10610'].item(), [734e6, 629e6, 524e6]),

        'gpu_memory': 0
    }

    benchmark_gpu = {
        'name': 'ImageNet',
        'runtime': df_row['gpu_latency_a100'].item(),
        'runtime_rating': check_scale(df_row['gpu_latency_a100'].item(), [0.01, 0.007, 0.003]),
        'accuracy': df_row['acc@top1'].item(),
        'accuracy_rating': check_scale(df_row['acc@top1'].item(), default_scales['accuracy']),
        'energy': df_row['gpu_energy_watt_seconds_a100'].item(),
        'energy_rating': check_scale(df_row['gpu_energy_watt_seconds_a100'].item(), [0.6, 0.3, 0.1]),
        'memory': df_row['device_memory_bytes_max_mean_a100'].item(),
        'memory_rating': check_scale(df_row['device_memory_bytes_max_mean_a100'].item(), [2100e6, 2080e6, 2050e6]),

        'gpu_memory': 0
    }

    exec_info_cpu = {
        'platform': 'CPU (Intel Core i7-10610U)',
        'software': 'Python, PyTorch'
    }

    exec_info_gpu = {
        'platform': 'GPU (NVIDIA A100)',
        'software': 'Python, PyTorch'
    }

    platforms = ['cpu', 'gpu']
    benchmarks = [benchmark_cpu, benchmark_gpu]
    exec_infos = [exec_info_cpu, exec_info_gpu]

    for plat, bench, exec_info in zip(platforms, benchmarks, exec_infos):

        info = {
            'Accuracy': df_row['acc@top1_static'].item(),
            'Relative_memory': df_row['params'].item(),
            'Relative_runtime': df_row['flops'].item(),
            'Train_energy': df_row['traing_energy_consumption_kwh'].item(),
            'Name': name,
            'Description': descr,
        }
        for key, value in info.items():
            if key.lower() in default_scales:
                info[key] = [value, check_scale(value, default_scales[key.lower()])]

        info['Software tests'] = {
            'corruptiontest': {'rating': check_scale(df_row['mce@top1'].item(), default_scales['corruptiontest'])},
            'perturbationtest': {'rating': check_scale(df_row['mean_flip_probability'].item(), default_scales['perturbationtest'])},
            'noisetest': {'rating': check_scale(df_row['eps'].item(), default_scales['noisetest'])}
        }

        info['benchmark'] = bench
        info['Execution information'] = exec_info

        if df_row['params'].item() < 10e6:
            info['badges'] = ['Suitable for edge devices?']
        else:
            info['badges'] = []
    	
        outname = f'label_{model}_{plat}.svg'

        cl = ModelCareLabel(info)
        cl.to_image(outname)
