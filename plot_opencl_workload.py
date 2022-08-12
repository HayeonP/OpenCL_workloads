import matplotlib.pyplot as plt
import csv

value_names=['TOTAL_RESPONSE_TIME','CREATE_BUFFER','WRITE_BUFFER','CREATE_KERNEL_ARGS','CREATE_KERNEL','LAUNCH_KERNEL','READ_BUFFER','KERNEL_EXECUTION_TIME']

def get_column(input, column):
    output = []
    for line in input:
        output.append(line[column])
    return output

def plot_opencl_workload(data):
    raw_data = []
    figname = data.split('/')[-1].split('.')[0]+'.png'

    with open(data) as f:
        reader = csv.reader(f)
        for i, line in enumerate(reader):
            if i == 0: continue
            raw_data.append([float(v) for v in line])
    
    raw_data=raw_data[10:] # Cold start
    
    x_data = range(len(raw_data))
    for i, name in enumerate(value_names):
        plt.plot(x_data, get_column(raw_data,i), label=name)
    plt.legend()
    plt.xlabel('Iteration')
    plt.ylabel('Response time(s)')
    plt.ylim([0.0, 0.6])
    plt.savefig('./figures/'+figname)    
    
    plt.close()



if __name__ == '__main__':
    data = '/Users/hayeonp/git/OpenCL_workloads/core1.csv'
    plot_opencl_workload(data)
    data = '/Users/hayeonp/git/OpenCL_workloads/core2.csv'
    plot_opencl_workload(data)
    data = '/Users/hayeonp/git/OpenCL_workloads/single.csv'
    plot_opencl_workload(data)