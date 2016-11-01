import random
import os
os.environ['GLOG_minloglevel'] = '1'
import caffe
import time
import numpy as np
import os
import HPOlib.benchmark_util as benchmark_util
import HPOlib.wrapping_util as wrapping_util
import sys
import pylrpredictor.terminationcriterion as termcrit

#Globals
base_lr = 1.0
weight_decay= 1.0
weight_param = dict(lr_mult=1, decay_mult=1)
bias_param   = dict(lr_mult=2, decay_mult=0)
learned_param = [weight_param, bias_param]

#class Logger(object):
#    def __init__(self,dir):
#        self.terminal = sys.stdout
#        self.log = open(dir+"/hyperband_run.log", "a")

#    def write(self, message):
#        self.terminal.write(message)
#        self.log.write(message)
#        self.log.flush()


def build_net(arm,split,problem):
    if problem=='cifar10':
        n_class = 10
        data_dir = '/home/lisha/school/HPOlib/benchmarks/networkin/cifar10'
        data_dirs=[data_dir+'/cifar-zca-ptrain', data_dir+'/cifar-zca-pval',data_dir+'/cifar-zca-test']
        backend = caffe.params.Data.LMDB
    else:
        n_class = 100
        data_dir = '/home/lisha/school/HPOlib/benchmarks/networkin/cifar100'
        data_dirs=[data_dir+'/cifar100_ptrain_lmdb', data_dir+'/cifar100_pval_lmdb',data_dir+'/cifar100_test_lmdb']
        backend = caffe.params.Data.LMDB
    def conv_layer(bottom, ks=5, nout=32, stride=1, pad=2, group=0,param=learned_param,
                  weight_filler=dict(type='gaussian', std=0.0001),
                  bias_filler=dict(type='constant')):
        if group==0:
            conv = caffe.layers.Convolution(bottom, kernel_size=ks, stride=stride,
                             num_output=nout, pad=pad, param=param, weight_filler=weight_filler,
                             bias_filler=bias_filler)
        else:
            conv = caffe.layers.Convolution(bottom, kernel_size=ks, stride=stride,
                             num_output=nout, pad=pad, group=group, param=param, weight_filler=weight_filler,
                             bias_filler=bias_filler)
        return conv

    def pooling_layer(bottom, type='ave', ks=3, stride=2):
        if type=='ave':
            return caffe.layers.Pooling(bottom, pool=caffe.params.Pooling.AVE, kernel_size=ks, stride=stride,engine=1)
        return caffe.layers.Pooling(bottom, pool=caffe.params.Pooling.MAX, kernel_size=ks, stride=stride,engine=1)

    n = caffe.NetSpec()
    if split==1:
        n.data, n.label = caffe.layers.Data(batch_size=128, backend=backend, source=data_dirs[0],ntop=2)
    elif split==2:
        n.data, n.label = caffe.layers.Data(batch_size=100, backend=backend, source=data_dirs[1],ntop=2)
    elif split==3:
        n.data, n.label = caffe.layers.Data(batch_size=100, backend=backend, source=data_dirs[2],ntop=2)
    param1=[dict(lr_mult=arm['learning_rate1'],decay_mult=arm['weight_cost1']),dict(lr_mult=2*arm['learning_rate1'],decay_mult=2*arm['weight_cost1'])]
    param2=[dict(lr_mult=arm['learning_rate2'],decay_mult=arm['weight_cost2']),dict(lr_mult=2*arm['learning_rate2'],decay_mult=2*arm['weight_cost2'])]
    param3=[dict(lr_mult=arm['learning_rate3'],decay_mult=arm['weight_cost3']),dict(lr_mult=2*arm['learning_rate3'],decay_mult=2*arm['weight_cost3'])]
    n.conv1a = conv_layer(n.data, ks=5,nout=192,stride=1,pad=2,group=0,param=param1,
        weight_filler=dict(type='gaussian', std=arm['init_std1']), bias_filler=dict(type='constant'))
    n.relu1a = caffe.layers.ReLU(n.conv1a,in_place=True)
    n.conv1b = conv_layer(n.conv1a, ks=1,nout=160,stride=1,pad=0,group=1,param=param1,
        weight_filler=dict(type='gaussian', std=arm['init_std1']), bias_filler=dict(type='constant'))
    n.relu1b = caffe.layers.ReLU(n.conv1b,in_place=True)
    n.conv1c = conv_layer(n.conv1b, ks=1,nout=96,stride=1,pad=0,group=1,param=param1,
        weight_filler=dict(type='gaussian', std=arm['init_std1']), bias_filler=dict(type='constant'))
    n.relu1c = caffe.layers.ReLU(n.conv1c, in_place=True)
    n.pool1 = pooling_layer(n.conv1c,type='max',ks=3,stride=2)
    n.dropout1 = caffe.layers.Dropout(n.pool1,dropout_ratio=arm['dropout1'],in_place=True)
    n.conv2a = conv_layer(n.pool1, ks=5,nout=192,stride=1,pad=2,group=0,param=param2,
        weight_filler=dict(type='gaussian', std=arm['init_std2']), bias_filler=dict(type='constant'))
    n.relu2a = caffe.layers.ReLU(n.conv2a,in_place=True)
    n.conv2b = conv_layer(n.conv2a, ks=1,nout=192,stride=1,pad=0,group=1,param=param2,
        weight_filler=dict(type='gaussian', std=arm['init_std2']), bias_filler=dict(type='constant'))
    n.relu2b = caffe.layers.ReLU(n.conv2b,in_place=True)
    n.conv2c = conv_layer(n.conv2b, ks=1,nout=192,stride=1,pad=0,group=1,param=param2,
        weight_filler=dict(type='gaussian', std=arm['init_std2']), bias_filler=dict(type='constant'))
    n.relu2c = caffe.layers.ReLU(n.conv2c, in_place=True)
    n.pool2 = pooling_layer(n.conv2c,type='max',ks=3,stride=2)
    n.dropout2 = caffe.layers.Dropout(n.pool2,dropout_ratio=arm['dropout2'],in_place=True)
    n.conv3a = conv_layer(n.pool2, ks=3,nout=192,stride=1,pad=1,group=0,param=param3,
        weight_filler=dict(type='gaussian', std=arm['init_std3']), bias_filler=dict(type='constant'))
    n.relu3a = caffe.layers.ReLU(n.conv3a,in_place=True)
    n.conv3b = conv_layer(n.conv3a, ks=1,nout=192,stride=1,pad=0,group=1,param=param3,
        weight_filler=dict(type='gaussian', std=arm['init_std3']), bias_filler=dict(type='constant'))
    n.relu3b = caffe.layers.ReLU(n.conv3b,in_place=True)
    n.conv3c = conv_layer(n.conv3b, ks=1,nout=n_class,stride=1,pad=0,group=1,param=param3,
        weight_filler=dict(type='gaussian', std=arm['init_std3']), bias_filler=dict(type='constant'))
    n.relu3c = caffe.layers.ReLU(n.conv3c, in_place=True)
    n.pool3 = pooling_layer(n.conv3c,type='ave',ks=8,stride=1)
    n.loss = caffe.layers.SoftmaxWithLoss(n.pool3,n.label)

    
    if split==1:
        filename=arm['dir']+'/network_train.prototxt'
    elif split==2:
        n.acc = caffe.layers.Accuracy(n.pool3, n.label)
        filename=arm['dir']+'/network_val.prototxt'
    elif split==3:
        n.acc = caffe.layers.Accuracy(n.pool3, n.label)
        filename=arm['dir']+'/network_test.prototxt'
    with open(filename,'w') as f:
        f.write(str(n.to_proto()))
        return f.name
def build_solver(arm,problem):
    s = caffe.proto.caffe_pb2.SolverParameter()

    # Specify locations of the train and (maybe) test networks.
    s.train_net = arm['train_net_file']
    s.test_net.append(arm['val_net_file'])
    s.test_net.append(arm['test_net_file'])
    s.test_interval = 60000  # Test after every 1000 training iterations.
    s.test_iter.append(int(10000/arm['batch_size'])) # Test on 100 batches each time we test.
    s.test_iter.append(int(10000/arm['batch_size'])) # Test on 100 batches each time we test.

    # The number of iterations over which to average the gradient.
    # Effectively boosts the training batch size by the given factor, without
    # affecting memory utilization.
    s.iter_size = 1

    # 150 epochs max
    s.max_iter = 60000     # # of times to update the net (training iterations)

    # Solve using the stochastic gradient descent (SGD) algorithm.
    # Other choices include 'Adam' and 'RMSProp'.
    #s.type = 'SGD'

    # Set the initial learning rate for SGD.
    s.base_lr = base_lr

    # Set `lr_policy` to define how the learning rate changes during training.
    # Here, we 'step' the learning rate by multiplying it by a factor `gamma`
    # every `stepsize` iterations.
    s.lr_policy = 'multistep'
    s.gamma = 0.1
    s.stepsize=s.max_iter/arm['lr_step']
    s.stepvalue.append(40000)
    s.stepvalue.append(50000)
    s.stepvalue.append(60000)

    # Set other SGD hyperparameters. Setting a non-zero `momentum` takes a
    # weighted average of the current gradient and previous gradients to make
    # learning more stable. L2 weight decay regularizes learning, to help prevent
    # the model from overfitting.
    s.momentum = arm['momentum']
    s.weight_decay = weight_decay

    # Display the current training loss and accuracy every 1000 iterations.
    s.display = 1000

    # Snapshots are files used to store networks we've trained.  Here, we'll
    # snapshot every 10K iterations -- ten times during training.
    s.snapshot = 10000
    s.snapshot_prefix = arm['dir']+"/"+problem+"cifar10_data"

    # Train on the GPU.  Using the CPU to train large networks is very slow.
    s.solver_mode = caffe.proto.caffe_pb2.SolverParameter.GPU

    # Write the solver to a temporary file and return its filename.
    filename=arm['dir']+"/network_solver.prototxt"
    with open(filename,'w') as f:
        f.write(str(s))
        return f.name
def generate_arm(params,dir,problem):
    os.chdir(dir)
    if params is not None:
        for key in params:
            try:
                params[key] = int(params[key])
            except Exception:
                try:
                    params[key] = float(params[key])
                    if params[key]-np.floor(params[key])==0:
                        params[key]=int(params[key])
                except Exception:
                    pass
    subdirs=next(os.walk('.'))[1]
    if len(subdirs)==0:
        start_count=0
    else:
        start_count=len(subdirs)

    dirname="arm"+str(start_count)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    arm={}
    arm['dir']=dir+"/"+dirname
    arm['n_iter']=0
    arm['momentum']=params['momentum']
    arm['learning_rate1']=params['learning_rate1']
    arm['learning_rate2']=params['learning_rate2']
    arm['learning_rate3']=params['learning_rate3']
    arm['weight_cost1']=params['weight_cost1']
    arm['weight_cost2']=params['weight_cost2']
    arm['weight_cost3']=params['weight_cost3']
    arm['batch_size']=100
    arm['lr_step']=params['lr_step']
    arm['dropout1']=params['dropout1']
    arm['dropout2']=params['dropout2']
    arm['init_std1']=params['w_init1']
    arm['init_std2']=params['w_init2']
    arm['init_std3']=params['w_init3']
    arm['train_net_file'] = build_net(arm,1,problem)
    arm['val_net_file'] = build_net(arm,2,problem)
    arm['test_net_file'] = build_net(arm,3,problem)
    arm['solver_file'] = build_solver(arm,problem)
    arm['results']=[]
    return arm

def run_solver(unit, n_units, arm, val_batch, test_batch, do_stop=False):
    s = caffe.get_solver(arm['solver_file'])

    start=time.time()
    time_val=0
    time_early=0
    if unit=='time':
        while time.time()-start<n_units:
            s.step(1)
            arm['n_iter']+=1

    elif unit=='iter':
        if do_stop:
            early_stop = 0
            while not early_stop and arm['n_iter']<60000:
                s.step(400)
                arm['n_iter']+=400
                val_acc=0
                st=time.time()
                for i in range(val_batch):
                    s.test_nets[0].forward()
                    val_acc += s.test_nets[0].blobs['acc'].data
                val_acc=val_acc/val_batch
                time_val+=time.time()-st
                if os.path.exists("learning_curve.txt"):
                    with open("learning_curve.txt", "a") as myfile:
                        myfile.write("\n")
                        myfile.write(str(val_acc))
                else:
                    with open("learning_curve.txt", "w") as myfile:
                        myfile.write(str(val_acc))
                if arm['n_iter']%8000==0:
                    st=time.time()
                    early_stop=check_early_stopping(150)
                    time_early+=time.time()-st
        else:
            s.step(n_units)
            arm['n_iter']+=n_units

    s.snapshot()
    train_loss = s.net.blobs['loss'].data
    val_acc=0
    test_acc=0

    if do_stop:
        if early_stop:
            p_file=open("y_predict.txt",'r')
            val_acc=float(p_file.readline())
            p_file.close()
            os.remove("learning_curve.txt")
            with open("run_log.txt", "a") as myfile:
                myfile.write("val_error," + str(val_acc) + ",test_error," + str(test_acc)+","+"epochs," +str(arm['n_iter']/400)+","+"val_time,"+str(time_val/60)+","+"early_time,"+str(time_early/60)+"\n")
            return train_loss,val_acc,test_acc

    for i in range(val_batch):
        s.test_nets[0].forward()
        val_acc += s.test_nets[0].blobs['acc'].data
    for i in range(test_batch):
        s.test_nets[1].forward()
        test_acc += s.test_nets[1].blobs['acc'].data
    val_acc=val_acc/val_batch
    test_acc=test_acc/test_batch

    if do_stop:
        best_val=0
        if os.path.exists("ybest.txt"):
            fh=open("ybest.txt", "r")
            for line in fh:
                pass
            best_val= float(line.strip())
        best_val=max(val_acc,best_val)
        with open("ybest.txt", "w") as myfile:
            myfile.write(str(best_val))
        os.remove("learning_curve.txt")
        with open("run_log.txt", "a") as myfile:
            myfile.write("val_error," + str(val_acc) + ",test_error," + str(test_acc)+","+"epochs," +str(arm['n_iter']/400)+","+"val_time,"+str(time_val/60)+","+"early_time,"+str(time_early/60)+"\n")
    return train_loss,val_acc, test_acc
def check_early_stopping(max_iter):
    return termcrit.main(mode="conservative",
                prob_x_greater_type="posterior_prob_x_greater_than",
                nthreads=4)
def main(params, dir,do_stop,problem):
    arm = generate_arm(params,dir,problem)
    print arm
    train_loss,val_acc, test_acc = run_solver('iter',60000,arm,100,100,do_stop)
    return train_loss, val_acc, test_acc

if __name__ == "__main__":
    starttime = time.time()
    experiment_dir = os.getcwd()
    #sys.stdout = Logger(experiment_dir)
    args, params = benchmark_util.parse_cli()
    config = wrapping_util.load_experiment_config_file()
    device= config.get("EXPERIMENT", "device")
    do_stop= config.get("EXPERIMENT", "do_stop")
    problem = config.get("EXPERIMENT","problem")
    caffe.set_device(int(device))
    caffe.set_mode_gpu()
    train_loss, val_acc, test_acc = main(params, experiment_dir,int(do_stop),problem)
    val_error=1-val_acc
    test_error = 1-test_acc
    duration = time.time() - starttime
    print "Result for ParamILS: %s, %f, 1, %f, %d, %s, %f" % \
        ("SAT", abs(duration), val_error, -1, "test_error", test_error)
