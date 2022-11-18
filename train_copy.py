"""Train paa and get checkpoint files."""

import os
import ast
import mindspore.nn as nn
from mindspore import context, Tensor
from mindspore.communication.management import init, get_rank
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint, LossMonitor, TimeMonitor, Callback
from mindspore.train import Model
from mindspore.context import ParallelMode
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.common import set_seed
from src.paa import paaWithLossCell, TrainingWrapper,PaaResNet101Fpn,PaaResNet50Fpn,paaInferWithDecoder,paaR50,resnet50
from src.dataset import create_paa_dataset, create_mindrecord
from src.lr_schedule import get_lr
from src.init_params import init_net_param, filter_checkpoint_parameter
from src.model_utils.config import config
from src.model_utils.device_adapter import get_device_id, get_device_num
from src.box_utils import default_boxes
from src.eval_utils import apply_eval
from src.eval_callback import EvalCallBack

os.environ['CUDA_VISIBLE_DEVICES']='1'
set_seed(1)

def paa_model_build():
    if config.model_name == "paa_resnet50_fpn":
        paa = PaaResNet50Fpn(config=config)
    
    elif config.model_name == "paa_resnet101_fpn":
        paa = PaaResNet101Fpn(config=config)
    else:
        raise ValueError(f'config.model: {config.model_name} is not supported')
    return paa

class Monitor(Callback):
    """
    Monitor loss and time.

    Args:
        lr_init (numpy array): train lr

    Returns:
        None

    Examples:
        >>> Monitor(100,lr_init=Tensor([0.05]*100).asnumpy())
    """

    def __init__(self, lr_init=None):
        super(Monitor, self).__init__()
        self.lr_init = lr_init
        self.lr_init_len = len(lr_init)

    def step_end(self, run_context):
        cb_params = run_context.original_args()
        print("lr:[{:8.6f}]".format(self.lr_init[cb_params.cur_step_num - 1]), flush=True)


def set_graph_kernel_context(device_target):
    if device_target == "GPU":
        # Enable graph kernel for default model on GPU back-end.
        context.set_context(enable_graph_kernel=True,
                            graph_kernel_flags="--enable_parallel_fusion --enable_expand_ops=Conv2D")


def main():
    #查看学习率是否有效
    config.lr_init = ast.literal_eval(config.lr_init)
    config.lr_end_rate = ast.literal_eval(config.lr_end_rate)
    #获取设备id
    device_id = get_device_id()
    #设置gpu环境环境
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    if config.device_target == "Ascend":
        if context.get_context("mode") == context.PYNATIVE_MODE:
            context.set_context(mempool_block_size="31GB")
    elif config.device_target == "GPU":
        set_graph_kernel_context(config.device_target)
    elif config.device_target == "CPU":
        device_id = 0
        config.distribute = False
    else:
        raise ValueError(f"device_target support ['Ascend', 'GPU', 'CPU'], but get {config.device_target}")
    if config.distribute:
        init()
        device_num = get_device_num()
        rank = get_rank()
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True)
    else:
        rank = 0
        device_num = 1
        context.set_context(device_id=device_id)

    
    mindrecord_file = create_mindrecord(config.dataset, "paa.mindrecord", True)
    loss_scale = float(config.loss_scale)

    # When create MindDataset, using the first mindrecord file, such as paa.mindrecord0.
    dataset = create_paa_dataset(mindrecord_file, repeat_num=1,
                                       num_parallel_workers=config.workers,
                                       batch_size=config.batch_size, device_num=device_num, rank=rank)
    dataset_size = dataset.get_dataset_size()
    print(f"Create dataset done! dataset size is {dataset_size}")
    
    # paa = paa_model_build()
    backbone = resnet50(config.num_classes)
    paa = paaR50(backbone, config)
    net = paaWithLossCell(paa, config)
    init_net_param(net)
    # if config.pre_trained:
    #     if config.pre_trained_epoch_size <= 0:
    #         raise KeyError("pre_trained_epoch_size must be greater than 0.")
    #     param_dict = load_checkpoint(config.pre_trained)
    #     print(config.pre_trained)
    #     if config.filter_weight:
    #         filter_checkpoint_parameter(param_dict)
    #     load_param_into_net(net, param_dict)
    #     print(param_dict)

    lr = Tensor(get_lr(global_step=0,
                       lr_init=config.lr_init, lr_end=config.lr_end_rate * config.lr, lr_max=config.lr,
                       warmup_epochs1=config.warmup_epochs1, warmup_epochs2=config.warmup_epochs2,
                       warmup_epochs3=config.warmup_epochs3, warmup_epochs4=config.warmup_epochs4,
                       warmup_epochs5=config.warmup_epochs5, total_epochs=config.epoch_size,
                       steps_per_epoch=dataset_size))
    opt = nn.Momentum(filter(lambda x: x.requires_grad, net.get_parameters()), lr,
                      config.momentum, config.weight_decay, loss_scale)
    # opt = nn.SGD(params=net.trainable_params(), learning_rate=0.01,
    #              momentum=0.99, weight_decay=5e-4)

    net = TrainingWrapper(net, opt, loss_scale)
    model = Model(net)
    print("Start train paa, the first epoch will be slower because of the graph compilation.")
    cb = [TimeMonitor(), LossMonitor()]
    cb += [Monitor(lr_init=lr.asnumpy())]
    config_ck = CheckpointConfig(save_checkpoint_steps=dataset_size * config.save_checkpoint_epochs,
                                 keep_checkpoint_max=config.keep_checkpoint_max)
    ckpt_cb = ModelCheckpoint(prefix="paa", directory=config.save_checkpoint_path, config=config_ck)
    
    #eval函数
    # ckpt_save_dir = config.output_path +'/ckpt_{}/'.format(rank)
    # callback = [TimeMonitor(data_size=dataset_size), LossMonitor(), ckpt_cb]
    # if config.run_eval:
    #     print("dddd")
    #     eval_net = paaInferWithDecoder(paa, Tensor(default_boxes), config)
    #     eval_net.set_train(False)
    #     mindrecord_file = create_mindrecord(config.dataset, "paa_eval.mindrecord", False)
    #     eval_dataset = create_paa_dataset(mindrecord_file, repeat_num=1,
    #                                    num_parallel_workers=config.workers,
    #                                    batch_size=config.batch_size, device_num=device_num, rank=rank)
    #     if config.dataset == "coco":
    #         anno_json = os.path.join(config.coco_root, config.instances_set.format(config.val_data_type))
    #     elif config.dataset == "voc":
    #         anno_json = os.path.join(config.voc_root, config.voc_json)
    #     else:
    #         raise ValueError('PAA eval only support dataset mode is coco and voc!')
    #     eval_param_dict = {"net": eval_net, "dataset": eval_dataset, "anno_json": anno_json}
    #     eval_cb = EvalCallBack(apply_eval, eval_param_dict, interval=config.eval_interval,
    #                            eval_start_epoch=config.eval_start_epoch, save_best_ckpt=True,
    #                            ckpt_directory=ckpt_save_dir, besk_ckpt_name="best_map.ckpt",
    #                            metrics_name="mAP")
    #     callback.append(eval_cb)
        
        
    if config.distribute:
        if rank == 0:
            cb += [ckpt_cb]
        model.train(config.epoch_size, dataset, callbacks=cb, dataset_sink_mode=True)
    else:
        cb += [ckpt_cb]
        model.train(config.epoch_size, dataset, callbacks=cb, dataset_sink_mode=True)

    # ckpt_config = CheckpointConfig(save_checkpoint_steps=dataset_size * config.save_checkpoint_epochs)
    # ckpt_save_dir = config.output_path +'/ckpt_{}/'.format(rank)
    # print(f"Create ckpt_save_dir done! ckpt_save_dir  is {ckpt_save_dir}")
    # ckpoint_cb = ModelCheckpoint(prefix="paa", directory=ckpt_save_dir, config=ckpt_config)
    # callback = [TimeMonitor(data_size=dataset_size), LossMonitor(), ckpoint_cb]
    # if config.run_eval:
    #     print("dddd")
    #     eval_net = paaInferWithDecoder(paa, Tensor(default_boxes), config)
    #     eval_net.set_train(False)
    #     mindrecord_file = create_mindrecord(config.dataset, "paa_eval.mindrecord", False)
    #     eval_dataset = create_paa_dataset(mindrecord_file, repeat_num=1,
    #                                    num_parallel_workers=config.workers,
    #                                    batch_size=config.batch_size, device_num=device_num, rank=rank)
    #     if config.dataset == "coco":
    #         anno_json = os.path.join(config.coco_root, config.instances_set.format(config.val_data_type))
    #     elif config.dataset == "voc":
    #         anno_json = os.path.join(config.voc_root, config.voc_json)
    #     else:
    #         raise ValueError('PAA eval only support dataset mode is coco and voc!')
    #     eval_param_dict = {"net": eval_net, "dataset": eval_dataset, "anno_json": anno_json}
    #     eval_cb = EvalCallBack(apply_eval, eval_param_dict, interval=config.eval_interval,
    #                            eval_start_epoch=config.eval_start_epoch, save_best_ckpt=True,
    #                            ckpt_directory=ckpt_save_dir, besk_ckpt_name="best_map.ckpt",
    #                            metrics_name="mAP")
    #     callback.append(eval_cb)
    # model = Model(net)
    # dataset_sink_mode = False
    # if config.device_target != "CPU":
    #     print("In sink mode, one epoch return a loss.")
    #     dataset_sink_mode = True
    # print("Start train PAA, the first epoch will be slower because of the graph compilation.")
    # model.train(config.epoch_size, dataset, callbacks=callback, dataset_sink_mode=dataset_sink_mode)
    
if __name__ == '__main__':
    main()
